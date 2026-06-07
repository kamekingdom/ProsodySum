#!/usr/bin/env python3
"""Transcribe wav files in data/audio into text files in data/text."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
import torch
from transformers import pipeline


DEFAULT_MODEL_ID = "openai/whisper-large-v3-turbo"
DEFAULT_DIARIZATION_MODEL_ID = "pyannote/speaker-diarization-3.1"
TARGET_SAMPLE_RATE = 16_000
SUPPORTED_EXTENSIONS = {".wav"}
LOCAL_DIARIZATION_MIN_ENERGY_PERCENTILE = 35
EXPLICIT_SENTENCE_ENDINGS = "。！？!?"
SENTENCE_END_RE = re.compile(r"(?<=[。！？!?])\s*")
LLM_SENTENCE_SPLIT_CHARS = 1_200
QUESTION_ENDINGS = ("?", "？", "か", "か。", "ですか", "ですか。", "ますか", "ますか。", "でしょうか", "でしょうか。")
SPEAKER_MARKERS = (
    ("高市さなえ内閣総理大臣", "高市さなえ"),
    ("高市内閣総理大臣", "高市さなえ"),
    ("小泉信二郎防衛大臣", "小泉信二郎"),
    ("小泉進次郎防衛大臣", "小泉進次郎"),
    ("片山さつき財務大臣", "片山さつき"),
    ("片山札樹財務大臣", "片山さつき"),
    ("片山沙月財務大臣", "片山さつき"),
    ("片山五月財務大臣", "片山さつき"),
    ("赤澤経産大臣", "赤澤経産大臣"),
    ("赤沢経産大臣", "赤澤経産大臣"),
    ("奥田文夫さん", "奥田文夫"),
    ("委員長", "委員長"),
)
CHAIR_KEYWORDS = (
    "質疑は終了",
    "次に",
    "質疑を行います",
    "しばらくお待ちください",
    "発言の前に",
    "不適切",
    "理事会",
    "速記",
    "調査の上",
    "処置をとること",
    "処置ををとること",
)
ANSWER_PREFIXES = ("はい", "いいえ", "そうです", "その通り", "違います", "お答えします", "回答します")
CONCERN_KEYWORDS = (
    "懸念",
    "問題",
    "課題",
    "異常",
    "苦しい",
    "貧困",
    "危機",
    "恐ろしい",
    "悪い",
    "最悪",
    "破壊",
    "奪う",
    "できない",
    "下がらない",
    "値上がり",
)
DECISION_KEYWORDS = ("決定", "決め", "合意", "結論", "承認", "採択", "廃止され", "行うことといたします")
ACTION_ITEM_KEYWORDS = (
    "対応",
    "提出",
    "検討",
    "取り組",
    "目指",
    "お願いします",
    "してください",
    "必要があります",
    "行います",
    "まいります",
)
POSITIVE_KEYWORDS = ("賛成", "よろしい", "ありがとうございます", "改善", "実現", "協力", "守る", "できる")
NEGATIVE_KEYWORDS = CONCERN_KEYWORDS + ("反対", "否定", "疑問", "不適切")
LOW_IMPORTANCE_KEYWORDS = ("ご視聴ありがとうございました", "以上で", "速記を止めてください", "しばらくお待ちください")


@dataclass(frozen=True)
class Utterance:
    text: str
    start: float | None = None
    end: float | None = None
    speaker: str | None = None


@dataclass(frozen=True)
class Labels:
    act: str
    stance: str
    importance: str


@dataclass(frozen=True)
class DiarizationSegment:
    start: float
    end: float
    speaker: str


def default_data_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def audio_files(audio_dir: Path) -> Iterable[Path]:
    for path in sorted(audio_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def text_files(text_dir: Path, labeled_suffix: str) -> Iterable[Path]:
    for path in sorted(text_dir.iterdir()):
        if path.is_file() and path.suffix.lower() == ".txt" and not path.name.endswith(f"{labeled_suffix}.txt"):
            yield path


def json_files(text_dir: Path) -> Iterable[Path]:
    for path in sorted(text_dir.iterdir()):
        if path.is_file() and path.suffix.lower() == ".json":
            yield path


def load_wav_for_whisper(path: Path) -> dict[str, np.ndarray | int]:
    sample_rate, audio = wavfile.read(path)

    if np.issubdtype(audio.dtype, np.integer):
        max_value = np.iinfo(audio.dtype).max
        audio = audio.astype(np.float32) / max_value
    else:
        audio = audio.astype(np.float32)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sample_rate != TARGET_SAMPLE_RATE:
        gcd = np.gcd(sample_rate, TARGET_SAMPLE_RATE)
        audio = resample_poly(audio, TARGET_SAMPLE_RATE // gcd, sample_rate // gcd).astype(np.float32)
        sample_rate = TARGET_SAMPLE_RATE

    return {"array": audio, "sampling_rate": sample_rate}


def split_long_text(text: str, max_chars: int) -> list[str]:
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]

    parts = []
    current = ""
    for token in text.split():
        if current and len(current) + 1 + len(token) > max_chars:
            parts.append(current)
            current = token
        elif current:
            current = f"{current} {token}"
        else:
            current = token

    if current:
        parts.append(current)

    if not parts:
        return [text[index : index + max_chars] for index in range(0, len(text), max_chars)]

    expanded = []
    for part in parts:
        if len(part) <= max_chars:
            expanded.append(part)
        else:
            expanded.extend(part[index : index + max_chars] for index in range(0, len(part), max_chars))
    return expanded


def split_long_utterance(text: str, max_chars: int, start: float | None = None, end: float | None = None) -> list[Utterance]:
    parts = split_long_text(text, max_chars)
    if start is None or end is None or len(parts) <= 1 or end <= start:
        return [Utterance(part, start, end) for part in parts]

    total_chars = sum(len(part) for part in parts)
    if total_chars <= 0:
        return [Utterance(part, start, end) for part in parts]

    duration = end - start
    elapsed_chars = 0
    utterances = []
    for part in parts:
        part_start = start + duration * (elapsed_chars / total_chars)
        elapsed_chars += len(part)
        part_end = start + duration * (elapsed_chars / total_chars)
        utterances.append(Utterance(part, part_start, part_end))
    return utterances


def split_sentence_spans(text: str) -> list[tuple[str, int, int]]:
    stripped = text.strip()
    if not stripped:
        return []

    offset = text.find(stripped)
    sentences = [part.strip() for part in SENTENCE_END_RE.split(stripped) if part.strip()]
    if not sentences:
        sentences = [stripped]

    spans = []
    cursor = 0
    for sentence in sentences:
        start = stripped.find(sentence, cursor)
        if start < 0:
            start = cursor
        end = start + len(sentence)
        spans.append((sentence, offset + start, offset + end))
        cursor = end
    return spans


def split_sentences(text: str, max_chars: int) -> list[str]:
    del max_chars
    return [sentence for sentence, _, _ in split_sentence_spans(text)]


def split_sentence_utterances(text: str, max_chars: int) -> list[Utterance]:
    return [Utterance(sentence) for sentence in split_sentences(text, max_chars)]


def parse_json_array(text: str) -> list:
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end < start:
        raise ValueError("No JSON array found in model output.")
    return json.loads(text[start : end + 1])


def split_sentences_with_llm(sentence_splitter, text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []

    messages = [
        {
            "role": "system",
            "content": (
                "Reasoning: low\n"
                "Split the Japanese transcript into natural complete sentences. "
                "Return only a compact JSON array of strings. "
                "Do not explain. Do not summarize. Do not add labels. "
                "Do not rewrite, translate, normalize, or delete words. "
                "Keep each string as an exact contiguous substring from the input whenever possible."
            ),
        },
        {"role": "user", "content": stripped},
    ]
    max_new_tokens = min(max(len(stripped) // 2 + 256, 512), 4096)
    output = sentence_splitter(messages, max_new_tokens=max_new_tokens, do_sample=False)
    generated = output[0].get("generated_text", output[0])
    if isinstance(generated, list):
        generated = generated[-1].get("content", "")

    sentences = [str(item).strip() for item in parse_json_array(str(generated)) if str(item).strip()]
    if not sentences:
        raise ValueError("Sentence splitter returned no sentences.")
    return sentences


def split_llm_windows(text: str, max_chars: int = LLM_SENTENCE_SPLIT_CHARS) -> list[str]:
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return [stripped] if stripped else []

    windows = []
    start = 0
    while start < len(stripped):
        end = min(start + max_chars, len(stripped))
        if end < len(stripped):
            boundary = max(
                stripped.rfind(marker, start, end)
                for marker in EXPLICIT_SENTENCE_ENDINGS + " \n"
            )
            if boundary > start + max_chars // 2:
                end = boundary + 1
        windows.append(stripped[start:end].strip())
        start = end
        while start < len(stripped) and stripped[start].isspace():
            start += 1
    return [window for window in windows if window]


def split_sentences_with_llm_windows(sentence_splitter, text: str) -> list[str]:
    sentences = []
    for window in split_llm_windows(text):
        try:
            sentences.extend(split_sentences_with_llm(sentence_splitter, window))
        except (ValueError, json.JSONDecodeError) as exc:
            print(f"warn: Qwen sentence split window failed; falling back to punctuation split: {exc}")
            sentences.extend(split_sentences(window, 0))
    return sentences


def alignment_chars(text: str) -> tuple[str, list[int]]:
    chars = []
    positions = []
    for index, char in enumerate(text):
        if char.isspace():
            continue
        chars.append(char)
        positions.append(index)
    return "".join(chars), positions


def normalized_sentence_for_alignment(sentence: str) -> str:
    return "".join(
        char
        for char in sentence.strip().strip(EXPLICIT_SENTENCE_ENDINGS)
        if not char.isspace()
    )


def locate_sentence_span_fuzzy(text: str, sentence: str, cursor: int) -> tuple[int, int] | None:
    normalized_text, positions = alignment_chars(text[cursor:])
    normalized_sentence = normalized_sentence_for_alignment(sentence)
    if not normalized_sentence:
        return None

    normalized_start = normalized_text.find(normalized_sentence)
    if normalized_start < 0:
        return None

    normalized_end = normalized_start + len(normalized_sentence) - 1
    start = cursor + positions[normalized_start]
    end = cursor + positions[normalized_end] + 1
    while end < len(text) and text[end] in EXPLICIT_SENTENCE_ENDINGS:
        end += 1
    return start, end


def locate_sentence_spans(text: str, sentences: Iterable[str]) -> list[tuple[str, int, int]]:
    spans = []
    cursor = 0
    for sentence in sentences:
        start = text.find(sentence, cursor)
        if start < 0:
            fuzzy_span = locate_sentence_span_fuzzy(text, sentence, cursor)
            if fuzzy_span is None:
                raise ValueError(f"Could not align sentence to source text: {sentence}")
            start, end = fuzzy_span
        else:
            end = start + len(sentence)
        spans.append((text[start:end].strip(), start, end))
        cursor = end
    return spans


def split_sentence_spans_with_llm(sentence_splitter, text: str) -> list[tuple[str, int, int]]:
    stripped = text.strip()
    if not stripped:
        return []
    offset = text.find(stripped)
    sentences = split_sentences_with_llm_windows(sentence_splitter, stripped)
    return [
        (sentence, offset + start, offset + end)
        for sentence, start, end in locate_sentence_spans(stripped, sentences)
    ]


def split_sentence_utterances_with_llm(sentence_splitter, text: str) -> list[Utterance]:
    return [Utterance(sentence) for sentence, _, _ in split_sentence_spans_with_llm(sentence_splitter, text)]


def speaker_marker_matches(text: str) -> list[tuple[int, int, str]]:
    matches = []
    for marker, speaker in SPEAKER_MARKERS:
        start = 0
        while True:
            index = text.find(marker, start)
            if index < 0:
                break
            matches.append((index, index + len(marker), speaker))
            start = index + len(marker)

    matches.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    filtered = []
    last_end = -1
    for start, end, speaker in matches:
        if start < last_end:
            continue
        filtered.append((start, end, speaker))
        last_end = end
    return filtered


def infer_text_speaker(text: str) -> str | None:
    if any(keyword in text for keyword in CHAIR_KEYWORDS):
        return "委員長"
    return None


def proportional_time(
    utterance: Utterance,
    start_char: int,
    end_char: int,
    total_chars: int,
) -> tuple[float | None, float | None]:
    if utterance.start is None or utterance.end is None or total_chars <= 0 or utterance.end <= utterance.start:
        return utterance.start, utterance.end

    duration = utterance.end - utterance.start
    start = utterance.start + duration * (start_char / total_chars)
    end = utterance.start + duration * (end_char / total_chars)
    return start, end


def attribute_speakers_from_text(
    utterances: Iterable[Utterance],
    *,
    default_speaker: str,
) -> list[Utterance]:
    attributed = []
    current_speaker = default_speaker

    for utterance in utterances:
        text = utterance.text.strip()
        if not text:
            continue

        matches = speaker_marker_matches(text)
        if not matches:
            speaker = infer_text_speaker(text) or utterance.speaker or current_speaker
            attributed.append(replace(utterance, speaker=speaker))
            current_speaker = speaker
            continue

        cursor = 0
        total_chars = len(text)
        for marker_start, marker_end, speaker in matches:
            before = text[cursor:marker_start].strip()
            if before:
                start, end = proportional_time(utterance, cursor, marker_start, total_chars)
                attributed.append(Utterance(before, start, end, infer_text_speaker(before) or current_speaker))

            current_speaker = speaker
            cursor = marker_end

        after = text[cursor:].strip()
        if after:
            start, end = proportional_time(utterance, cursor, total_chars, total_chars)
            speaker = infer_text_speaker(after) or current_speaker
            attributed.append(Utterance(after, start, end, speaker))
            current_speaker = speaker

    return attributed


def infer_act(text: str, previous_act: str | None = None) -> str:
    stripped = text.strip()
    if stripped.endswith(QUESTION_ENDINGS) or "お尋ねします" in stripped or "教えてください" in stripped:
        return "question"
    if previous_act == "question" or stripped.startswith(ANSWER_PREFIXES):
        return "answer"
    if any(keyword in stripped for keyword in DECISION_KEYWORDS):
        return "decision"
    if any(keyword in stripped for keyword in ACTION_ITEM_KEYWORDS):
        return "action_item"
    if any(keyword in stripped for keyword in CONCERN_KEYWORDS):
        return "concern"
    return "explanation"


def infer_stance(text: str) -> str:
    stripped = text.strip()
    negative_hits = sum(1 for keyword in NEGATIVE_KEYWORDS if keyword in stripped)
    positive_hits = sum(1 for keyword in POSITIVE_KEYWORDS if keyword in stripped)
    if negative_hits > positive_hits:
        return "negative"
    if positive_hits > negative_hits:
        return "positive"
    return "neutral"


def infer_importance(text: str, act: str) -> str:
    stripped = text.strip()
    if not stripped:
        return "low"
    if any(keyword in stripped for keyword in LOW_IMPORTANCE_KEYWORDS):
        return "low"
    if act in {"decision", "action_item"}:
        return "high"
    if act in {"question", "answer", "concern"} and len(stripped) >= 12:
        return "high"
    if any(keyword in stripped for keyword in ("総理", "大臣", "法案", "減税", "増税", "緊急", "結論", "質問", "お尋ね")):
        return "high"
    if len(stripped) <= 12:
        return "low"
    if len(stripped) >= 32:
        return "medium"
    return "low"


def infer_labels(text: str, previous_act: str | None = None) -> Labels:
    act = infer_act(text, previous_act)
    return Labels(
        act=act,
        stance=infer_stance(text),
        importance=infer_importance(text, act),
    )


def chunk_timestamp(chunk: dict) -> tuple[float | None, float | None]:
    timestamp = chunk.get("timestamp")
    if not timestamp or len(timestamp) != 2:
        return None, None
    start, end = timestamp
    return start, end


def needs_chunk_separator(previous: str, current: str) -> bool:
    if not previous or not current:
        return False
    if previous[-1].isspace() or current[0].isspace():
        return False
    if previous[-1] in EXPLICIT_SENTENCE_ENDINGS:
        return False
    previous_is_ascii = previous[-1].isascii() and previous[-1].isalnum()
    current_is_ascii = current[0].isascii() and current[0].isalnum()
    return previous_is_ascii and current_is_ascii


def combined_chunk_text(
    chunks: Iterable[dict],
) -> tuple[str, list[tuple[int, int, float | None, float | None]]]:
    text_parts = []
    spans = []
    cursor = 0

    for chunk in chunks:
        chunk_text = chunk.get("text", "").strip()
        if not chunk_text:
            continue
        if text_parts and needs_chunk_separator(text_parts[-1], chunk_text):
            text_parts.append(" ")
            cursor += 1

        start_char = cursor
        text_parts.append(chunk_text)
        cursor += len(chunk_text)
        end_char = cursor
        start_time, end_time = chunk_timestamp(chunk)
        spans.append((start_char, end_char, start_time, end_time))

    return "".join(text_parts), spans


def interpolate_chunk_time(
    char_index: int,
    chunk_start_char: int,
    chunk_end_char: int,
    chunk_start_time: float | None,
    chunk_end_time: float | None,
) -> float | None:
    if chunk_start_time is None or chunk_end_time is None:
        return None
    if chunk_end_char <= chunk_start_char or chunk_end_time <= chunk_start_time:
        return chunk_start_time
    ratio = (char_index - chunk_start_char) / (chunk_end_char - chunk_start_char)
    ratio = min(max(ratio, 0.0), 1.0)
    return chunk_start_time + (chunk_end_time - chunk_start_time) * ratio


def sentence_span_timestamp(
    sentence_start: int,
    sentence_end: int,
    chunk_spans: list[tuple[int, int, float | None, float | None]],
) -> tuple[float | None, float | None]:
    overlapping = [
        chunk_span
        for chunk_span in chunk_spans
        if chunk_span[0] < sentence_end and sentence_start < chunk_span[1]
    ]
    if not overlapping:
        return None, None

    first = overlapping[0]
    last = overlapping[-1]
    start = interpolate_chunk_time(sentence_start, *first)
    end = interpolate_chunk_time(sentence_end, *last)
    return start, end


def result_utterances(result: dict, fallback_text: str, max_chars: int, sentence_splitter=None) -> list[Utterance]:
    chunks = result.get("chunks") or []
    text, chunk_spans = combined_chunk_text(chunks)
    if text:
        utterances = []
        try:
            sentence_spans = (
                split_sentence_spans_with_llm(sentence_splitter, text)
                if sentence_splitter is not None
                else split_sentence_spans(text)
            )
        except (ValueError, json.JSONDecodeError) as exc:
            print(f"warn: Qwen sentence split failed; falling back to punctuation split: {exc}")
            sentence_spans = split_sentence_spans(text)

        for sentence, start_char, end_char in sentence_spans:
            start, end = sentence_span_timestamp(start_char, end_char, chunk_spans)
            utterances.append(Utterance(sentence, start, end))
        if utterances:
            return utterances
    if sentence_splitter is not None:
        try:
            return split_sentence_utterances_with_llm(sentence_splitter, fallback_text)
        except (ValueError, json.JSONDecodeError) as exc:
            print(f"warn: Qwen sentence split failed; falling back to punctuation split: {exc}")
    return split_sentence_utterances(fallback_text, max_chars)


def load_diarizer(model_id: str, device: int | str):
    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "Speaker diarization requires pyannote.audio. Install requirements, then set HF_TOKEN "
            "if the selected diarization model requires Hugging Face access."
        ) from exc

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    pretrained_kwargs = {"token": token}
    if "token" not in inspect.signature(Pipeline.from_pretrained).parameters:
        pretrained_kwargs = {"use_auth_token": token}
    diarizer = Pipeline.from_pretrained(model_id, **pretrained_kwargs)
    if device == 0 and torch.cuda.is_available():
        diarizer.to(torch.device("cuda"))
    return diarizer


def normalize_diarization_speakers(segments: list[DiarizationSegment]) -> list[DiarizationSegment]:
    speaker_names: dict[str, str] = {}
    normalized = []
    for segment in segments:
        if segment.speaker not in speaker_names:
            speaker_names[segment.speaker] = f"Speaker{len(speaker_names) + 1}"
        normalized.append(
            DiarizationSegment(
                start=segment.start,
                end=segment.end,
                speaker=speaker_names[segment.speaker],
            )
        )
    return normalized


def diarize_file(diarizer, wav_path: Path, num_speakers: int) -> list[DiarizationSegment]:
    kwargs = {"num_speakers": num_speakers} if num_speakers > 0 else {}
    diarization = diarizer(str(wav_path), **kwargs)
    segments = [
        DiarizationSegment(start=turn.start, end=turn.end, speaker=speaker)
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]
    return normalize_diarization_speakers(segments)


def window_rms(audio: np.ndarray, frame_samples: int) -> np.ndarray:
    if len(audio) < frame_samples:
        return np.array([float(np.sqrt(np.mean(np.square(audio))))])

    frame_count = len(audio) // frame_samples
    trimmed = audio[: frame_count * frame_samples]
    frames = trimmed.reshape(frame_count, frame_samples)
    return np.sqrt(np.mean(np.square(frames), axis=1))


def local_diarization_features(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    window = np.hanning(len(audio))
    spectrum = np.abs(np.fft.rfft(audio * window)) + 1e-8
    freqs = np.fft.rfftfreq(len(audio), d=1 / sample_rate)

    voice_mask = (freqs >= 80) & (freqs <= 4_000)
    voice_freqs = freqs[voice_mask]
    voice_spectrum = spectrum[voice_mask]
    voice_sum = float(np.sum(voice_spectrum)) + 1e-8

    centroid = float(np.sum(voice_freqs * voice_spectrum) / voice_sum)
    bandwidth = float(np.sqrt(np.sum(((voice_freqs - centroid) ** 2) * voice_spectrum) / voice_sum))
    zcr = float(np.mean(np.abs(np.diff(np.sign(audio)))))
    rms = float(np.sqrt(np.mean(np.square(audio))))

    bands = np.linspace(80, 4_000, 13)
    band_energies = []
    for lower, upper in zip(bands[:-1], bands[1:]):
        band_mask = (freqs >= lower) & (freqs < upper)
        band_energies.append(float(np.log(np.mean(spectrum[band_mask]) + 1e-8)))

    return np.array([rms, zcr, centroid / 4_000, bandwidth / 4_000, *band_energies], dtype=np.float32)


def merge_diarization_segments(segments: list[DiarizationSegment], max_gap_s: float = 0.25) -> list[DiarizationSegment]:
    if not segments:
        return []

    merged = [segments[0]]
    for segment in segments[1:]:
        previous = merged[-1]
        if segment.speaker == previous.speaker and segment.start - previous.end <= max_gap_s:
            merged[-1] = DiarizationSegment(previous.start, max(previous.end, segment.end), previous.speaker)
        else:
            merged.append(segment)
    return merged


def local_diarize_file(
    wav_path: Path,
    num_speakers: int,
    window_s: float,
    hop_s: float,
) -> list[DiarizationSegment]:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    if num_speakers <= 0:
        num_speakers = 2

    loaded_audio = load_wav_for_whisper(wav_path)
    audio = loaded_audio["array"]
    sample_rate = loaded_audio["sampling_rate"]
    assert isinstance(audio, np.ndarray)
    assert isinstance(sample_rate, int)

    window_samples = max(int(window_s * sample_rate), sample_rate // 2)
    hop_samples = max(int(hop_s * sample_rate), sample_rate // 4)
    rms_values = window_rms(audio, max(sample_rate // 10, 1))
    energy_threshold = np.percentile(rms_values, LOCAL_DIARIZATION_MIN_ENERGY_PERCENTILE)

    starts = list(range(0, max(len(audio) - window_samples + 1, 1), hop_samples))
    if starts and starts[-1] + window_samples < len(audio):
        starts.append(len(audio) - window_samples)

    feature_rows = []
    kept_starts = []
    for start_sample in starts:
        end_sample = min(start_sample + window_samples, len(audio))
        chunk = audio[start_sample:end_sample]
        if len(chunk) < sample_rate // 2:
            continue
        if float(np.sqrt(np.mean(np.square(chunk)))) < energy_threshold:
            continue
        if len(chunk) < window_samples:
            chunk = np.pad(chunk, (0, window_samples - len(chunk)))
        feature_rows.append(local_diarization_features(chunk, sample_rate))
        kept_starts.append(start_sample)

    if len(feature_rows) < num_speakers:
        duration = len(audio) / sample_rate
        return [DiarizationSegment(0.0, duration, "Speaker1")]

    features = StandardScaler().fit_transform(np.vstack(feature_rows))
    labels = KMeans(n_clusters=num_speakers, random_state=0, n_init=10).fit_predict(features)

    speaker_names: dict[int, str] = {}
    segments = []
    for index, (start_sample, label) in enumerate(zip(kept_starts, labels)):
        if int(label) not in speaker_names:
            speaker_names[int(label)] = f"Speaker{len(speaker_names) + 1}"
        start = start_sample / sample_rate
        window_end = min((start_sample + window_samples) / sample_rate, len(audio) / sample_rate)
        if index + 1 < len(kept_starts):
            next_start = kept_starts[index + 1] / sample_rate
            end = next_start if next_start <= window_end else window_end
        else:
            end = window_end
        segments.append(DiarizationSegment(start, end, speaker_names[int(label)]))

    return merge_diarization_segments(segments)


def assign_speaker(
    utterance: Utterance,
    diarization_segments: list[DiarizationSegment],
    default_speaker: str,
) -> str:
    if utterance.start is None or utterance.end is None or not diarization_segments:
        return default_speaker

    best_speaker = default_speaker
    best_overlap = 0.0
    for segment in diarization_segments:
        overlap = max(0.0, min(utterance.end, segment.end) - max(utterance.start, segment.start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = segment.speaker

    if best_overlap > 0:
        return best_speaker

    midpoint = (utterance.start + utterance.end) / 2
    nearest = min(
        diarization_segments,
        key=lambda segment: min(abs(midpoint - segment.start), abs(midpoint - segment.end)),
    )
    return nearest.speaker


def diarization_to_json(segments: list[DiarizationSegment]) -> list[dict[str, float | str]]:
    return [
        {"start": segment.start, "end": segment.end, "speaker": segment.speaker}
        for segment in segments
    ]


def diarization_from_json(data: dict) -> list[DiarizationSegment]:
    return [
        DiarizationSegment(
            start=float(segment["start"]),
            end=float(segment["end"]),
            speaker=str(segment["speaker"]),
        )
        for segment in data.get("diarization", [])
        if {"start", "end", "speaker"} <= set(segment)
    ]


def load_text_labeler(model_id: str, device: int | str):
    kwargs = {
        "model": model_id,
        "torch_dtype": "auto",
    }
    if device == 0:
        kwargs["device_map"] = "auto"
    else:
        kwargs["device"] = device
    return pipeline("text-generation", **kwargs)


def parse_json_object(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("No JSON object found in model output.")
    return json.loads(text[start : end + 1])


def normalize_label(value: str, allowed: set[str], default: str) -> str:
    value = str(value).strip()
    return value if value in allowed else default


def classify_labels_with_llm(labeler, text: str) -> Labels:
    messages = [
        {
            "role": "system",
            "content": (
                "Reasoning: low\n"
                "Classify a Japanese utterance for meeting summarization. "
                "Return only compact JSON. Do not explain. Do not translate label values. "
                'Use exactly this shape: {"act":"question","stance":"neutral","importance":"high"}. '
                "act value must be exactly one of: question, answer, explanation, concern, decision, action_item. "
                "stance value must be exactly one of: positive, neutral, negative. "
                "importance value must be exactly one of: high, medium, low."
            ),
        },
        {"role": "user", "content": text},
    ]
    output = labeler(messages, max_new_tokens=80, do_sample=False)
    generated = output[0].get("generated_text", output[0])
    if isinstance(generated, list):
        generated = generated[-1].get("content", "")
    data = parse_json_object(str(generated))
    fallback = infer_labels(text)
    return Labels(
        act=normalize_label(
            data.get("act", fallback.act),
            {"question", "answer", "explanation", "concern", "decision", "action_item"},
            fallback.act,
        ),
        stance=normalize_label(
            data.get("stance", fallback.stance),
            {"positive", "neutral", "negative"},
            fallback.stance,
        ),
        importance=normalize_label(
            data.get("importance", fallback.importance),
            {"high", "medium", "low"},
            fallback.importance,
        ),
    )


def format_labeled_text(
    utterances: Iterable[Utterance],
    *,
    default_speaker: str,
    default_act: str,
    default_stance: str,
    default_importance: str,
    infer_acts: bool,
    infer_stances: bool,
    infer_importance_labels: bool,
    speaker_mode: str,
    num_speakers: int,
    diarization_segments: list[DiarizationSegment],
    prefer_text_speaker: bool,
    labeler=None,
) -> str:
    lines = []
    previous_act: str | None = None

    for index, utterance in enumerate(utterances):
        if prefer_text_speaker and utterance.speaker:
            speaker = utterance.speaker
        elif diarization_segments:
            speaker = assign_speaker(utterance, diarization_segments, default_speaker)
        elif utterance.speaker:
            speaker = utterance.speaker
        elif speaker_mode == "alternate":
            speaker = f"Speaker{(index % num_speakers) + 1}"
        else:
            speaker = default_speaker

        inferred = classify_labels_with_llm(labeler, utterance.text) if labeler is not None else infer_labels(utterance.text, previous_act)
        act = inferred.act if infer_acts else default_act
        stance = inferred.stance if infer_stances else default_stance
        importance = inferred.importance if infer_importance_labels else default_importance
        previous_act = act
        lines.append(f"[{speaker}] act: {act} | stance: {stance} | importance: {importance} | text: {utterance.text}")

    return "\n".join(lines)


def choose_device(requested: str) -> tuple[int | str, torch.dtype]:
    if requested == "cpu":
        return -1, torch.float32
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
        return 0, torch.float16
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested, but torch.backends.mps.is_available() is false.")
        return "mps", torch.float32

    if torch.cuda.is_available():
        return 0, torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float32
    return -1, torch.float32


def build_parser() -> argparse.ArgumentParser:
    data_dir = default_data_dir()

    parser = argparse.ArgumentParser(
        description="Transcribe wav files from data/audio to text files in data/text."
    )
    parser.add_argument("--audio-dir", type=Path, default=data_dir / "audio")
    parser.add_argument("--text-dir", type=Path, default=data_dir / "text")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--language", default=None, help="Language code such as ja or en. Omit for auto detection.")
    parser.add_argument("--task", choices=("transcribe", "translate"), default="transcribe")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--chunk-length-s", type=float, default=30.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--timestamps", action="store_true", help="Also write timestamp data as JSON.")
    parser.add_argument("--output-format", choices=("plain", "labeled"), default="labeled")
    parser.add_argument("--default-speaker", default="Speaker1")
    parser.add_argument(
        "--default-act",
        choices=("question", "answer", "explanation", "concern", "decision", "action_item"),
        default="explanation",
    )
    parser.add_argument("--default-stance", choices=("positive", "neutral", "negative"), default="neutral")
    parser.add_argument("--default-importance", choices=("high", "medium", "low"), default="medium")
    parser.add_argument("--no-infer-act", action="store_true", help="Do not infer act labels.")
    parser.add_argument("--no-infer-stance", action="store_true", help="Do not infer stance labels.")
    parser.add_argument("--no-infer-importance", action="store_true", help="Do not infer importance labels.")
    parser.add_argument(
        "--labeler-backend",
        choices=("heuristic", "hf-llm", "gpt-oss"),
        default="heuristic",
        help="Label classifier backend for act/stance/importance.",
    )
    parser.add_argument("--labeler-model-id", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument(
        "--sentence-splitter",
        choices=("punctuation", "qwen"),
        default="punctuation",
        help="How to split transcript text into one sentence per output line.",
    )
    parser.add_argument("--sentence-splitter-model-id", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--speaker-mode", choices=("single", "alternate"), default="single")
    parser.add_argument(
        "--speaker-attribution",
        choices=("text", "audio", "hybrid", "none"),
        default="hybrid",
        help="How to assign speaker labels. hybrid prefers text markers and falls back to diarization.",
    )
    parser.add_argument("--num-speakers", type=int, default=2)
    parser.add_argument("--diarize", action="store_true", help="Enable speaker diarization.")
    parser.add_argument(
        "--diarization-backend",
        choices=("pyannote", "local"),
        default="pyannote",
        help="Speaker diarization backend. local uses no gated model downloads.",
    )
    parser.add_argument("--diarization-model-id", default=DEFAULT_DIARIZATION_MODEL_ID)
    parser.add_argument("--local-diarization-window-s", type=float, default=2.0)
    parser.add_argument("--local-diarization-hop-s", type=float, default=1.0)
    parser.add_argument(
        "--label-existing-text",
        action="store_true",
        help="Label existing .txt files in --text-dir without transcribing audio.",
    )
    parser.add_argument(
        "--format-existing-json",
        action="store_true",
        help="Format existing Whisper .json files in --text-dir without transcribing audio.",
    )
    parser.add_argument("--labeled-suffix", default=".labeled", help="Suffix for --label-existing-text outputs.")
    parser.add_argument("--max-line-chars", type=int, default=120)
    parser.add_argument("--force", action="store_true", help="Overwrite existing text files.")
    return parser


def build_labeler(args: argparse.Namespace, device: int | str):
    if args.labeler_backend in {"hf-llm", "gpt-oss"}:
        return load_text_labeler(args.labeler_model_id, device)
    return None


def build_sentence_splitter(args: argparse.Namespace, device: int | str, labeler=None):
    if args.sentence_splitter != "qwen":
        return None
    if labeler is not None and args.labeler_model_id == args.sentence_splitter_model_id:
        return labeler
    return load_text_labeler(args.sentence_splitter_model_id, device)


def label_existing_text_files(args: argparse.Namespace) -> int:
    if not args.text_dir.exists():
        raise FileNotFoundError(f"Text directory does not exist: {args.text_dir}")

    device, _ = choose_device(args.device)
    labeler = build_labeler(args, device)
    sentence_splitter = build_sentence_splitter(args, device, labeler)

    input_files = list(text_files(args.text_dir, args.labeled_suffix))
    if not input_files:
        print(f"No text files found in {args.text_dir}")
        return 0

    for txt_path in input_files:
        output_path = txt_path.with_name(f"{txt_path.stem}{args.labeled_suffix}.txt")
        if output_path.exists() and not args.force:
            print(f"skip: {txt_path.name} -> {output_path.name} already exists")
            continue

        text = txt_path.read_text(encoding="utf-8").strip()
        if sentence_splitter is not None:
            try:
                utterances = split_sentence_utterances_with_llm(sentence_splitter, text)
            except (ValueError, json.JSONDecodeError) as exc:
                print(f"warn: Qwen sentence split failed; falling back to punctuation split: {exc}")
                utterances = split_sentence_utterances(text, args.max_line_chars)
        else:
            utterances = split_sentence_utterances(text, args.max_line_chars)
        if args.speaker_attribution in {"text", "hybrid"}:
            utterances = attribute_speakers_from_text(
                utterances,
                default_speaker=args.default_speaker,
            )
        output_text = format_labeled_text(
            utterances,
            default_speaker=args.default_speaker,
            default_act=args.default_act,
            default_stance=args.default_stance,
            default_importance=args.default_importance,
            infer_acts=not args.no_infer_act,
            infer_stances=not args.no_infer_stance,
            infer_importance_labels=not args.no_infer_importance,
            speaker_mode=args.speaker_mode,
            num_speakers=max(args.num_speakers, 1),
            diarization_segments=[],
            prefer_text_speaker=args.speaker_attribution in {"text", "hybrid"},
            labeler=labeler,
        )
        output_path.write_text(output_text.strip() + "\n", encoding="utf-8")
        print(f"write: {output_path}")

    return 0


def format_existing_json_files(args: argparse.Namespace) -> int:
    if not args.text_dir.exists():
        raise FileNotFoundError(f"Text directory does not exist: {args.text_dir}")

    device, _ = choose_device(args.device)
    labeler = build_labeler(args, device)
    sentence_splitter = build_sentence_splitter(args, device, labeler)

    input_files = list(json_files(args.text_dir))
    if not input_files:
        print(f"No json files found in {args.text_dir}")
        return 0

    for json_path in input_files:
        txt_path = json_path.with_suffix(".txt")
        if txt_path.exists() and not args.force:
            print(f"skip: {json_path.name} -> {txt_path.name} already exists")
            continue

        result = json.loads(json_path.read_text(encoding="utf-8"))
        text = result.get("text", "").strip()
        utterances = result_utterances(result, text, args.max_line_chars, sentence_splitter)
        if args.speaker_attribution in {"text", "hybrid"}:
            utterances = attribute_speakers_from_text(
                utterances,
                default_speaker=args.default_speaker,
            )
        diarization_segments = diarization_from_json(result) if args.speaker_attribution in {"audio", "hybrid"} else []
        output_text = format_labeled_text(
            utterances,
            default_speaker=args.default_speaker,
            default_act=args.default_act,
            default_stance=args.default_stance,
            default_importance=args.default_importance,
            infer_acts=not args.no_infer_act,
            infer_stances=not args.no_infer_stance,
            infer_importance_labels=not args.no_infer_importance,
            speaker_mode=args.speaker_mode,
            num_speakers=max(args.num_speakers, 1),
            diarization_segments=diarization_segments,
            prefer_text_speaker=args.speaker_attribution in {"text", "hybrid"},
            labeler=labeler,
        )
        txt_path.write_text(output_text.strip() + "\n", encoding="utf-8")
        print(f"write: {txt_path}")

    return 0


def main() -> int:
    args = build_parser().parse_args()

    if args.label_existing_text:
        return label_existing_text_files(args)
    if args.format_existing_json:
        return format_existing_json_files(args)

    if not args.audio_dir.exists():
        raise FileNotFoundError(f"Audio directory does not exist: {args.audio_dir}")
    args.text_dir.mkdir(parents=True, exist_ok=True)

    device, torch_dtype = choose_device(args.device)
    transcriber = pipeline(
        "automatic-speech-recognition",
        model=args.model_id,
        dtype=torch_dtype,
        device=device,
    )
    diarizer = load_diarizer(args.diarization_model_id, device) if args.diarize and args.diarization_backend == "pyannote" else None
    labeler = build_labeler(args, device)
    sentence_splitter = build_sentence_splitter(args, device, labeler)

    wav_files = list(audio_files(args.audio_dir))
    if not wav_files:
        print(f"No wav files found in {args.audio_dir}")
        return 0

    generate_kwargs = {"task": args.task}
    if args.language:
        generate_kwargs["language"] = args.language

    for wav_path in wav_files:
        txt_path = args.text_dir / f"{wav_path.stem}.txt"
        json_path = args.text_dir / f"{wav_path.stem}.json"

        if txt_path.exists() and not args.force:
            print(f"skip: {wav_path.name} -> {txt_path.name} already exists")
            continue

        print(f"transcribe: {wav_path.name}")
        result = transcriber(
            load_wav_for_whisper(wav_path),
            chunk_length_s=args.chunk_length_s,
            batch_size=args.batch_size,
            generate_kwargs=generate_kwargs,
            return_timestamps=args.timestamps or args.output_format == "labeled",
        )
        diarization_segments = []
        if args.diarize and args.diarization_backend == "local":
            print(f"diarize-local: {wav_path.name}")
            diarization_segments = local_diarize_file(
                wav_path,
                max(args.num_speakers, 0),
                args.local_diarization_window_s,
                args.local_diarization_hop_s,
            )
        elif diarizer is not None:
            print(f"diarize: {wav_path.name}")
            diarization_segments = diarize_file(diarizer, wav_path, max(args.num_speakers, 0))

        text = result["text"].strip()
        if args.output_format == "labeled":
            utterances = result_utterances(result, text, args.max_line_chars, sentence_splitter)
            if args.speaker_attribution in {"text", "hybrid"}:
                utterances = attribute_speakers_from_text(
                    utterances,
                    default_speaker=args.default_speaker,
                )
            output_text = format_labeled_text(
                utterances,
                default_speaker=args.default_speaker,
                default_act=args.default_act,
                default_stance=args.default_stance,
                default_importance=args.default_importance,
                infer_acts=not args.no_infer_act,
                infer_stances=not args.no_infer_stance,
                infer_importance_labels=not args.no_infer_importance,
                speaker_mode=args.speaker_mode,
                num_speakers=max(args.num_speakers, 1),
                diarization_segments=diarization_segments,
                prefer_text_speaker=args.speaker_attribution in {"text", "hybrid"},
                labeler=labeler,
            )
        else:
            output_text = text

        txt_path.write_text(output_text.strip() + "\n", encoding="utf-8")
        print(f"write: {txt_path}")

        if args.timestamps:
            if diarization_segments:
                result["diarization"] = diarization_to_json(diarization_segments)
            json_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"write: {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
