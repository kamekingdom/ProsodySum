#!/usr/bin/env python3
"""Build Baseline and Proposed summarization JSONL files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


LABEL_LINE_RE = re.compile(
    r"^\[(?P<speaker>[^\]]+)\]\s+act:\s*(?P<act>[^|]+)\|\s*stance:\s*(?P<stance>[^|]+)"
    r"(?:\|\s*importance:\s*[^|]+)?\|\s*text:\s*(?P<text>.*)$"
)
NOISE_PHRASES = (
    "ご視聴ありがとうございました",
    "ご清聴ありがとうございました",
)


def collapse_repeated_phrases(text: str) -> str:
    # Collapse short phrase loops such as "群馬の群馬の..." that appear in noisy ASR.
    previous = None
    current = text
    pattern = re.compile(r"(.{2,12}?)\1{3,}")
    while previous != current:
        previous = current
        current = pattern.sub(r"\1", current)
    return current


def default_data_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def clean_source_text(text: str) -> str:
    for phrase in NOISE_PHRASES:
        text = text.replace(phrase, "")
    return normalize_text(collapse_repeated_phrases(text))


def read_json_text(path: Path, clean_source: bool) -> str:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{path} does not contain a non-empty string field: text")
    text = normalize_text(text)
    return clean_source_text(text) if clean_source else text


def read_plain_from_labeled_text(path: Path, clean_source: bool) -> str:
    parts: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = LABEL_LINE_RE.match(stripped)
        parts.append(match.group("text") if match else stripped)
    text = normalize_text(" ".join(parts))
    return clean_source_text(text) if clean_source else text


def read_proposed_text(path: Path, proposed_format: str, clean_source: bool) -> str:
    lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = LABEL_LINE_RE.match(stripped)
        if match:
            speaker = normalize_text(match.group("speaker"))
            act = normalize_text(match.group("act"))
            stance = normalize_text(match.group("stance"))
            text = normalize_text(match.group("text"))
            if clean_source:
                text = clean_source_text(text)
            if proposed_format == "compact":
                lines.append(f"{speaker} <{act}> <{stance}> {text}")
            else:
                lines.append(f"[{speaker}] act: {act} | stance: {stance} | text: {text}")
        else:
            lines.append(clean_source_text(stripped) if clean_source else stripped)
    return normalize_text("\n".join(lines))


def iter_ids(summary_dir: Path, split: str) -> list[str]:
    return sorted(path.stem for path in summary_dir.glob(f"{split}_*.txt"))


def build_records(
    data_dir: Path,
    split: str,
    condition: str,
    baseline_source: str,
    proposed_format: str,
    clean_source: bool,
) -> list[dict[str, str]]:
    text_dir = data_dir / "text"
    summary_dir = data_dir / "summary"
    records: list[dict[str, str]] = []

    for sample_id in iter_ids(summary_dir, split):
        summary_path = summary_dir / f"{sample_id}.txt"
        transcript_txt_path = text_dir / f"{sample_id}.txt"
        transcript_json_path = text_dir / f"{sample_id}.json"

        if not summary_path.exists():
            raise FileNotFoundError(summary_path)
        if not transcript_txt_path.exists():
            raise FileNotFoundError(transcript_txt_path)

        if condition == "baseline":
            if baseline_source == "json":
                if not transcript_json_path.exists():
                    raise FileNotFoundError(transcript_json_path)
                source = read_json_text(transcript_json_path, clean_source)
            else:
                source = read_plain_from_labeled_text(transcript_txt_path, clean_source)
        elif condition == "proposed":
            source = read_proposed_text(transcript_txt_path, proposed_format, clean_source)
        else:
            raise ValueError(f"unknown condition: {condition}")

        summary = normalize_text(summary_path.read_text(encoding="utf-8"))
        if not source:
            raise ValueError(f"empty source for {sample_id}")
        if not summary:
            raise ValueError(f"empty summary for {sample_id}")

        records.append(
            {
                "id": sample_id,
                "split": split,
                "condition": condition,
                "input": f"要約: {source}",
                "summary": summary,
            }
        )

    return records


def write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--output-dir", type=Path, default=default_data_dir() / "summarization")
    parser.add_argument("--baseline-source", choices=("json", "txt"), default="json")
    parser.add_argument("--proposed-format", choices=("verbose", "compact"), default="verbose")
    parser.add_argument("--clean-source", action="store_true")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for condition in ("baseline", "proposed"):
        for split in args.splits:
            records = build_records(
                args.data_dir,
                split,
                condition,
                args.baseline_source,
                args.proposed_format,
                args.clean_source,
            )
            output_path = args.output_dir / condition / f"{split}.jsonl"
            write_jsonl(output_path, records)
            print(f"wrote {len(records):>3} records: {output_path}")


if __name__ == "__main__":
    main()
