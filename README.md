# ProsodySum

ProsodySum is a small transcription workspace for preparing Japanese meeting or Diet-style speech data. It converts `.wav` files placed under `data/audio` into text and optional JSON outputs under `data/text` using Whisper, then formats utterances with speaker, act, stance, and importance labels.

## Repository Layout

```text
data/
  audio/          Input .wav files
  prog/           Transcription scripts and Python requirements
  text/           Generated and curated .txt/.json transcript data
```

The main script is `data/prog/transcribe.py`. More detailed operation notes are in `data/prog/README.md`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r data/prog/requirements.txt
```

The script reads `.wav` files directly from Python, so `ffmpeg` is usually not required for the default workflow.

## Basic Usage

Place one or more `.wav` files in `data/audio`, then run:

```bash
python data/prog/transcribe.py --language ja
```

For example, `data/audio/example.wav` will produce `data/text/example.txt`. Existing `.txt` files are not overwritten unless `--force` is specified.

```bash
python data/prog/transcribe.py --language ja --force
```

The default output is formatted for downstream fine-tuning or analysis:

```text
[Speaker1] act: question | stance: negative | importance: high | text: 今回の対応は遅かったのではないですか。
[Speaker1] act: answer | stance: neutral | importance: high | text: 関係機関との調整に時間を要しました。
```

To save only the raw transcription text:

```bash
python data/prog/transcribe.py --language ja --output-format plain
```

## Speaker Attribution and Diarization

The default speaker attribution mode is `hybrid`, which prefers speaker names found in the transcript text and falls back to generic speaker labels when needed.

For simple local diarization without Hugging Face gated pyannote models:

```bash
python data/prog/transcribe.py \
  --language ja \
  --diarize \
  --diarization-backend local \
  --speaker-attribution hybrid \
  --num-speakers 2 \
  --timestamps \
  --force
```

For pyannote-based diarization:

```bash
export HF_TOKEN=...
python data/prog/transcribe.py \
  --language ja \
  --diarize \
  --diarization-backend pyannote \
  --num-speakers 2 \
  --force
```

The default pyannote model may require accepting the model terms on Hugging Face.

## GPU Workflow

If CUDA is available, use the helper script:

```bash
chmod +x data/prog/run_gpu_local.sh
data/prog/run_gpu_local.sh
```

This runs Whisper with local diarization, hybrid speaker attribution, Qwen sentence splitting, timestamps, and `--device cuda`.

## Label Existing Text

To add labels to existing `data/text/*.txt` files without retranscribing audio:

```bash
python data/prog/transcribe.py --label-existing-text --force
```

To use a Hugging Face instruct model for `act`, `stance`, and `importance` estimation:

```bash
python data/prog/transcribe.py \
  --label-existing-text \
  --labeler-backend hf-llm \
  --labeler-model-id Qwen/Qwen2.5-3B-Instruct \
  --speaker-attribution text \
  --device cuda \
  --force
```

## Regenerate Text From JSON

Existing `data/text/*.json` files can be reformatted into `.txt` without rerunning Whisper:

```bash
python data/prog/transcribe.py \
  --format-existing-json \
  --speaker-attribution hybrid \
  --force
```

## Notes

- Default Whisper model: `openai/whisper-large-v3-turbo`
- Supported audio extension: `.wav`
- Default output directory: `data/text`
- Detailed CLI examples: `data/prog/README.md`
