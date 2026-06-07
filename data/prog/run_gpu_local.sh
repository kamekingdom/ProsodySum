#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cache}"

python3 - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA GPU is not available from this process. "
        "Run this script outside the Codex sandbox or check the NVIDIA driver."
    )

print(f"CUDA device: {torch.cuda.get_device_name(0)}")
PY

python3 data/prog/transcribe.py \
  --language ja \
  --diarize \
  --diarization-backend local \
  --speaker-attribution hybrid \
  --sentence-splitter qwen \
  --num-speakers 2 \
  --timestamps \
  --force \
  --device cuda \
  "$@"
