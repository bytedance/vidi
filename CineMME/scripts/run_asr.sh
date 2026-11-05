#!/usr/bin/env bash
set -euo pipefail

VIDEO=${1:?Usage: scripts/run_asr.sh path/to/video.mp4 runs/asr/video.json}
OUTPUT=${2:?Usage: scripts/run_asr.sh path/to/video.mp4 runs/asr/video.json}

python -m data_process.asr.run_whisperx \
  --video "$VIDEO" \
  --output "$OUTPUT"
