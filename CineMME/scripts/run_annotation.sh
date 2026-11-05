#!/usr/bin/env bash
set -euo pipefail

VIDEO=${1:?Usage: scripts/run_annotation.sh video.mp4 asr.json face_tracklets.json output.json model}
ASR_JSON=${2:?Usage: scripts/run_annotation.sh video.mp4 asr.json face_tracklets.json output.json model}
FACE_JSON=${3:?Usage: scripts/run_annotation.sh video.mp4 asr.json face_tracklets.json output.json model}
OUTPUT=${4:?Usage: scripts/run_annotation.sh video.mp4 asr.json face_tracklets.json output.json model}
MODEL=${5:?Usage: scripts/run_annotation.sh video.mp4 asr.json face_tracklets.json output.json model}

python -m data_process.annotation.run_multimodal_annotation \
  --mode video-analysis \
  --video "$VIDEO" \
  --asr-json "$ASR_JSON" \
  --face-tracks-json "$FACE_JSON" \
  --output "$OUTPUT" \
  --model "$MODEL" \
  --api-base "${VLLM_API_BASE:-http://localhost:8000/v1}" \
  --api-key "${VLLM_API_KEY:-EMPTY}"
