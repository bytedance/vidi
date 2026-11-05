#!/usr/bin/env bash
set -euo pipefail

MODEL=${1:?Usage: scripts/run_grounding_eval.sh model local_videos outputs/grounding_predictions.jsonl}
VIDEO_ROOT=${2:?Usage: scripts/run_grounding_eval.sh model local_videos outputs/grounding_predictions.jsonl}
OUTPUT=${3:?Usage: scripts/run_grounding_eval.sh model local_videos outputs/grounding_predictions.jsonl}

python -m evaluation.inference.run_grounding \
  --dataset datasets/cinemme_grounding.json \
  --video-root "$VIDEO_ROOT" \
  --output "$OUTPUT" \
  --model "$MODEL" \
  --api-base "${VLLM_API_BASE:-http://localhost:8000/v1}" \
  --api-key "${VLLM_API_KEY:-EMPTY}" \
  --use-audio

python -m evaluation.scoring.score_grounding \
  --gt-file datasets/cinemme_grounding.json \
  --input-file "$OUTPUT" \
  --output-dir outputs/grounding_eval
