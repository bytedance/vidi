#!/usr/bin/env bash
set -euo pipefail

MODEL=${1:?Usage: scripts/run_mcq_eval.sh model local_videos outputs/mcq_predictions.jsonl}
VIDEO_ROOT=${2:?Usage: scripts/run_mcq_eval.sh model local_videos outputs/mcq_predictions.jsonl}
OUTPUT=${3:?Usage: scripts/run_mcq_eval.sh model local_videos outputs/mcq_predictions.jsonl}

python -m evaluation.inference.run_mcq \
  --dataset datasets/cinemme_reasoning_mcq.json \
  --video-root "$VIDEO_ROOT" \
  --output "$OUTPUT" \
  --model "$MODEL" \
  --api-base "${VLLM_API_BASE:-http://localhost:8000/v1}" \
  --api-key "${VLLM_API_KEY:-EMPTY}" \
  --use-audio

python -m evaluation.scoring.score_mcq \
  --predictions "$OUTPUT" \
  --dataset datasets/cinemme_reasoning_mcq.json
