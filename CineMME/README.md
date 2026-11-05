# CineMME

CineMME is a movie-scene benchmark for coupled cinematic grounding and reasoning.
This directory contains annotation metadata, evaluation scripts, and core construction prompts.
It does not redistribute movie clips. Each sample provides the original YouTube URL so users can retrieve videos under their own legal and institutional constraints.

This package lives in `CineMME/` and replaces the earlier `VUE_PLOT/` directory.
It contains 546 grounding videos, 1,600 MCQs, and 316 open-ended questions. See the
[migration notes](MIGRATION.md) for changed paths, question IDs, and option order.

## Contents

- `datasets/`
  - `cinemme_grounding.json`: speech transcript, temporal boundary, and active-speaker face-box annotations.
  - `cinemme_reasoning_mcq.json`: multiple-choice reasoning questions with gold answers.
  - `cinemme_reasoning_oeq.json`: open-ended reasoning questions with reference answers, reasoning traces, and judge context.
  - `video_sources.json`: unique source video IDs and YouTube URLs.
  - `croissant.json`: Croissant metadata for dataset hosting.
- `evaluation/`
  - vLLM/OpenAI-compatible inference scripts for MCQ, OEQ, and grounding.
  - metric utilities for MCQ answer matching, OEQ LLM-as-a-judge scoring, and grounding tIoU/sIoU/WER.
- `data_process/`
  - ASR, face tracking, multimodal annotation, distractor-refinement scripts, schemas, and prompt templates used in dataset construction.
- `scripts/`
  - wrappers for construction and evaluation, plus dataset validation.
- `tests/`
  - offline regression tests for evaluation and dataset validation.

## Data Policy

The dataset contains annotations and source URLs. Code and annotation licenses
are described in [LICENSES.md](LICENSES.md).
Users are responsible for complying with YouTube terms and the copyright status of source videos.

## Quick Start

From the Vidi repository root, enter `CineMME/` and install dependencies. Run all
subsequent commands in this guide from that directory:

```bash
cd CineMME
pip install -r requirements.txt
```

Run MCQ inference against a vLLM OpenAI-compatible endpoint:

```bash
python -m evaluation.inference.run_mcq \
  --dataset datasets/cinemme_reasoning_mcq.json \
  --video-root /path/to/local/videos \
  --output outputs/mcq_predictions.jsonl \
  --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --api-base http://localhost:8000/v1 \
  --api-key EMPTY \
  --use-audio
```

The evaluation scripts assume local videos are named by YouTube ID, for example `xY6bhYtH8Pw.mp4`.
Use `--video-template` if your local naming convention differs.

To reproduce construction stages, see `data_process/README.md` or use the wrappers in `scripts/`.
See `evaluation/README.md` for all three scoring protocols and their coverage denominators.
See [datasets/README.md](datasets/README.md) for schemas, version notes, and validation.
