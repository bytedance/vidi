# Evaluation

Run the commands in this guide from `CineMME/` (`cd CineMME` from the Vidi
repository root).

All inference scripts talk to an OpenAI-compatible endpoint: a local vLLM server or a
hosted API such as OpenRouter. The OEQ judge runs on a local vLLM server. `ffmpeg` must be
on `PATH` (or set `FFMPEG_BINARY`).

Videos are resolved as `<video_id>.mp4`, `.mkv`, `.webm`, or `.mov` under `--video-root`,
including nested folders (e.g. `<video_root>/2018/<video_id>.mkv`). Use `--video-template`
when your local naming differs.

## Protocol

- Clips are downscaled to 480p. vLLM receives a silent mp4 sampled at `--fps` (default 1)
  plus separate 16 kHz mono audio (`--use-audio`), both inlined as base64 data URLs, so the
  server can run on another machine (`--transport file` sends `file://` paths instead).
  OpenRouter receives one 1 fps mp4 carrying the audio track, inlined as a base64 data URL.
  vLLM sampling is passed through request-level `media_io_kwargs.video` with `num_frames=-1`.
  For fixed frame counts, `--num-frames 128` uses OpenCV uniform sampling across the clip and
  overrides `--fps`; it requires inline/file transport and at least 128 source frames.
- Decoding: `temperature=0.7`, `top_p=0.01`, `max_tokens=16384`. Keep the large token budget
  for reasoning models, whose thinking tokens count against it. `--max-tokens 0` lets vLLM
  use the remaining model context after the actual multimodal input.
  MiniCPM audio runs can use `--chat-template-kwargs '{"enable_thinking":false,"use_tts_template":true}'`
  to match the HF audio generation prefix. This does not interleave video and audio inputs.
- Output is JSONL, written as results arrive. Rerunning the same command keeps rows with a
  response and retries empty or failed rows. `--workers` sets concurrency.
  Use a new output path when changing the dataset or sampling configuration.

## Backends

vLLM (local or remote; clips are sent inline):

```bash
vllm serve openbmb/MiniCPM-o-4_5 --port 8000 --trust-remote-code
python -m evaluation.inference.run_mcq \
  --dataset datasets/cinemme_reasoning_mcq.json \
  --video-root local_videos \
  --output outputs/minicpm_o_4_5/mcq.jsonl \
  --model openbmb/MiniCPM-o-4_5 \
  --api-base http://localhost:8000/v1 \
  --use-audio
```

OpenRouter: put `OPENROUTER_API_KEY=...` in the environment or in `.env` (git-ignored). Any
`--api-base` on openrouter.ai switches to base64 media and JSON-schema structured output.

```bash
python -m evaluation.inference.run_mcq \
  --dataset datasets/cinemme_reasoning_mcq.json \
  --video-root local_videos \
  --output outputs/mimo_v2_5/mcq.jsonl \
  --model xiaomi/mimo-v2.5 \
  --api-base https://openrouter.ai/api/v1 \
  --use-audio --workers 8
```

`run_oeq` and `run_grounding` take the same arguments with `datasets/cinemme_reasoning_oeq.json`
and `datasets/cinemme_grounding.json`. Rows record `served_model`, `provider`, `finish_reason`,
and token `usage`.

### MiniCPM-o-4.5: 128-frame example

This configuration uses 128 uniformly sampled frames with full audio, temperature
0.7, top-p 0.01, and a 40,960-token context. `--max-tokens 0` gives generation the remaining
context. The tested server uses vLLM 0.19.0, Transformers 5.5.0, and model revision
`44151b35f1b232a280bda5a87ea1a7575d5433fc`. Apply the included multi-chunk audio patch once
to that vLLM environment before serving (commands start in `CineMME/`):

```bash
MINICPM_PATCH="$(pwd)/scripts/patches/vllm-0.19.0-minicpmo-audio.patch"
(
  cd "$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  patch -p1 < "$MINICPM_PATCH"
)
vllm serve openbmb/MiniCPM-o-4_5 \
  --revision 44151b35f1b232a280bda5a87ea1a7575d5433fc \
  --served-model-name openbmb/MiniCPM-o-4_5 --trust-remote-code --dtype bfloat16 \
  --port 8000 -tp 1 --data-parallel-size 2 \
  --max-model-len 40960 --max-num-batched-tokens 40960 --max-num-seqs 2
```

The server command uses two GPUs. In a separate shell, run OEQ with a new output path:

```bash
python -m evaluation.inference.run_oeq \
  --dataset datasets/cinemme_reasoning_oeq.json --video-root local_videos \
  --output outputs/minicpm_o_4_5_128f/oeq.jsonl \
  --model openbmb/MiniCPM-o-4_5 --api-base http://localhost:8000/v1 \
  --use-audio --num-frames 128 --transport inline --max-tokens 0 \
  --temperature 0.7 --top-p 0.01 --workers 2 \
  --chat-template-kwargs '{"enable_thinking":false,"use_tts_template":true}'
```

Score the complete, unmodified answers with the Qwen2.5-72B-Instruct judge below. Retain
`finish_reason=length` records and report their count alongside coverage.

## Scoring

MCQ accuracy is computed over questions with an extractable answer (`accuracy`, the paper
metric); `accuracy_all` also counts unanswered questions as wrong.
When `--dataset` is supplied, its complete question count is the denominator for
`accuracy_all`, including missing predictions. Duplicate or unknown question IDs
are rejected. Use the same dataset and option order that produced the predictions;
historical runs with reordered options must be aligned before using this CLI.

```bash
python -m evaluation.scoring.score_mcq \
  --predictions outputs/mimo_v2_5/mcq.jsonl \
  --dataset datasets/cinemme_reasoning_mcq.json
```

OEQ answers are judged by `Qwen/Qwen2.5-72B-Instruct` served with vLLM, on a 0–10 scale.
Empty responses and items whose judge output cannot be parsed after 10 attempts are excluded
from the mean; the paper reports `score_100` (mean × 10). The judge requires the clip's
`dense_caption`, included in release 1.1.0, and rejects missing context and duplicate
or unknown question IDs. Judge scores outside [0, 10] are treated as failed judgments.
Each completed judgment is flushed to the JSONL output immediately, in completion
order. The summary is written after all requests finish; a partial JSONL is not a
completed benchmark score. Use a new output path for each run.

```bash
vllm serve Qwen/Qwen2.5-72B-Instruct --port 8001 -tp 4
python -m evaluation.scoring.judge_oeq \
  --dataset datasets/cinemme_reasoning_oeq.json \
  --predictions outputs/mimo_v2_5/oeq.jsonl \
  --output outputs/mimo_v2_5/oeq_judged.jsonl \
  --summary-output outputs/mimo_v2_5/oeq_summary.json \
  --api-base http://localhost:8001/v1
```

Grounding:

```bash
python -m evaluation.scoring.score_grounding \
  --gt-file datasets/cinemme_grounding.json \
  --input-file outputs/mimo_v2_5/grounding.jsonl \
  --output-dir outputs/mimo_v2_5/grounding_eval
```

### Grounding scoring contract

The file passed to `--gt-file` is the authoritative reference. For official
comparisons use the complete `datasets/cinemme_grounding.json` (546 videos).
Predictions must carry the exact canonical `video_id`; `sample_id`, `query_id`,
and any submitted `gt` are ignored for matching and scoring. Duplicate or
unknown prediction video IDs, missing video IDs, and duplicate canonical video
IDs cause an error rather than silently replacing or discarding rows.

Headline metrics (`tIoU`, `vIoU`, `WER`) are macro averages over videos whose
prediction parsed, as in the paper; `parse_rate` reports that coverage and
should be shown next to them. `all_videos` repeats the metrics over every
canonical video, with missing or unparsable predictions scored as tIoU = 0,
vIoU = 0, WER = 1. `eval_results.json` includes an entry for every canonical
video, including failures. An explicit empty output list is a valid,
zero-detection prediction.

Prediction files may be JSONL, a JSON list, or a JSON object containing a
`results` list. Each row uses `response`, `pred`, or an `outputs` list. Responses
accept an output list or an object containing `outputs`/`segments`, including
JSON text. Invalid segment/box structures and non-finite numeric values count
as parse failures. Times are seconds; canonical `speaker_boxes` coordinates
are already normalized and are never automatically divided by 1000, including
out-of-range annotations flagged by dataset validation. Legacy box scaling for
prediction coordinates and the older ground-truth `boxes` field is retained.

Within each video, segments are matched greedily at tIoU >= 0.5; tIoU is
averaged over matched segments and WER is computed on the matched text
(lowercased, clipped to [0, 1]). These are conditional metrics, not recall.

vIoU scores the active-speaker face track over the whole clip, following the
continuous STVG definition: every box timestamp owns a 1 s time cell (split at
the midpoint with a closer neighbour and clipped to the clip), and vIoU is the
time integral of the box IoU where both tracks are present, divided by the
duration of the union of both tracks. Several boxes at one time are compared
by symmetric best-match IoU. Videos whose ground truth has no boxes are
excluded from vIoU. `sIoU` (box IoU at timestamps within 20 ms of a GT box) is
kept in the output for reference only.

Run the regression checks without API access or GPUs:

```bash
python -m unittest discover -s tests -v
```

## Structure

- `inference/run_mcq.py`, `inference/run_oeq.py`, `inference/run_grounding.py`
  call vLLM/OpenAI-compatible models.
- `scoring/score_mcq.py`, `scoring/judge_oeq.py`, `scoring/score_grounding.py`
  compute task metrics.
- `utils.py`, `prompts.py`, and `metrics.py` hold shared helpers.
