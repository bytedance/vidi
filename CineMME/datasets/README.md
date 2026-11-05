# CineMME Dataset Files

This directory contains the released annotation files.
Each item uses `video_id` and `youtube_url` to identify the source video.
For the previous VUE-PLOT layout and the issue #36 answer correction, see the
[migration notes](../MIGRATION.md).

## Dataset validation

Run from `CineMME/` (Python standard library only):

```bash
python scripts/validate_dataset.py --output outputs/dataset_validation.json
```

The command does not edit annotations. It writes every detected issue with a
record ID and field location, then exits with status 1 if errors are present.
Checks cover required text, options and answers, duplicate IDs/questions/options,
grounding times and boxes, source membership, release counts, and file hashes.
Validation checks structure and consistency; it does not verify answer semantics
or source-video availability.

## Files

| File | Records | Description |
| --- | ---: | --- |
| `cinemme_grounding.json` | 546 videos | Grounding track with ASR-aligned speech segments and active-speaker face boxes. |
| `cinemme_reasoning_mcq.json` | 1,600 questions | Multiple-choice reasoning questions with six options and gold answer letters. |
| `cinemme_reasoning_oeq.json` | 316 questions | Open-ended questions, reference answers, reasoning traces, and dense captions for the judge. |
| `video_sources.json` | 975 videos | Unique YouTube source URLs and membership in the grounding/MCQ/OEQ files. |
| `release_stats.json` | 1 object | Basic release counts generated from the dataset files. |
| `croissant.json` | 1 object | Croissant metadata for dataset hosting. |

The reasoning tracks cover **667 unique videos**: 473 MCQ videos and 263 OEQ
videos share 69 IDs. The complete benchmark contains 975 unique videos.

## Version notes

Version 1.1.0 corrects grounding time bounds and box coordinates, revises eight
MCQ distractor sets, and includes the dense captions used to judge all 316 OEQs.
Preserve the exact input dataset and sampling configuration with each run.
Rescoring historical answers does not repeat inference with revised options.

Known annotation limitations:

- Duplicate-question pairs `mcq_000207` / `mcq_000378` and `mcq_000040` /
  `mcq_001463` are retained with different option orders.
- Grounding intervals `P2qz2B9snxE_seg_0007` and `e4Dlc6yqJuA_seg_0002` use
  estimated utterance durations rather than verified temporal boundaries.
- The revised distractors for `mcq_000123`, `mcq_000159`, `mcq_000274`,
  `mcq_000938`, `mcq_001145`, `mcq_001301`, `mcq_001483`, and `mcq_001577`
  were written manually and have not been rechecked against the videos.

## Source videos

Use the URLs in `video_sources.json` to locate source clips. Video availability
can change over time. The evaluation scripts accept videos stored under
`--video-root`; see [the evaluation guide](../evaluation/README.md) for supported
filenames and directory layouts.

## Grounding Schema

```json
{
  "sample_id": "grounding_000000",
  "video_id": "xY6bhYtH8Pw",
  "youtube_url": "https://www.youtube.com/watch?v=xY6bhYtH8Pw",
  "duration": 300.09,
  "task": "speaker_grounding",
  "modality": "audio_video",
  "query": "Transcribe ...",
  "segments": [
    {
      "segment_id": "xY6bhYtH8Pw_seg_0000",
      "start": 7.1,
      "end": 10.66,
      "transcript": "Underneath our feet",
      "speaker_boxes": [
        {"timestamp": 8.0, "bbox": [0.1, 0.2, 0.3, 0.4]}
      ]
    }
  ]
}
```

Bounding boxes use normalized `[x0, y0, x1, y1]` coordinates in `[0, 1]`.

## Reasoning MCQ Schema

```json
{
  "question_id": "mcq_000001",
  "video_id": "-W_4EZvbrEI",
  "youtube_url": "https://www.youtube.com/watch?v=-W_4EZvbrEI",
  "question": "...",
  "options": [{"label": "A", "text": "..."}],
  "answer": "C",
  "category": "Cinematic",
  "question_type": "Sound Design Analysis"
}
```

## Reasoning OEQ Schema

```json
{
  "question_id": "oeq_000002",
  "video_id": "vL21VCK_zLk",
  "youtube_url": "https://www.youtube.com/watch?v=vL21VCK_zLk",
  "question": "...",
  "reference_answer": "...",
  "reasoning_trace": "...",
  "dense_caption": "...",
  "category": "Narrative",
  "question_type": "Causal Chain Reconstruction"
}
```
