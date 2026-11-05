# Data Processing

Run the commands in this guide from `CineMME/` (`cd CineMME` from the Vidi
repository root).

This directory contains the construction code for CineMME: speech transcription,
face tracking, multimodal annotation, and question refinement.

## Structure

- `asr/run_whisperx.py`: extracts audio and runs WhisperX ASR with optional
  word-level alignment.
- `face/run_insightface_tracking.py`: runs InsightFace and assigns lightweight
  face track IDs from face embeddings.
- `annotation/run_multimodal_annotation.py`: packages video frames, ASR chunks,
  face tracks, and optional movie metadata for a Gemini/OpenAI-compatible
  multimodal annotator. It supports both video-analysis metadata and MCQ
  generation prompts.
- `refinement/refine_distractors.py`: refines MCQ distractors using model
  probing results.
- `prompts.py`: prompt templates for video analysis, MCQ generation,
  distractor refinement, OEQ conversion, and grounding.
- `schemas.py`: structured-output schemas for the above stages.
- `utils.py`: shared JSON, video, frame sampling, and normalization helpers.

## Typical Construction Flow

1. Download source videos from the released YouTube URLs.
2. Run ASR:

```bash
python -m data_process.asr.run_whisperx \
  --video local_videos/example.mp4 \
  --output runs/asr/example.json
```

3. Run face tracking:

```bash
python -m data_process.face.run_insightface_tracking \
  --video local_videos/example.mp4 \
  --output runs/faces/example_face_tracklets.json
```

4. Generate construction metadata:

```bash
python -m data_process.annotation.run_multimodal_annotation \
  --mode video-analysis \
  --video local_videos/example.mp4 \
  --asr-json runs/asr/example.json \
  --face-tracks-json runs/faces/example_face_tracklets.json \
  --output runs/annotations/example_video_analysis.json \
  --model <multimodal-model> \
  --api-base http://localhost:8000/v1
```

5. Generate and refine MCQs, then send candidates through human verification.

See [the dataset guide](../datasets/README.md) for released schemas and version notes.
