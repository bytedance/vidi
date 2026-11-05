"""Generate CineMME construction annotations with a multimodal LLM.

This script keeps the core Gemini/OpenAI-compatible annotation logic from the
original pipeline: it packages video frames, ASR chunks, face tracks, and
optional movie metadata, then asks a multimodal model to produce character
identification, speaker diarization, narrative segments, or MCQs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from data_process.utils import (
    extract_json_from_text,
    image_bytes_to_data_url,
    load_json,
    normalize_asr_segments,
    normalize_face_tracks,
    read_video_metadata,
    sample_video_frames,
    save_json,
)
from data_process.prompts import QA_GENERATION_PROMPT, VIDEO_ANALYSIS_PROMPT
from data_process.schemas import QA_RESPONSE_SCHEMA, VIDEO_ANALYSIS_SCHEMA


def compact_face_tracks_for_prompt(face_tracks: Any, width: int, height: int, max_frames_per_track: int = 8) -> dict[str, Any]:
    prompt_tracks: dict[str, Any] = {}
    for track in normalize_face_tracks(face_tracks):
        detections = track["detections"]
        if not detections:
            continue
        stride = max(1, len(detections) // max_frames_per_track)
        frames = []
        for det in detections[::stride][:max_frames_per_track]:
            x1, y1, x2, y2 = det["bbox"]
            frames.append(
                {
                    "time": round(det["timestamp"], 3),
                    "bbox": [
                        round(x1 / width, 4),
                        round(y1 / height, 4),
                        round(x2 / width, 4),
                        round(y2 / height, 4),
                    ],
                }
            )
        prompt_tracks[str(track["track_id"])] = {"face_id": track["track_id"], "frames": frames}
    return prompt_tracks


def build_video_analysis_prompt(video_path: Path, asr_json: Path, face_tracks_json: Path, metadata_json: Path | None) -> str:
    meta = read_video_metadata(video_path)
    asr_chunks = normalize_asr_segments(load_json(asr_json))
    face_tracks = compact_face_tracks_for_prompt(load_json(face_tracks_json), meta.width, meta.height)
    movie_metadata = load_json(metadata_json) if metadata_json else {}
    return VIDEO_ANALYSIS_PROMPT.substitute(
        metadata=json.dumps(movie_metadata, ensure_ascii=False, indent=2),
        asr_chunks=json.dumps(asr_chunks, ensure_ascii=False, indent=2),
        face_tracks=json.dumps(face_tracks, ensure_ascii=False, indent=2),
    )


def build_qa_prompt(metadata_json: Path | None, asr_json: Path, video_analysis_json: Path, num_questions: int) -> str:
    movie_metadata = load_json(metadata_json) if metadata_json else {}
    asr_chunks = normalize_asr_segments(load_json(asr_json))
    video_analysis = load_json(video_analysis_json)
    return QA_GENERATION_PROMPT.substitute(
        metadata=json.dumps(movie_metadata, ensure_ascii=False, indent=2),
        asr_chunks=json.dumps(asr_chunks, ensure_ascii=False, indent=2),
        video_analysis=json.dumps(video_analysis, ensure_ascii=False, indent=2),
        num_questions=num_questions,
    )


def structured_extra_body(schema: dict[str, Any], dialect: str) -> dict[str, Any] | None:
    if dialect == "none":
        return None
    if dialect == "structured_outputs":
        return {"structured_outputs": {"json": schema}}
    if dialect == "guided_json":
        return {"guided_json": schema}
    if dialect == "response_format":
        return {"response_format": {"type": "json_object"}}
    raise ValueError(f"Unsupported structured output dialect: {dialect}")


def call_multimodal_model(
    *,
    client: Any,
    model: str,
    prompt: str,
    frames: list[bytes],
    schema: dict[str, Any],
    structured_output: str,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for frame in frames:
        content.append({"type": "image_url", "image_url": {"url": image_bytes_to_data_url(frame)}})

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    extra_body = structured_extra_body(schema, structured_output)
    if extra_body:
        kwargs["extra_body"] = extra_body
    response = client.chat.completions.create(**kwargs)
    return extract_json_from_text(response.choices[0].message.content or "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["video-analysis", "qa-generation"], required=True)
    parser.add_argument("--video", type=Path, help="Video path. Required for video-analysis.")
    parser.add_argument("--asr-json", type=Path, required=True)
    parser.add_argument("--face-tracks-json", type=Path, help="Required for video-analysis.")
    parser.add_argument("--video-analysis-json", type=Path, help="Required for qa-generation.")
    parser.add_argument("--metadata-json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--write-prompt-only", action="store_true")
    parser.add_argument("--num-questions", type=int, default=15)
    parser.add_argument("--model")
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--structured-output", choices=["structured_outputs", "guided_json", "response_format", "none"], default="structured_outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "video-analysis":
        if not args.video or not args.face_tracks_json:
            raise ValueError("--video and --face-tracks-json are required for video-analysis")
        prompt = build_video_analysis_prompt(args.video, args.asr_json, args.face_tracks_json, args.metadata_json)
        schema = VIDEO_ANALYSIS_SCHEMA
        frames = sample_video_frames(args.video, max_frames=args.max_frames)
    else:
        if not args.video_analysis_json:
            raise ValueError("--video-analysis-json is required for qa-generation")
        prompt = build_qa_prompt(args.metadata_json, args.asr_json, args.video_analysis_json, args.num_questions)
        schema = QA_RESPONSE_SCHEMA
        frames = sample_video_frames(args.video, max_frames=args.max_frames) if args.video else []

    if args.write_prompt_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(prompt, encoding="utf-8")
        return

    if not args.model:
        raise ValueError("--model is required unless --write-prompt-only is set")
    from openai import OpenAI

    client = OpenAI(api_key=args.api_key, base_url=args.api_base)
    result = call_multimodal_model(
        client=client,
        model=args.model,
        prompt=prompt,
        frames=frames,
        schema=schema,
        structured_output=args.structured_output,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    save_json(result, args.output)


if __name__ == "__main__":
    main()
