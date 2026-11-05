"""Shared helpers for CineMME data construction scripts."""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class VideoMetadata:
    video_id: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_video_metadata(video_path: Path) -> VideoMetadata:
    import cv2  # type: ignore

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0.0
    cap.release()
    return VideoMetadata(video_path.stem, width, height, fps, frame_count, duration)


def normalize_asr_segments(payload: Any) -> list[dict[str, Any]]:
    """Normalize ASR output from WhisperX or a prior JSON export."""

    segments = payload.get("segments", payload.get("chunks", payload)) if isinstance(payload, dict) else payload
    normalized = []
    for idx, seg in enumerate(segments or []):
        text = str(seg.get("text") or seg.get("transcript") or "").strip()
        if not text:
            continue
        normalized.append(
            {
                "segment_id": seg.get("segment_id", f"asr_{idx:04d}"),
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "transcript": text,
                "words": seg.get("words", []),
            }
        )
    return normalized


def normalize_face_tracks(face_tracks: Any) -> list[dict[str, Any]]:
    """Normalize face tracks to ``track_id`` plus timestamped bboxes."""

    if isinstance(face_tracks, dict):
        iterable = []
        for track_id, rows in face_tracks.items():
            detections = []
            for row in rows:
                if len(row) < 6:
                    continue
                detections.append(
                    {
                        "frame_index": int(row[0]),
                        "timestamp": float(row[1]),
                        "bbox": [float(v) for v in row[2:6]],
                        "confidence": 1.0,
                    }
                )
            iterable.append({"track_id": int(track_id), "detections": detections})
    else:
        iterable = face_tracks or []

    tracks = []
    for track in iterable:
        detections = []
        for det in track.get("detections", []):
            box = det.get("bbox") or det.get("box_2d")
            if not box or len(box) != 4:
                continue
            detections.append(
                {
                    "frame_index": int(det.get("frame_index", -1)),
                    "timestamp": float(det["timestamp"]),
                    "bbox": [float(v) for v in box],
                    "confidence": float(det.get("confidence", 1.0)),
                }
            )
        tracks.append({"track_id": int(track["track_id"]), "detections": detections})
    return tracks


def extract_json_from_text(text: str) -> dict[str, Any]:
    """Extract a JSON object from a model response."""

    if not text or not text.strip():
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if match:
        return json.loads(match.group(1))
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    return {}


def image_bytes_to_data_url(image_bytes: bytes, mime: str = "image/jpeg") -> str:
    return f"data:{mime};base64," + base64.b64encode(image_bytes).decode("utf-8")


def sample_video_frames(video_path: Path, max_frames: int = 16, max_side: int = 512) -> list[bytes]:
    """Uniformly sample JPEG frames from a video."""

    import cv2  # type: ignore
    import numpy as np

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        return []
    indices = np.linspace(0, frame_count - 1, min(max_frames, frame_count), dtype=int)
    frames = []
    for frame_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        scale = min(1.0, max_side / max(h, w))
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        ok, encoded = cv2.imencode(".jpg", frame)
        if ok:
            frames.append(encoded.tobytes())
    cap.release()
    return frames
