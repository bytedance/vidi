"""Run InsightFace detection and lightweight face-track assignment."""

from __future__ import annotations

import argparse
import math
import queue
import threading
from pathlib import Path
from typing import Any

import numpy as np

from data_process.utils import save_json


class VideoReader:
    """Threaded OpenCV reader for sequential face detection."""

    def __init__(self, path: Path, queue_size: int = 64):
        import cv2  # type: ignore

        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {path}")
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.q: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self) -> None:
        while not self.stop_event.is_set():
            ok, frame = self.cap.read()
            if not ok:
                break
            self.q.put((ok, frame))
        self.q.put((False, None))
        self.cap.release()

    def read(self) -> tuple[bool, Any]:
        return self.q.get()

    def stop(self) -> None:
        self.stop_event.set()
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except queue.Empty:
                break
        self.thread.join(timeout=2)


class FaceIDAssigner:
    """Assign stable track IDs using cosine distance over face embeddings."""

    def __init__(self, threshold: float = 0.7, max_history: int = 10):
        self.threshold = threshold
        self.max_history = max_history
        self.face_db: dict[int, list[np.ndarray]] = {}
        self.next_id = 1

    @staticmethod
    def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
        return float(1.0 - np.dot(a, b) / denom)

    def assign(self, embedding: np.ndarray) -> int:
        best_id = None
        best_dist = math.inf
        for face_id, history in self.face_db.items():
            dist = self.cosine_distance(embedding, np.mean(history, axis=0))
            if dist < best_dist:
                best_dist = dist
                best_id = face_id
        if best_id is not None and best_dist < self.threshold:
            self.update(best_id, embedding)
            return best_id
        face_id = self.next_id
        self.face_db[face_id] = [embedding]
        self.next_id += 1
        return face_id

    def update(self, face_id: int, embedding: np.ndarray) -> None:
        history = self.face_db.setdefault(face_id, [])
        history.append(embedding)
        if len(history) > self.max_history:
            del history[:-self.max_history]


def load_insightface_app(model_name: str, ctx_id: int, det_size: tuple[int, int]) -> Any:
    import insightface  # type: ignore

    app = insightface.app.FaceAnalysis(name=model_name)
    app.prepare(ctx_id=ctx_id, det_size=det_size)
    return app


def track_faces(
    video_path: Path,
    output_json: Path,
    *,
    model_name: str = "buffalo_l",
    ctx_id: int = 0,
    det_size: tuple[int, int] = (640, 640),
    track_threshold: float = 0.7,
    sample_every: int = 1,
    save_patches_dir: Path | None = None,
    patch_interval_seconds: float = 1.0,
) -> dict[str, list[list[float]]]:
    """Detect faces and save tracklets as ``track_id -> frame/time/bbox``."""

    import cv2  # type: ignore

    app = load_insightface_app(model_name, ctx_id, det_size)
    reader = VideoReader(video_path)
    fps = reader.fps or 25.0
    patch_stride = max(1, int(round(fps * patch_interval_seconds)))
    assigner = FaceIDAssigner(threshold=track_threshold)
    tracks: dict[int, list[list[float]]] = {}
    frame_idx = 0

    try:
        while True:
            ok, frame = reader.read()
            if not ok:
                break
            if frame_idx % sample_every != 0:
                frame_idx += 1
                continue

            for face in app.get(frame):
                embedding = np.asarray(face.embedding, dtype=np.float32)
                embedding = embedding / (np.linalg.norm(embedding) or 1.0)
                face_id = assigner.assign(embedding)
                x1, y1, x2, y2 = [max(float(v), 0.0) for v in face.bbox]
                tracks.setdefault(face_id, []).append([frame_idx, frame_idx / fps, x1, y1, x2, y2])

                if save_patches_dir and frame_idx % patch_stride == 0:
                    h, w = frame.shape[:2]
                    px1, py1 = max(0, int(x1 * 0.9)), max(0, int(y1 * 0.9))
                    px2, py2 = min(w, int(x2 * 1.1)), min(h, int(y2 * 1.1))
                    patch_dir = save_patches_dir / str(face_id)
                    patch_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(patch_dir / f"fr_{frame_idx}_id_{face_id}.jpg"), frame[py1:py2, px1:px2])
            frame_idx += 1
    finally:
        reader.stop()

    payload = {str(face_id): rows for face_id, rows in tracks.items()}
    save_json(payload, output_json)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-name", default="buffalo_l")
    parser.add_argument("--ctx-id", type=int, default=0)
    parser.add_argument("--det-size", type=int, nargs=2, default=(640, 640))
    parser.add_argument("--track-threshold", type=float, default=0.7)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--save-patches-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    track_faces(
        args.video,
        args.output,
        model_name=args.model_name,
        ctx_id=args.ctx_id,
        det_size=tuple(args.det_size),
        track_threshold=args.track_threshold,
        sample_every=args.sample_every,
        save_patches_dir=args.save_patches_dir,
    )


if __name__ == "__main__":
    main()
