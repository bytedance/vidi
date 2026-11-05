"""Run WhisperX ASR for CineMME source videos."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any

import numpy as np

from data_process.utils import normalize_asr_segments, save_json


def extract_audio_from_video(video_path: Path, audio_path: Path, sample_rate: int = 16000) -> Path:
    """Extract 16 kHz mono WAV audio with PyAV."""

    import av  # type: ignore
    import soundfile as sf  # type: ignore

    audio_path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(video_path)) as container:
        audio_stream = next((stream for stream in container.streams if stream.type == "audio"), None)
        if audio_stream is None:
            raise ValueError(f"No audio stream found in {video_path}")
        resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)
        frames = []
        for frame in container.decode(audio_stream):
            for resampled in resampler.resample(frame):
                frames.append(resampled.to_ndarray())
    if not frames:
        raise ValueError(f"No audio frames decoded from {video_path}")
    audio = np.concatenate(frames, axis=1).T
    sf.write(str(audio_path), audio, sample_rate, subtype="PCM_16")
    return audio_path


def transcribe_with_whisperx(
    audio_path: Path,
    *,
    model_name: str = "medium.en",
    batch_size: int = 16,
    device: str | None = None,
    compute_type: str | None = None,
    align_words: bool = True,
) -> dict[str, Any]:
    """Transcribe audio and optionally run WhisperX word alignment."""

    import soundfile as sf  # type: ignore
    import torch  # type: ignore
    import whisperx  # type: ignore

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    compute_type = compute_type or ("float16" if device == "cuda" else "int8")
    model = whisperx.load_model(model_name, device, compute_type=compute_type, vad_model=None)

    audio_data, sample_rate = sf.read(str(audio_path), dtype="float32")
    if sample_rate != 16000:
        raise ValueError(f"Expected 16 kHz audio, got {sample_rate} Hz")
    audio_np = np.squeeze(audio_data)
    result = model.transcribe(audio_np, batch_size=batch_size)

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    if align_words:
        align_model, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(
            result["segments"],
            align_model,
            metadata,
            audio_np,
            device,
            return_char_alignments=False,
        )
        del align_model, metadata
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    result["segments"] = normalize_asr_segments(result)
    return result


def run_video_asr(video_path: Path, output_json: Path, audio_dir: Path | None = None, **kwargs: Any) -> dict[str, Any]:
    audio_dir = audio_dir or output_json.parent / "audio"
    audio_path = audio_dir / f"{video_path.stem}.wav"
    if not audio_path.exists():
        extract_audio_from_video(video_path, audio_path)
    result = transcribe_with_whisperx(audio_path, **kwargs)
    payload = {"video_id": video_path.stem, "segments": result["segments"]}
    save_json(payload, output_json)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audio-dir", type=Path)
    parser.add_argument("--model-name", default="medium.en")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device")
    parser.add_argument("--compute-type")
    parser.add_argument("--no-align", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_video_asr(
        args.video,
        args.output,
        audio_dir=args.audio_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
        compute_type=args.compute_type,
        align_words=not args.no_align,
    )


if __name__ == "__main__":
    main()
