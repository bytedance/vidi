"""Shared helpers for CineMME evaluation scripts."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import subprocess
import tempfile
import threading
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **_: object):
        return iterable


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_dotenv(path: Path = Path(".env")) -> None:
    """Load KEY=VALUE lines from .env without overriding variables already set."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def is_openrouter(api_base: str) -> bool:
    return "openrouter.ai" in api_base


@lru_cache(maxsize=None)
def nested_video_index(video_root: Path) -> dict[str, Path]:
    index = {}
    for folder, _, files in sorted(os.walk(video_root, followlinks=True)):
        for name in sorted(files):
            path = Path(folder) / name
            if path.suffix in {".mp4", ".mkv", ".webm", ".mov"}:
                index.setdefault(path.stem, path)
    return index


def resolve_video_path(video_root: Path, video_id: str, template: str | None = None) -> Path:
    if template:
        path = Path(template.format(video_id=video_id))
        if path.exists():
            return path
    suffixes = (".mp4", ".mkv", ".webm", ".mov")
    for suffix in suffixes:
        path = video_root / f"{video_id}{suffix}"
        if path.exists():
            return path
    # Also accept nested layouts such as <video_root>/<year>/<video_id>.mkv.
    if path := nested_video_index(video_root).get(video_id):
        return path
    raise FileNotFoundError(f"Could not find local video for {video_id} under {video_root}")


def file_to_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("utf-8")


def ffmpeg(*args: str) -> None:
    """Run ffmpeg; the last arg is the output file. Existing outputs are reused (media cache)."""
    output = Path(args[-1])
    if output.exists():
        return
    partial = output.with_name(f".{os.getpid()}.{threading.get_ident()}.{output.name}")
    subprocess.run([os.environ.get("FFMPEG_BINARY", "ffmpeg"), "-y", "-loglevel", "error", *args[:-1], str(partial)],
                   check=True)
    os.replace(partial, output)  # atomic, so concurrent workers never read a half-written file


SCALE_480P = "scale='min(480,iw)':-2"
X264 = ["-c:v", "libx264", "-crf", "32", "-preset", "superfast"]


def media_content(video_path: Path, workdir: Path, use_audio: bool, transport: str, fps: float) -> list[dict[str, Any]]:
    """Build the multimodal message parts for one clip.

    - ``inline`` (vLLM server, local or remote): a 480p silent mp4 sampled by the server
      at ``fps`` plus a separate 16 kHz mono FLAC, both sent as base64 data URLs.
    - ``file``: the same two parts as ``file://`` paths; only for a server on the same
      machine, started with ``--allowed-local-media-path``.
    - ``base64`` (hosted APIs such as OpenRouter): one 480p mp4 re-encoded at ``fps``
      that carries the audio track, sent inline as a data URL.
    """
    if transport == "base64":
        clip = workdir / "clip.mp4"
        audio = ["-c:a", "aac", "-ac", "1", "-ar", "16000"] if use_audio else ["-an"]
        ffmpeg("-i", str(video_path), "-vf", f"fps={fps},{SCALE_480P}", *X264, *audio, str(clip))
        return [{"type": "video_url", "video_url": {"url": file_to_data_url(clip)}}]
    video = workdir / "video.mp4"
    ffmpeg("-i", str(video_path), "-vf", SCALE_480P, *X264, "-an", str(video))
    url = (lambda path: f"file://{path}") if transport == "file" else file_to_data_url
    content = [{"type": "video_url", "video_url": {"url": url(video)}}]
    if use_audio:
        # 16 kHz mono PCM; inline requests use lossless FLAC (same samples, ~half the bytes).
        audio = workdir / ("audio.wav" if transport == "file" else "audio.flac")
        ffmpeg("-i", str(video_path), "-vn", "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", str(audio))
        content.append({"type": "audio_url", "audio_url": {"url": url(audio)}})
    return content


def add_inference_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--video-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", help="Defaults to OPENROUTER_API_KEY for openrouter.ai, else VLLM_API_KEY or EMPTY.")
    parser.add_argument("--transport", choices=["inline", "file", "base64"], help="Default: base64 for openrouter.ai, else inline.")
    parser.add_argument("--video-template")
    parser.add_argument("--use-audio", action="store_true")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--num-frames", type=int, help="vLLM: uniformly sample this many frames across the clip, overriding --fps.")
    parser.add_argument("--chat-template-kwargs", type=json.loads, help="vLLM chat template parameters as a JSON object.")
    parser.add_argument("--max-tokens", type=int, default=16384, help="Output limit; 0 lets the server use remaining context.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=5, help="Retries for rate limits and server errors.")
    parser.add_argument("--limit", type=int, help="Only run the first N samples (smoke tests).")
    parser.add_argument("--media-cache", type=Path, help="Keep encoded clips here and reuse them across runs.")
    parser.add_argument("--provider", help="OpenRouter only: comma-separated provider order, no fallbacks (e.g. Xiaomi).")
    add_header_arg(parser)


def add_header_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--header", action="append", default=[], metavar="NAME:VALUE",
                        help="Extra HTTP header for every request (repeatable), e.g. for gateways in front of the server.")


def parse_headers(items: list[str]) -> dict[str, str]:
    return dict(item.split(":", 1) for item in items)


def make_client(args: argparse.Namespace):
    from openai import OpenAI

    load_dotenv()
    if args.api_key is None:
        if is_openrouter(args.api_base):
            args.api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("openroute_api_key")
            if not args.api_key:
                raise SystemExit("Set OPENROUTER_API_KEY (environment or .env) for openrouter.ai.")
        else:
            args.api_key = os.environ.get("VLLM_API_KEY", "EMPTY")
    if hasattr(args, "transport") and args.transport is None:
        args.transport = "base64" if is_openrouter(args.api_base) else "inline"
    return OpenAI(api_key=args.api_key, base_url=args.api_base, max_retries=args.max_retries, timeout=1800,
                  default_headers=parse_headers(args.header))


def run_requests(
    args: argparse.Namespace,
    id_field: str,
    build_prompt: Callable[[dict[str, Any]], str],
    extra_body: dict[str, Any] | None = None,
) -> None:
    """Query the model for every dataset item and append rows to ``args.output`` (JSONL).

    Rows with a non-empty response are kept on rerun; failed or empty rows are retried.
    """
    client = make_client(args)
    num_frames = getattr(args, "num_frames", None)
    if num_frames is not None and (num_frames <= 0 or args.transport == "base64"):
        raise ValueError("--num-frames requires a positive count and inline/file vLLM transport")
    if args.transport != "base64":
        # vLLM ignores fps inside video_url and otherwise samples only 32 frames.
        sampling = ({"num_frames": num_frames, "fps": -1, "video_backend": "opencv"}
                    if num_frames is not None else {"fps": args.fps, "num_frames": -1})
        extra_body = {**(extra_body or {}), "media_io_kwargs": {"video": sampling}}
    if template_kwargs := getattr(args, "chat_template_kwargs", None):
        extra_body = {**(extra_body or {}), "chat_template_kwargs": template_kwargs}
    if getattr(args, "provider", None):
        order = args.provider.split(",")
        extra_body = {**(extra_body or {}), "provider": {"order": order, "allow_fallbacks": False}}
    records = load_json(args.dataset)[: args.limit]
    done: list[dict[str, Any]] = []
    if args.output.exists():
        done = [row for row in load_jsonl(args.output) if row.get("response")]
    write_jsonl(done, args.output)
    done_ids = {row[id_field] for row in done}
    todo = [item for item in records if item[id_field] not in done_ids]

    def query(item: dict[str, Any]) -> dict[str, Any]:
        row = {id_field: item[id_field], "video_id": item["video_id"], "response": ""}
        try:
            video_path = resolve_video_path(args.video_root, item["video_id"], args.video_template)
            with tempfile.TemporaryDirectory(prefix="cinemme_") as tmp:
                workdir = Path(tmp)
                if args.media_cache:  # encoded clips are reused across questions and tracks
                    workdir = args.media_cache / f"{args.transport}_fps{args.fps}" / item["video_id"]
                    workdir.mkdir(parents=True, exist_ok=True)
                content = media_content(video_path, workdir, args.use_audio, args.transport, args.fps)
                content.append({"type": "text", "text": build_prompt(item)})
                kwargs = {
                    "model": args.model,
                    "messages": [{"role": "user", "content": content}],
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_tokens": args.max_tokens or None,
                }
                if extra_body:
                    kwargs["extra_body"] = extra_body
                response = client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            row.update(
                response=choice.message.content or "",
                finish_reason=choice.finish_reason,
                served_model=response.model,
                provider=(response.model_extra or {}).get("provider"),
                usage=response.usage.model_dump() if response.usage else None,
            )
        except Exception as exc:  # keep going; the row is retried on the next run
            row["error"] = f"{type(exc).__name__}: {exc}"[:2000]
        return row

    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool, args.output.open("a", encoding="utf-8") as fout:
        futures = [pool.submit(query, item) for item in todo]
        for future in tqdm(as_completed(futures), total=len(futures), dynamic_ncols=True):
            row = future.result()
            failed += not row["response"]
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
    print(json.dumps({"total": len(records), "reused": len(done), "queried": len(todo), "empty_or_failed": failed}))
