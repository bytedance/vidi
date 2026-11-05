"""Metric utilities for CineMME evaluation.

The grounding evaluator intentionally follows the original ICML submission
script in ``/evaluation/eval.py``: temporal segments are first matched by a
greedy tIoU >= 0.5 rule, WER is computed on the concatenated matched text, and
speaker-box IoU is averaged only over timestamp-matched box pairs within 20 ms.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class GroundingSegment:
    start: float
    end: float
    text: str
    boxes: list[dict[str, Any]]


def temporal_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    inter_start = max(a_start, b_start)
    inter_end = min(a_end, b_end)
    inter = max(0.0, inter_end - inter_start)
    len_a = max(0.0, a_end - a_start)
    len_b = max(0.0, b_end - b_start)
    union = len_a + len_b - inter
    return inter / union if union > 0 else 0.0


def box_iou(box_a: list[float], box_b: list[float]) -> float:
    if len(box_a) != 4 or len(box_b) != 4:
        return 0.0
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def word_error_rate(reference: str, hypothesis: str) -> float:
    reference = (reference or "").lower()
    hypothesis = (hypothesis or "").lower()
    if not reference.strip():
        return 1.0

    try:
        import jiwer

        wer = jiwer.wer(reference, hypothesis)
    except ImportError:
        wer = _whitespace_wer(reference, hypothesis)
    return max(0.0, min(1.0, float(wer)))


def parse_grounding_response(response: str | dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Parse a model grounding response into its ``outputs`` list.

    Raises:
        ValueError: if the response cannot be parsed as structured JSON.
    """

    if isinstance(response, str):
        if not response.strip():
            raise ValueError("empty grounding response")
        try:
            payload = json.loads(response)
        except json.JSONDecodeError as exc:
            match = re.search(r"\{.*\}", response, flags=re.DOTALL)
            if match is None:
                raise ValueError("grounding response is not valid JSON") from exc
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError as nested_exc:
                raise ValueError("embedded grounding JSON is invalid") from nested_exc
    elif isinstance(response, (dict, list)):
        payload = response
    else:
        raise ValueError(f"unsupported grounding response type: {type(response).__name__}")

    # Some models wrap the whole object in a one-element list: [{"outputs": [...]}].
    if isinstance(payload, list) and len(payload) == 1 and isinstance(payload[0], dict) \
            and ("outputs" in payload[0] or "segments" in payload[0]):
        payload = payload[0]
    if isinstance(payload, list):
        outputs = payload
    elif isinstance(payload, dict):
        outputs = payload.get("outputs", payload.get("segments"))
    else:
        raise ValueError("grounding response JSON must be a list or object")

    if not isinstance(outputs, list):
        raise ValueError("grounding outputs must be a list")
    for index, segment in enumerate(outputs):
        try:
            if not isinstance(segment, dict):
                raise ValueError("segment must be an object")
            for field in ("start", "end"):
                if isinstance(segment.get(field), bool) or not math.isfinite(float(segment[field])):
                    raise ValueError(f"{field} must be finite seconds")
            if not isinstance(segment.get("text", segment.get("transcript")), str):
                raise ValueError("text/transcript must be a string")
            boxes = segment.get("boxes", segment.get("speaker_boxes", []))
            if not isinstance(boxes, list):
                raise ValueError("boxes must be a list")
            for box in boxes:
                if not isinstance(box, dict):
                    raise ValueError("box must be an object")
                if isinstance(box.get("timestamp"), bool) or not math.isfinite(float(box["timestamp"])):
                    raise ValueError("box timestamp must be finite seconds")
                coords = box.get("box_2d", box.get("bbox"))
                if not isinstance(coords, (list, tuple)) or len(coords) != 4:
                    raise ValueError("box coordinates must contain four numbers")
                if any(isinstance(coord, bool) or not math.isfinite(float(coord)) for coord in coords):
                    raise ValueError("box coordinates must be finite numbers")
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"invalid grounding segment {index}: {exc}") from exc
    return outputs


def evaluate_grounding_sample(gt_item: dict[str, Any], pred_outputs: list[dict[str, Any]]) -> dict[str, Any]:
    gt_segments = _prepare_gt_segments(gt_item)
    pred_segments = _prepare_pred_segments(pred_outputs)
    result = compare_grounding_segments(pred_segments, gt_segments)
    duration = _as_float(gt_item.get("duration"), 0.0) or max((seg.end for seg in gt_segments), default=0.0)
    result["metrics"]["video_iou"] = video_iou(_frame_map(pred_segments), _frame_map(gt_segments), duration)
    return result


def video_iou(
    pred_frames: dict[float, list[list[float]]],
    gt_frames: dict[float, list[list[float]]],
    duration: float,
    fps: float = 1.0,
) -> float | None:
    """Continuous spatio-temporal IoU of the active-speaker face tube (STVG vIoU).

    Each box timestamp owns a time cell of width 1/fps (split at the midpoint with a
    closer neighbour, clipped to the clip). vIoU = integral over time of the box-group
    IoU where both tubes are present, divided by the duration of the union of the two
    tubes. Returns None when the ground truth has no boxes.
    """
    gt_cells = _tube_cells(gt_frames, fps, duration)
    if not gt_cells:
        return None
    pred_cells = _tube_cells(pred_frames, fps, duration)
    union = sum(c[1] - c[0] for c in gt_cells) + sum(c[1] - c[0] for c in pred_cells)
    integral = 0.0
    i = j = 0
    while i < len(pred_cells) and j < len(gt_cells):
        (p_start, p_end, p_boxes), (g_start, g_end, g_boxes) = pred_cells[i], gt_cells[j]
        overlap = min(p_end, g_end) - max(p_start, g_start)
        if overlap > 0:
            union -= overlap
            integral += overlap * _box_group_iou(p_boxes, g_boxes)
        if p_end < g_end:
            i += 1
        else:
            j += 1
    return integral / union if union > 0 else 0.0


def _frame_map(segments: list[GroundingSegment]) -> dict[float, list[list[float]]]:
    frames: dict[float, list[list[float]]] = {}
    for segment in segments:
        for box in segment.boxes:
            if len(box.get("box_2d", [])) == 4:
                frames.setdefault(float(box["timestamp"]), []).append(box["box_2d"])
    return frames


def _tube_cells(
    frames: dict[float, list[list[float]]], fps: float, duration: float
) -> list[tuple[float, float, list[list[float]]]]:
    half = 0.5 / fps
    times = sorted(frames)
    cells = []
    for k, t in enumerate(times):
        start, end = t - half, t + half
        if k > 0 and times[k - 1] + half > start:
            start = max(start, (times[k - 1] + t) / 2)
        if k + 1 < len(times) and end > times[k + 1] - half:
            end = min(end, (t + times[k + 1]) / 2)
        start, end = min(duration, max(0.0, start)), min(duration, max(0.0, end))
        if end > start:
            cells.append((start, end, frames[t]))
    return cells


def _box_group_iou(left: list[list[float]], right: list[list[float]]) -> float:
    """Symmetric best-match IoU between the boxes present at one time on each side."""
    def directional(source: list[list[float]], target: list[list[float]]) -> float:
        return sum(max(box_iou(a, b) for b in target) for a in source) / len(source)

    return (directional(left, right) + directional(right, left)) / 2


def compare_grounding_segments(
    pred_segments: list[GroundingSegment],
    gt_segments: list[GroundingSegment],
    iou_threshold: float = 0.5,
    bbox_time_tolerance: float = 0.02,
) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    matched_pred_indices: set[int] = set()

    for gt_seg in gt_segments:
        best_iou = -1.0
        best_pred_idx = -1
        for pred_idx, pred_seg in enumerate(pred_segments):
            if pred_idx in matched_pred_indices:
                continue
            score = temporal_iou(gt_seg.start, gt_seg.end, pred_seg.start, pred_seg.end)
            if score > best_iou:
                best_iou = score
                best_pred_idx = pred_idx

        if best_iou >= iou_threshold:
            matches.append({"gt": gt_seg, "pred": pred_segments[best_pred_idx], "iou": best_iou})
            matched_pred_indices.add(best_pred_idx)

    if not matches:
        overall_wer = word_error_rate(
            " ".join(seg.text for seg in gt_segments),
            " ".join(seg.text for seg in pred_segments),
        )
        return {
            "metrics": {
                "temporal_iou_avg": 0.0,
                "word_error_rate": 1.0,
                "overall_word_accuracy": 1.0 - overall_wer,
                "overall_word_error": overall_wer,
                "average_box_iou": 0.0,
                "total_gt_segments": len(gt_segments),
                "total_pred_segments": len(pred_segments),
                "matched_segments": 0,
            },
            "matches": [],
        }

    total_iou = 0.0
    total_box_iou = 0.0
    box_matches = 0
    gt_full_text: list[str] = []
    pred_full_text: list[str] = []

    for match in matches:
        gt_seg: GroundingSegment = match["gt"]
        pred_seg: GroundingSegment = match["pred"]
        total_iou += float(match["iou"])
        gt_full_text.append(gt_seg.text)
        pred_full_text.append(pred_seg.text)

        if gt_seg.boxes and pred_seg.boxes:
            for gt_box in gt_seg.boxes:
                min_ts_diff = float("inf")
                best_pred_box: dict[str, Any] | None = None
                gt_time = _as_float(gt_box.get("timestamp"), 0.0)
                for pred_box in pred_seg.boxes:
                    ts_diff = abs(gt_time - _as_float(pred_box.get("timestamp"), 0.0))
                    if ts_diff < min_ts_diff:
                        min_ts_diff = ts_diff
                        best_pred_box = pred_box

                if best_pred_box is not None and float(min_ts_diff) < bbox_time_tolerance:
                    total_box_iou += box_iou(gt_box.get("box_2d", []), best_pred_box.get("box_2d", []))
                    box_matches += 1

    wer = word_error_rate(" ".join(gt_full_text), " ".join(pred_full_text))
    overall_wer = word_error_rate(
        " ".join(seg.text for seg in gt_segments),
        " ".join(seg.text for seg in pred_segments),
    )
    average_box_iou = total_box_iou / box_matches if box_matches > 0 else 0.0

    return {
        "metrics": {
            "temporal_iou_avg": total_iou / len(matches),
            "average_box_iou": average_box_iou,
            "word_error_rate": wer,
            "overall_word_accuracy": 1.0 - overall_wer,
            "overall_word_error": overall_wer,
            "total_gt_segments": len(gt_segments),
            "total_pred_segments": len(pred_segments),
            "matched_segments": len(matches),
        },
        "matches": [
            {
                "gt": asdict(match["gt"]),
                "pred": asdict(match["pred"]),
                "iou": match["iou"],
            }
            for match in matches
        ],
    }


def aggregate_grounding_results(sample_results: list[dict[str, Any]]) -> dict[str, float]:
    vious = [r["metrics"]["video_iou"] for r in sample_results if r["metrics"].get("video_iou") is not None]
    total_metrics = {
        "temporal_iou_avg": 0.0,
        "average_box_iou": 0.0,
        "word_error_rate": 0.0,
        "overall_word_accuracy": 0.0,
        "overall_word_error": 0.0,
        "total_gt_segments": 0.0,
        "total_pred_segments": 0.0,
        "matched_segments": 0.0,
    }
    for result in sample_results:
        for key, value in result["metrics"].items():
            if key in total_metrics:
                total_metrics[key] += float(value)

    sample_count = len(sample_results)
    if sample_count > 0:
        for key in total_metrics:
            if "total" not in key and "matched" not in key:
                total_metrics[key] /= sample_count
    else:
        total_metrics["word_error_rate"] = 1.0
        total_metrics["overall_word_error"] = 1.0

    total_metrics["tIoU"] = total_metrics["temporal_iou_avg"]
    # vIoU is averaged over videos whose ground truth has face boxes.
    total_metrics["vIoU"] = sum(vious) / len(vious) if vious else 0.0
    total_metrics["vIoU_samples"] = float(len(vious))
    total_metrics["sIoU"] = total_metrics["average_box_iou"]  # legacy 20 ms box match, kept for reference
    total_metrics["WER"] = total_metrics["word_error_rate"]
    total_metrics["samples"] = float(sample_count)
    total_metrics["segments"] = total_metrics["total_gt_segments"]
    return total_metrics


def summarize_grounding(sample_results: list[dict[str, Any]]) -> dict[str, float]:
    return aggregate_grounding_results(sample_results)


def _prepare_gt_segments(gt_item: dict[str, Any]) -> list[GroundingSegment]:
    raw_segments = gt_item.get("segments")
    if raw_segments is None:
        raw_segments = gt_item.get("gt") if isinstance(gt_item.get("gt"), list) else gt_item.get("json_gt", [])
    return [_prepare_segment(segment, is_prediction=False) for segment in raw_segments]


def _prepare_pred_segments(pred_outputs: list[dict[str, Any]]) -> list[GroundingSegment]:
    return [_prepare_segment(segment, is_prediction=True) for segment in pred_outputs]


def _prepare_segment(raw_segment: dict[str, Any], is_prediction: bool) -> GroundingSegment:
    if not isinstance(raw_segment, dict):
        raw_segment = {}
    return GroundingSegment(
        start=_as_float(raw_segment.get("start"), 0.0),
        end=_as_float(raw_segment.get("end"), 0.0),
        text=str(raw_segment.get("text", raw_segment.get("transcript", ""))),
        boxes=_prepare_boxes(raw_segment, is_prediction=is_prediction),
    )


def _prepare_boxes(raw_segment: dict[str, Any], is_prediction: bool) -> list[dict[str, Any]]:
    raw_boxes = raw_segment.get("boxes")
    if raw_boxes is None:
        raw_boxes = raw_segment.get("speaker_boxes", [])
    boxes = []
    for raw_box in raw_boxes or []:
        if not isinstance(raw_box, dict):
            continue
        default_timestamp = 0.0 if is_prediction else raw_box.get("timestamp", 0.0)
        boxes.append(
            {
                "timestamp": _as_float(raw_box.get("timestamp"), _as_float(default_timestamp, 0.0)),
                "box_2d": _normalize_box(
                    raw_box.get("box_2d", raw_box.get("bbox", [])),
                    rescale=is_prediction or "boxes" in raw_segment,
                ),
            }
        )
    return boxes


def _normalize_box(raw_box: Any, rescale: bool = True) -> list[float]:
    if not isinstance(raw_box, (list, tuple)) or len(raw_box) < 4:
        return []
    try:
        box = [float(coord) for coord in raw_box[:4]]
    except (TypeError, ValueError):
        return []
    if rescale and any(coord > 1.0 for coord in box):
        box = [coord / 1000.0 for coord in box]
    return box


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _whitespace_wer(reference: str, hypothesis: str) -> float:
    ref = reference.split()
    hyp = hypothesis.split()
    if not ref:
        return 1.0
    dp = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        dp[i][0] = i
    for j in range(len(hyp) + 1):
        dp[0][j] = j
    for i, ref_word in enumerate(ref, start=1):
        for j, hyp_word in enumerate(hyp, start=1):
            cost = 0 if ref_word == hyp_word else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[-1][-1] / len(ref)
