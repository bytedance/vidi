"""Score CineMME grounding predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.utils import load_json, load_jsonl, save_json
from evaluation.metrics import aggregate_grounding_results, evaluate_grounding_sample, parse_grounding_response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-file", "-g", required=True, type=Path)
    parser.add_argument("--input-file", "-i", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/grounding_eval"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, details = score_predictions(load_json(args.gt_file), _load_predictions(args.input_file))
    save_json(details, args.output_dir / "eval_results.json")
    save_json(summary, args.output_dir / "eval_summary.json")
    print(json.dumps(summary, indent=2))


def score_predictions(gold_items: list[dict], predictions: list[dict]) -> tuple[dict, list[dict]]:
    gold_by_video = _index_by_video(gold_items, "ground truth")
    if not gold_by_video:
        raise ValueError("ground truth must contain at least one video")
    pred_by_video = _index_by_video(predictions, "predictions")
    unknown = pred_by_video.keys() - gold_by_video.keys()
    if unknown:
        raise ValueError(f"unknown prediction video_id(s): {', '.join(sorted(unknown))}")

    sample_results = []
    details = []

    missing_predictions = 0
    parse_failures = 0
    for item in gold_items:
        pred = pred_by_video.get(item["video_id"])
        detail = {
            "sample_id": item.get("sample_id", item["video_id"]),
            "video_id": item["video_id"],
            "parse_success": False,
        }
        outputs = []
        if pred is None:
            missing_predictions += 1
            detail.update(status="missing_prediction", error="no prediction for canonical video_id")
        else:
            try:
                outputs = parse_grounding_response(pred.get("response", pred.get("pred", pred.get("outputs"))))
                detail.update(status="ok", parse_success=True)
            except ValueError as exc:
                parse_failures += 1
                detail.update(status="parse_failure", error=str(exc))

        result = evaluate_grounding_sample(item, outputs)
        sample_results.append(result)
        detail.update(metrics=result["metrics"], matches=result["matches"])
        details.append(detail)

    # Headline metrics average over videos with a parsed prediction (as in the paper);
    # `all_videos` also counts missing/unparsable predictions as zero-score videos.
    parsed_results = [r for r, d in zip(sample_results, details) if d["parse_success"]]
    summary = aggregate_grounding_results(parsed_results)
    summary["parse_rate"] = len(parsed_results) / len(gold_by_video)
    summary["all_videos"] = {k: v for k, v in aggregate_grounding_results(sample_results).items()
                             if k in ("tIoU", "vIoU", "sIoU", "WER", "samples", "vIoU_samples")}
    summary["predicted_samples"] = len(pred_by_video)
    summary["evaluated_samples"] = len(sample_results)
    summary["parsed_samples"] = len(pred_by_video) - parse_failures
    summary["missing_predictions"] = missing_predictions
    summary["parse_failures"] = parse_failures
    summary["gold_samples"] = len(gold_by_video)
    return summary, details


def _load_predictions(path: Path) -> list[dict]:
    data = load_jsonl(path) if path.suffix == ".jsonl" else load_json(path)
    if isinstance(data, dict) and "results" in data:
        data = data["results"]
    if not isinstance(data, list):
        raise ValueError("grounding predictions must be a list or an object with a 'results' list")
    return data


def _index_by_video(records: list[dict], source: str) -> dict[str, dict]:
    if not isinstance(records, list):
        raise ValueError(f"{source} must be a list")
    indexed = {}
    for index, record in enumerate(records):
        video_id = record.get("video_id") if isinstance(record, dict) else None
        if not isinstance(video_id, str) or not video_id.strip():
            raise ValueError(f"{source} row {index}: missing or invalid video_id")
        if video_id in indexed:
            raise ValueError(f"{source}: duplicate video_id {video_id!r}")
        indexed[video_id] = record
    return indexed


if __name__ == "__main__":
    main()
