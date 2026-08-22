import copy
import json
from pathlib import Path


DEFAULT_GROUND_TRUTH_FILE = (
    Path(__file__).resolve().parent
    / "Character_Grounding"
    / "VUE-PLOT_Character_ground_truth.json"
)


def _load_json_list(path, description):
    with Path(path).open(encoding="utf-8") as file:
        records = json.load(file)
    if not isinstance(records, list):
        raise ValueError(f"{description} must be a JSON array")
    return records


def _index_by_video_id(records, description):
    indexed = {}
    for record in records:
        video_id = record.get("video_id")
        if not video_id:
            raise ValueError(f"{description} contains a record without video_id")
        if video_id in indexed:
            raise ValueError(f"{description} contains duplicate video_id: {video_id}")
        indexed[video_id] = record
    return indexed


def load_character_evaluation_examples(
    prediction_file, ground_truth_file=DEFAULT_GROUND_TRUTH_FILE
):
    ground_truth = _load_json_list(ground_truth_file, "ground truth")
    predictions = _load_json_list(prediction_file, "predictions")

    ground_truth_by_video = _index_by_video_id(ground_truth, "ground truth")
    predictions_by_video = _index_by_video_id(predictions, "predictions")

    unknown_video_ids = sorted(predictions_by_video.keys() - ground_truth_by_video.keys())
    if unknown_video_ids:
        preview = ", ".join(unknown_video_ids[:5])
        raise ValueError(f"predictions contain unknown video_id values: {preview}")

    evaluation_examples = [
        (
            copy.deepcopy(ground_truth_by_video[prediction["video_id"]]),
            copy.deepcopy(prediction),
        )
        for prediction in predictions
    ]
    evaluation_examples.extend(
        (copy.deepcopy(canonical_record), None)
        for video_id, canonical_record in ground_truth_by_video.items()
        if video_id not in predictions_by_video
    )
    return evaluation_examples
