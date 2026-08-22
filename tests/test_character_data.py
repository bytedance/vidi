import json
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "VUE_PLOT"))

from character_data import load_character_evaluation_examples


class CharacterEvaluationDataTest(unittest.TestCase):
    def write_json(self, directory, name, records):
        path = Path(directory) / name
        path.write_text(json.dumps(records), encoding="utf-8")
        return path

    def test_canonical_ground_truth_controls_order_and_coverage(self):
        ground_truth = [
            {"query_id": 10, "video_id": "video-a", "gt": ["canonical-a"]},
            {"query_id": 11, "video_id": "video-b", "gt": ["canonical-b"]},
        ]
        predictions = [
            {
                "query_id": 999,
                "video_id": "video-a",
                "gt": ["submitted-ground-truth"],
                "pred": ["prediction-a"],
            }
        ]

        with tempfile.TemporaryDirectory() as directory:
            ground_truth_file = self.write_json(directory, "ground-truth.json", ground_truth)
            prediction_file = self.write_json(directory, "predictions.json", predictions)
            examples = load_character_evaluation_examples(
                prediction_file, ground_truth_file
            )

        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0][0]["gt"], ["canonical-a"])
        self.assertEqual(examples[0][1]["pred"], ["prediction-a"])
        self.assertIsNone(examples[1][1])

    def test_duplicate_prediction_video_id_is_rejected(self):
        ground_truth = [{"video_id": "video-a", "gt": []}]
        predictions = [
            {"video_id": "video-a", "pred": []},
            {"video_id": "video-a", "pred": []},
        ]

        with tempfile.TemporaryDirectory() as directory:
            ground_truth_file = self.write_json(directory, "ground-truth.json", ground_truth)
            prediction_file = self.write_json(directory, "predictions.json", predictions)
            with self.assertRaisesRegex(ValueError, "duplicate video_id"):
                load_character_evaluation_examples(prediction_file, ground_truth_file)

    def test_unknown_prediction_video_id_is_rejected(self):
        ground_truth = [{"video_id": "video-a", "gt": []}]
        predictions = [{"video_id": "video-unknown", "pred": []}]

        with tempfile.TemporaryDirectory() as directory:
            ground_truth_file = self.write_json(directory, "ground-truth.json", ground_truth)
            prediction_file = self.write_json(directory, "predictions.json", predictions)
            with self.assertRaisesRegex(ValueError, "unknown video_id"):
                load_character_evaluation_examples(prediction_file, ground_truth_file)


if __name__ == "__main__":
    unittest.main()
