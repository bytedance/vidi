"""Run with: python -m unittest discover -s tests -v."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from evaluation.metrics import evaluate_grounding_sample, parse_grounding_response
from evaluation.scoring.score_grounding import score_predictions
from scripts.validate_dataset import validate


def gold_sample(video="video000001"):
    return {
        "sample_id": "grounding_" + video,
        "video_id": video,
        "youtube_url": "https://www.youtube.com/watch?v=" + video,
        "duration": 2.0,
        "query": "Transcribe and track the speaker.",
        "task": "speaker_grounding",
        "modality": "audio_video",
        "segments": [{
            "segment_id": video + "_seg_0000", "start": 0.0, "end": 1.0, "transcript": "hello world",
            "speaker_boxes": [{"timestamp": 0.5, "bbox": [0.1, 0.2, 0.4, 0.8]}],
        }],
    }


def prediction(video="video000001"):
    return {
        "video_id": video, "query_id": "unstable_submission_id", "sample_id": "also_unstable",
        "gt": [{"start": 30, "end": 40, "text": "conflicting submitted ground truth"}],
        "outputs": [{"start": 0, "end": 1, "text": "hello world",
                     "boxes": [{"timestamp": 0.5, "box_2d": [0.1, 0.2, 0.4, 0.8]}]}],
    }


class GroundingChecks(unittest.TestCase):
    def test_canonical_alignment_and_denominators(self):
        gold = [gold_sample(), gold_sample("video000002")]
        summary, details = score_predictions(gold, [prediction()])
        self.assertEqual((summary["samples"], summary["gold_samples"], summary["evaluated_samples"]), (1, 2, 2))
        self.assertEqual((summary["tIoU"], summary["vIoU"], summary["WER"], summary["parse_rate"]), (1, 1, 0, 0.5))
        all_videos = summary["all_videos"]
        self.assertEqual((all_videos["samples"], all_videos["tIoU"], all_videos["vIoU"], all_videos["WER"]), (2, 0.5, 0.5, 0.5))
        self.assertEqual(summary["segments"], 1)  # GT segments of the parsed videos
        self.assertEqual(summary["missing_predictions"], 1)
        self.assertEqual(details[1]["status"], "missing_prediction")
        self.assertEqual(details[0]["sample_id"], gold[0]["sample_id"])
        for response in ("not JSON", "", {}, {"outputs": "bad"}, {"outputs": [None]},
                         {"outputs": [{"start": "NaN", "end": 1, "text": "hello"}]},
                         {"outputs": [{"start": 0, "end": 1, "text": "hello", "boxes": 5}]}):
            with self.subTest(response=response):
                bad = {"video_id": "video000002", "response": response}
                summary, details = score_predictions(gold, [prediction(), bad])
                self.assertEqual((summary["tIoU"], summary["vIoU"], summary["WER"]), (1, 1, 0))
                self.assertEqual(summary["all_videos"]["tIoU"], 0.5)
                self.assertEqual(summary["parse_failures"], 1)
                self.assertEqual(details[1]["status"], "parse_failure")
        summary, _ = score_predictions(gold, [])
        self.assertEqual((summary["tIoU"], summary["sIoU"], summary["WER"]), (0, 0, 1))
        self.assertEqual(summary["missing_predictions"], 2)

    def test_reject_duplicate_unknown_and_missing_ids(self):
        gold = [gold_sample()]
        for preds, message in (([prediction(), prediction()], "duplicate video_id"),
                               ([prediction("unknown0001")], "unknown prediction"),
                               ([{"sample_id": gold[0]["sample_id"]}], "invalid video_id"),
                               ([None], "invalid video_id")):
            with self.subTest(preds=preds), self.assertRaisesRegex(ValueError, message):
                score_predictions(gold, preds)
        with self.assertRaisesRegex(ValueError, "duplicate video_id"):
            score_predictions(gold + gold, [])
        with self.assertRaisesRegex(ValueError, "at least one"):
            score_predictions([], [])

    def test_formats_and_box_units(self):
        outputs = prediction()["outputs"]
        for response in (outputs, {"outputs": outputs}, {"segments": outputs}, json.dumps(outputs),
                         "```json\n" + json.dumps({"outputs": outputs}) + "\n```"):
            self.assertEqual(parse_grounding_response(response), outputs)
        scaled = deepcopy(outputs)
        scaled[0]["boxes"][0]["box_2d"] = [100, 200, 400, 800]
        self.assertEqual(evaluate_grounding_sample(gold_sample(), scaled)["metrics"]["average_box_iou"], 1)
        gold = gold_sample()
        gold["segments"][0]["speaker_boxes"][0]["bbox"] = [0.1, 0.2, 0.4, 1.1]
        result = evaluate_grounding_sample(gold, outputs)
        self.assertEqual(result["matches"][0]["gt"]["boxes"][0]["box_2d"], [0.1, 0.2, 0.4, 1.1])
        self.assertAlmostEqual(result["metrics"]["average_box_iou"], 2 / 3)

    def test_cli_json_and_jsonl(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            gold = root / "gold.json"
            gold.write_text(json.dumps([gold_sample(), gold_sample("video000002")]))
            for suffix, content in (("json", json.dumps({"results": [prediction()]})),
                                    ("jsonl", json.dumps(prediction()) + "\n")):
                pred = root / ("pred." + suffix)
                pred.write_text(content)
                process = subprocess.run([
                    sys.executable, "-m", "evaluation.scoring.score_grounding", "--gt-file", str(gold),
                    "--input-file", str(pred), "--output-dir", str(root / "out"),
                ], capture_output=True, text=True, check=True)
                self.assertEqual(json.loads(process.stdout)["all_videos"]["tIoU"], 0.5)
                self.assertEqual(len(json.loads((root / "out/eval_results.json").read_text())), 2)


class DatasetValidationChecks(unittest.TestCase):
    def test_clean_release_and_detected_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            mcq = {"question_id": "mcq_1", "video_id": "video000001",
                   "youtube_url": "https://www.youtube.com/watch?v=video000001", "question": "Which sound?",
                   "options": [{"label": label, "text": "Sound " + label} for label in "ABCDEF"],
                   "answer": "A", "category": "Audio", "question_type": "Sound"}
            oeq = {key: value for key, value in mcq.items() if key not in ("options", "answer")}
            oeq.update(question_id="oeq_1", reference_answer="Sound A", reasoning_trace="It is audible.", dense_caption="Sound A is heard.")
            files = {
                "cinemme_grounding.json": [gold_sample()], "cinemme_reasoning_mcq.json": [mcq],
                "cinemme_reasoning_oeq.json": [oeq],
                "video_sources.json": [{"video_id": "video000001", "youtube_url": mcq["youtube_url"],
                                        "tracks": ["grounding", "mcq", "oeq"]}],
                "release_stats.json": {"grounding_samples": 1, "grounding_segments": 1,
                                       "mcq_questions": 1, "oeq_questions": 1, "unique_videos": 1},
            }
            for name, content in files.items():
                (root / name).write_text(json.dumps(content))
            metadata = {"url": "https://github.com/bytedance/vidi/tree/main/CineMME", "creator": "Authors", "distribution": [
                {"name": name, "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest()} for name in files
            ]}
            (root / "croissant.json").write_text(json.dumps(metadata))
            self.assertEqual(validate(root)["findings"], [])
            mcq["options"][0]["text"] = " "
            mcq["options"][2]["text"] = mcq["options"][1]["text"]
            mcq["options"].pop()
            mcq["answer"] = "Z"
            (root / "cinemme_reasoning_mcq.json").write_text(json.dumps([mcq, mcq]))
            gold = gold_sample()
            gold["segments"][0]["end"] = -1
            gold["segments"][0]["speaker_boxes"][0]["bbox"][3] = 1.1
            (root / "cinemme_grounding.json").write_text(json.dumps([gold]))
            report = validate(root)
            for code in ("empty_option", "duplicate_option_text", "option_count", "invalid_answer",
                         "duplicate_id", "duplicate_question", "invalid_segment_interval", "box_out_of_range",
                         "box_outside_segment", "release_count_mismatch", "checksum_mismatch"):
                self.assertIn(code, report["counts_by_code"])


if __name__ == "__main__":
    unittest.main()
