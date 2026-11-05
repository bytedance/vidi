"""Run with: python -m unittest discover -s tests -v (no API access, GPUs, or videos needed)."""

import argparse
import json
from pathlib import Path
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

from evaluation import utils
from evaluation.scoring.score_mcq import extract_mcq_answer, score_records


class McqExtraction(unittest.TestCase):
    def test_canonical_denominator_and_ids(self):
        gold = [{"question_id": "q1", "answer": "A"}, {"question_id": "q2", "answer": "B"}]
        rows = [{"question_id": "q1", "answer": "F", "response": "<Answer> A"}]
        scores, _ = score_records(rows, gold)
        self.assertEqual(scores["accuracy"], 1)
        self.assertEqual(scores["accuracy_all"], 0.5)
        self.assertEqual(scores["total"], 2)
        for bad in (rows * 2, [{"question_id": "unknown", "response": "A"}], [{"response": "A"}]):
            with self.assertRaises(ValueError):
                score_records(bad, gold)

    def test_strategy_order(self):
        self.assertEqual(extract_mcq_answer("<Step 1> A. is wrong\n<Answer> C"), "C")
        self.assertEqual(extract_mcq_answer("<Answer> B\nOn reflection, the answer is D."), "B")
        self.assertEqual(extract_mcq_answer("Reasoning...\nD"), "D")
        self.assertEqual(extract_mcq_answer("So the final answer is E."), "E")
        self.assertIsNone(extract_mcq_answer(""))


class GroundingParsing(unittest.TestCase):
    def test_single_wrapped_object(self):
        from evaluation.metrics import parse_grounding_response

        segment = {"text": "hi", "start": 1.0, "end": 2.0, "boxes": []}
        self.assertEqual(parse_grounding_response(json.dumps([{"outputs": [segment]}])), [segment])
        self.assertEqual(parse_grounding_response(json.dumps([segment])), [segment])


class JudgeValidation(unittest.TestCase):
    def test_completed_judgments_are_saved_while_an_earlier_request_waits(self):
        from evaluation.scoring import judge_oeq

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            dataset, predictions = tmp / "gold.json", tmp / "pred.jsonl"
            dataset.write_text(json.dumps([
                {"question_id": key, "video_id": key, "dense_caption": "A scene"}
                for key in ("slow", "fast")]))
            predictions.write_text("".join(json.dumps({"question_id": key, "response": "answer"}) + "\n"
                                           for key in ("slow", "fast")))
            output, summary = tmp / "judged.jsonl", tmp / "summary.json"
            args = argparse.Namespace(dataset=dataset, predictions=predictions, output=output,
                                      summary_output=summary, workers=2, api_key="EMPTY",
                                      api_base="http://unused", header=[])
            release, fast_done = threading.Event(), threading.Event()
            errors = []

            def fake_judge(client, args, item, response):
                if item["question_id"] == "slow":
                    release.wait(5)
                else:
                    fast_done.set()
                return {"score": 4}

            def run():
                try:
                    judge_oeq.main()
                except BaseException as exc:
                    errors.append(exc)

            with mock.patch.object(judge_oeq, "parse_args", return_value=args), \
                 mock.patch.object(judge_oeq, "load_dotenv"), \
                 mock.patch.object(judge_oeq, "judge", side_effect=fake_judge), \
                 mock.patch.dict("sys.modules", {"openai": types.SimpleNamespace(OpenAI=lambda **_: object())}):
                worker = threading.Thread(target=run)
                worker.start()
                try:
                    self.assertTrue(fast_done.wait(2))
                    deadline = time.monotonic() + 2
                    while time.monotonic() < deadline and not (output.exists() and output.stat().st_size):
                        time.sleep(0.01)
                    self.assertTrue(output.exists() and output.stat().st_size,
                                    "Completed judgments must be flushed before slow requests finish")
                    self.assertEqual(utils.load_jsonl(output)[0]["question_id"], "fast")
                finally:
                    release.set()
                    worker.join(5)
            self.assertFalse(worker.is_alive())
            self.assertFalse(errors)
            self.assertEqual(len(utils.load_jsonl(output)), 2)
            self.assertEqual(json.loads(summary.read_text())["score_100"], 40)

    def test_out_of_range_judge_output_is_not_counted(self):
        from evaluation.scoring.judge_oeq import judge

        create = mock.Mock(return_value=types.SimpleNamespace(choices=[
            types.SimpleNamespace(message=types.SimpleNamespace(content='{"score": 12}'))]))
        client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
        args = argparse.Namespace(retries=1, model="judge", temperature=0)
        self.assertTrue(judge(client, args, {"dense_caption": "A scene"}, "answer")["failed"])


class VideoIoU(unittest.TestCase):
    def test_continuous_tube(self):
        from evaluation.metrics import video_iou

        box = [0.1, 0.1, 0.5, 0.5]
        gt = {1.0: [box], 2.0: [box]}
        self.assertAlmostEqual(video_iou(gt, gt, duration=10), 1.0)
        # Same boxes, one extra predicted second: union 3 s, overlap 2 s.
        self.assertAlmostEqual(video_iou({**gt, 3.0: [box]}, gt, duration=10), 2 / 3)
        # Half-width box on the shared seconds.
        self.assertAlmostEqual(video_iou({1.0: [[0.1, 0.1, 0.3, 0.5]], 2.0: [box]}, gt, duration=10), 0.75)
        self.assertEqual(video_iou({}, gt, duration=10), 0.0)
        self.assertIsNone(video_iou(gt, {}, duration=10))


class RunRequests(unittest.TestCase):
    def test_fixed_frames_and_remaining_context_preserve_audio_template_and_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            dataset = tmp / "grounding.json"
            dataset.write_text(json.dumps([{"sample_id": "s0", "video_id": "v0"}]))
            create = mock.Mock(return_value=types.SimpleNamespace(
                choices=[types.SimpleNamespace(
                    message=types.SimpleNamespace(content='{"outputs": []}'), finish_reason="stop")],
                model="m", model_extra={}, usage=None))
            client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
            args = argparse.Namespace(
                dataset=dataset, output=tmp / "pred.jsonl", video_root=tmp, video_template=None,
                model="m", limit=None, use_audio=True, transport="inline", fps=1.0,
                num_frames=128, chat_template_kwargs={"enable_thinking": False, "use_tts_template": True},
                temperature=0.7, top_p=0.01, max_tokens=0, workers=1, media_cache=None)
            schema = {"structured_outputs": {"json": {"type": "object"}}}
            with mock.patch.object(utils, "make_client", return_value=client), \
                 mock.patch.object(utils, "resolve_video_path", return_value=tmp / "v.mp4"), \
                 mock.patch.object(utils, "media_content", return_value=[]):
                utils.run_requests(args, "sample_id", lambda _: "grounding", schema)
            request = create.call_args.kwargs
            self.assertIsNone(request["max_tokens"])
            self.assertEqual(request["extra_body"]["media_io_kwargs"]["video"],
                             {"num_frames": 128, "fps": -1, "video_backend": "opencv"})
            self.assertEqual(request["extra_body"]["chat_template_kwargs"], args.chat_template_kwargs)
            self.assertEqual(request["extra_body"]["structured_outputs"], schema["structured_outputs"])
            self.assertNotIn("chat_template_kwargs", schema)

    def test_vllm_sampling_is_request_level_and_preserves_schema(self):
        for transport in ("file", "inline", "base64"):
            with self.subTest(transport=transport), tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                dataset = tmp / "grounding.json"
                dataset.write_text(json.dumps([{"sample_id": "s0", "video_id": "v0"}]))
                create = mock.Mock(return_value=types.SimpleNamespace(
                    choices=[types.SimpleNamespace(
                        message=types.SimpleNamespace(content='{"outputs": []}'), finish_reason="stop")],
                    model="m", model_extra={}, usage=None,
                ))
                client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
                args = argparse.Namespace(
                    dataset=dataset, output=tmp / "pred.jsonl", video_root=tmp, video_template=None,
                    model="m", limit=None, use_audio=True, transport=transport, fps=2.0,
                    temperature=0.7, top_p=0.01, max_tokens=16, workers=1, media_cache=None,
                )
                schema = {"structured_outputs": {"json": {"type": "object"}}}
                with mock.patch.object(utils, "make_client", return_value=client), \
                     mock.patch.object(utils, "resolve_video_path", return_value=tmp / "v.mp4"), \
                     mock.patch.object(utils, "media_content", return_value=[]):
                    utils.run_requests(args, "sample_id", lambda _: "grounding", schema)
                body = create.call_args.kwargs["extra_body"]
                self.assertEqual(body["structured_outputs"], schema["structured_outputs"])
                if transport == "base64":
                    self.assertNotIn("media_io_kwargs", body)
                else:
                    self.assertEqual(body["media_io_kwargs"], {"video": {"fps": 2.0, "num_frames": -1}})
                self.assertNotIn("media_io_kwargs", schema)

    def test_resume_keeps_answers_and_retries_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            dataset = tmp / "mcq.json"
            dataset.write_text(json.dumps([{"question_id": f"q{i}", "video_id": f"v{i}"} for i in range(3)]))
            output = tmp / "pred.jsonl"
            output.write_text(
                json.dumps({"question_id": "q0", "video_id": "v0", "response": "<Answer> A"}) + "\n"
                + json.dumps({"question_id": "q1", "video_id": "v1", "response": "", "error": "timeout"}) + "\n"
            )
            asked = []

            def create(**kwargs):
                asked.append(kwargs["messages"][0]["content"][-1]["text"])
                message = types.SimpleNamespace(content="<Answer> B")
                return types.SimpleNamespace(
                    choices=[types.SimpleNamespace(message=message, finish_reason="stop")],
                    model="m", model_extra={"provider": "p"}, usage=None,
                )

            client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
            args = argparse.Namespace(
                dataset=dataset, output=output, video_root=tmp, video_template=None, model="m", limit=None,
                use_audio=True, transport="file", fps=1.0, temperature=0.7, top_p=0.01, max_tokens=16, workers=2, media_cache=None,
            )
            with mock.patch.object(utils, "make_client", return_value=client), \
                 mock.patch.object(utils, "resolve_video_path", return_value=tmp / "v.mp4"), \
                 mock.patch.object(utils, "media_content", return_value=[]):
                utils.run_requests(args, "question_id", lambda item: item["question_id"])

            rows = {row["question_id"]: row for row in utils.load_jsonl(output)}
            self.assertEqual(sorted(asked), ["q1", "q2"])
            self.assertEqual(len(utils.load_jsonl(output)), 3)
            self.assertEqual(rows["q0"]["response"], "<Answer> A")
            self.assertEqual(rows["q1"]["provider"], "p")


if __name__ == "__main__":
    unittest.main()
