"""Extract and score multiple-choice answers."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from evaluation.utils import load_json, load_jsonl


def extract_mcq_answer(text: str) -> str | None:
    """Extract the option letter with the same strategy order used for the paper results."""
    if not isinstance(text, str) or not text.strip():
        return None
    first = lambda pattern: re.search(pattern, text, re.IGNORECASE)
    last = lambda pattern: re.findall(pattern, text, re.IGNORECASE)
    if match := first(r"<Answer>\s*(?:Option)?\s*(?:letter)?\s*[:\s]*([A-F])"):
        return match.group(1).upper()
    last_line = text.strip().splitlines()[-1].strip()
    if re.fullmatch(r"[A-F]", last_line, re.IGNORECASE):
        return last_line.upper()
    if matches := last(r"\b(?:Answer|The final answer is|is|therefore|conclusion)\b[:\s]+(?:Option\s+)?([A-F])(?:\.|\s|,|$)"):
        return matches[-1].upper()
    for pattern in (r"\\boxed\{([A-F])\}", r"<\|begin_of_box\|>\s*([A-F])\s*<\|end_of_box\|>"):
        if match := first(pattern):
            return match.group(1).upper()
    for pattern in (r"\b([A-F])[).:]", r"\b([A-F])\b"):
        if matches := last(pattern):
            return matches[-1].upper()
    return None


def load_predictions(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return load_jsonl(path)
    payload = load_json(path)
    return payload.get("results", payload) if isinstance(payload, dict) else payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--response-field", default="response")
    return parser.parse_args()


def score_records(records, dataset=None, response_field="response"):
    answer_key = {item["question_id"]: item["answer"] for item in dataset} if dataset is not None else {}
    if dataset is not None and (not answer_key or len(answer_key) != len(dataset)):
        raise ValueError("Dataset must contain unique question IDs and must not be empty")
    seen = set()
    total = correct = answered = 0
    annotated = []
    for record in records:
        item = dict(record)
        if dataset is not None:
            key = item.get("question_id")
            if key not in answer_key or key in seen:
                raise ValueError(f"Unknown, missing, or duplicate question_id: {key!r}")
            seen.add(key)
        pred_answer = extract_mcq_answer(item.get(response_field, ""))
        item["pred_answer"] = pred_answer or ""
        gold = answer_key[item["question_id"]] if dataset is not None else item.get("answer")
        if gold:
            total += 1
            answered += pred_answer is not None
            item["correct"] = float(pred_answer == gold)
            correct += pred_answer == gold
        annotated.append(item)
    if dataset is not None:
        total = len(dataset)
    # Paper metric: accuracy over questions with an extractable answer; accuracy_all counts misses as wrong.
    summary = {
        "total": total,
        "answered": answered,
        "correct": correct,
        "accuracy": correct / answered if answered else 0.0,
        "accuracy_all": correct / total if total else 0.0,
    }
    return summary, annotated


def main() -> None:
    args = parse_args()
    summary, annotated = score_records(
        load_predictions(args.predictions), load_json(args.dataset) if args.dataset else None, args.response_field)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(annotated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
