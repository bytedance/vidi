"""Refine CineMME MCQ distractors with an OpenAI-compatible LLM."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **_: object):
        return iterable

from data_process.utils import extract_json_from_text, load_json, save_json
from data_process.prompts import DISTRACTOR_REFINEMENT_PROMPT
from data_process.schemas import DISTRACTOR_REFINEMENT_SCHEMA


def to_refinement_item(item: dict[str, Any], idx: int) -> dict[str, Any]:
    options = item.get("options", {})
    if isinstance(options, list):
        option_map = {opt["label"]: opt["text"] for opt in options}
    else:
        option_map = options
    answer = item.get("answer", "")
    distractors = [f"{label}. {text}" for label, text in option_map.items() if label != answer]
    return {
        "id": item.get("question_id", item.get("id", str(idx))),
        "question": item.get("question", ""),
        "correct_answer": f"{answer}. {option_map.get(answer, '')}",
        "current_distractors": distractors,
        "evidence_explanation": item.get("explanation", item.get("rationale", "")),
        "model_predictions": item.get("model_predictions", {}),
    }


def structured_extra_body(dialect: str) -> dict[str, Any] | None:
    if dialect == "none":
        return None
    if dialect == "structured_outputs":
        return {"structured_outputs": {"json": DISTRACTOR_REFINEMENT_SCHEMA}}
    if dialect == "guided_json":
        return {"guided_json": DISTRACTOR_REFINEMENT_SCHEMA}
    if dialect == "response_format":
        return {"response_format": {"type": "json_object"}}
    raise ValueError(f"Unsupported structured output dialect: {dialect}")


def refine_batch(
    client: Any,
    *,
    model: str,
    items: list[dict[str, Any]],
    structured_output: str,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    prompt = DISTRACTOR_REFINEMENT_PROMPT.substitute(items=json.dumps(items, ensure_ascii=False, indent=2))
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    extra_body = structured_extra_body(structured_output)
    if extra_body:
        kwargs["extra_body"] = extra_body
    response = client.chat.completions.create(**kwargs)
    return extract_json_from_text(response.choices[0].message.content or "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="MCQ JSON file.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--structured-output", choices=["structured_outputs", "guided_json", "response_format", "none"], default="structured_outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_json(args.input)
    if isinstance(records, dict) and "questions_and_answers" in records:
        records = records["questions_and_answers"]
    from openai import OpenAI

    client = OpenAI(api_key=args.api_key, base_url=args.api_base)
    outputs = []
    for start in tqdm(range(0, len(records), args.batch_size), dynamic_ncols=True):
        batch = [to_refinement_item(item, start + offset) for offset, item in enumerate(records[start : start + args.batch_size])]
        outputs.append(
            refine_batch(
                client,
                model=args.model,
                items=batch,
                structured_output=args.structured_output,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
        )
    save_json({"batches": outputs}, args.output)


if __name__ == "__main__":
    main()
