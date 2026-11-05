"""Judge CineMME open-ended answers with an OpenAI-compatible LLM (paper protocol).

The paper judge is Qwen/Qwen2.5-72B-Instruct served by vLLM. Empty responses and
items whose judge output cannot be parsed after all retries are excluded from the mean,
matching the scores reported in the paper.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from evaluation.utils import add_header_arg, load_dotenv, load_json, load_jsonl, parse_headers, save_json, tqdm


SYSTEM_PROMPT = """
You are an impartial and meticulous AI Judge specializing in evaluating Video Understanding and Reasoning models.

Your task is to evaluate a [Candidate Response] against a provided [Video Context] (Dense Caption) and a [Gold Standard] (Reference Answer & Reasoning Trace).

### Evaluation Principles:
1. **Factuality based on Context**: The [Dense Caption] is the ABSOLUTE GROUND TRUTH. If the candidate mentions visual/audio details that contradict the caption, it is a hallucination error.
2. **Reasoning Alignment**: The candidate has provided a step-by-step reasoning process (<Step 1>, <Step 2>...). You must verify if their logic aligns with the [Correct Reasoning Trace]. A correct answer derived from wrong reasoning (lucky guess) should receive a penalty.
3. **Completeness**: Does the candidate cover all key elements mentioned in the [Correct Answer]?
4. **Consistency**: The candidate's final <Answer> must be consistent with their own reasoning steps.

### Scoring Scale (0-10):
- **10**: Perfect. Matches reference reasoning and answer completely.
- **8-9**: Correct answer, reasoning is good but misses minor nuance.
- **6-7**: Correct answer, but reasoning is vague or partially flawed.
- **4-5**: Answer is partially correct (captures part of the truth), or reasoning is weak or None.
- **2-3**: Incorrect answer, or significant hallucination in reasoning.
- **0-1**: Completely irrelevant or factually wrong.
"""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "checklist": {
            "type": "object",
            "properties": {
                "steps_logic_aligned": {
                    "type": "boolean",
                    "description": "True if the candidate's reasoning steps logically match the reference trace.",
                },
                "key_entities_identified": {
                    "type": "boolean",
                    "description": "True if the specific object/sound (e.g., turbine blades) was identified.",
                },
                "no_hallucination": {
                    "type": "boolean",
                    "description": "True if the response contains NO contradictions to the dense caption.",
                },
            },
            "required": ["steps_logic_aligned", "key_entities_identified", "no_hallucination"],
        },
        "critique": {
            "type": "string",
            "description": "A concise explanation of the pros and cons of the response, referencing specific steps.",
        },
        "score": {
            "type": "integer",
            "minimum": 0,
            "maximum": 10,
            "description": "The final score from 0 to 10 based on the scoring scale.",
        },
        "error_category": {
            "type": "string",
            "enum": ["None", "Reasoning Error", "Hallucination", "Missed Detail", "Wrong Answer"],
            "description": "The primary type of error, if any.",
        },
    },
    "required": ["checklist", "critique", "score", "error_category"],
}


def build_prompt(item: dict[str, Any], response: str) -> str:
    # Kept byte-for-byte with the judge used for the paper, including indentation.
    return f"""
    ## 1. Ground Truth Context
    "{item.get('dense_caption', '')}"

    ## 2. The Question
    "{item.get('question', '')}"

    ## 3. Gold Standard
    Reference Answer: "{item.get('reference_answer', '')}"
    Reference Trace: "{item.get('reasoning_trace', '')}"

    ## 4. Candidate Response
    "{response}"
    """


def judge(client: Any, args: argparse.Namespace, item: dict[str, Any], response: str) -> dict[str, Any]:
    error = ""
    for _ in range(args.retries):
        try:
            completion = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_prompt(item, response)},
                ],
                temperature=args.temperature,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                    "structured_outputs": {"json": JUDGE_SCHEMA},
                },
            )
            result = json.loads(completion.choices[0].message.content)
            score = result.get("score")
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 10:
                raise ValueError("Judge score must be a finite number in [0, 10]")
            return result
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    return {"error": error, "score": 0, "failed": True}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path, help="OEQ dataset including the paper judge's dense_caption context.")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen2.5-72B-Instruct")
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--retries", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    add_header_arg(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = load_json(args.dataset)
    gold = {item["question_id"]: item for item in items}
    if not gold or len(gold) != len(items):
        raise ValueError("OEQ dataset must contain unique question IDs and must not be empty")
    if any(not isinstance(item.get("dense_caption"), str) or not item["dense_caption"].strip() for item in items):
        raise ValueError("The paper judge requires dense_caption for every OEQ; use release 1.1.0 or later")
    preds = load_jsonl(args.predictions) if args.predictions.suffix == ".jsonl" else load_json(args.predictions)
    seen = set()
    for pred in preds:
        key = pred.get("question_id")
        if key not in gold or key in seen:
            raise ValueError(f"Unknown, missing, or duplicate question_id: {key!r}")
        seen.add(key)
    preds = [p for p in preds if isinstance(p.get("response"), str) and p["response"].strip()]
    from openai import OpenAI

    load_dotenv()
    client = OpenAI(api_key=args.api_key or os.environ.get("VLLM_API_KEY", "EMPTY"), base_url=args.api_base,
                    default_headers=parse_headers(args.header))

    def run(pred: dict[str, Any]) -> dict[str, Any]:
        item = gold[pred["question_id"]]
        return {
            "question_id": item["question_id"],
            "video_id": item["video_id"],
            "category": item.get("category"),
            "judge_result": judge(client, args, item, pred["response"]),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool, args.output.open("w", encoding="utf-8") as fout:
        futures = [pool.submit(run, pred) for pred in preds]
        for future in tqdm(as_completed(futures), total=len(futures), dynamic_ncols=True):
            row = future.result()
            rows.append(row)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()

    valid = [row["judge_result"]["score"] for row in rows if not row["judge_result"].get("failed")]
    mean = sum(valid) / len(valid) if valid else 0.0
    summary = {
        "judged": len(rows),
        "valid": len(valid),
        "failed": len(rows) - len(valid),
        "average_score": mean,
        "score_100": mean * 10,
    }
    if args.summary_output:
        save_json(summary, args.summary_output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
