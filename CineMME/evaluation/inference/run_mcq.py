"""Run CineMME MCQ inference with a vLLM/OpenAI-compatible API (local vLLM or OpenRouter)."""

from __future__ import annotations

import argparse

from evaluation.prompts import MCQ_PROMPT
from evaluation.utils import add_inference_args, run_requests


def format_options(options: list[dict] | dict) -> str:
    if isinstance(options, dict):
        return "\n".join(f"{label}. {text}" for label, text in options.items())
    return "\n".join(f"{opt['label']}. {opt['text']}" for opt in options)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_inference_args(parser)
    args = parser.parse_args()
    run_requests(
        args,
        "question_id",
        lambda item: MCQ_PROMPT.substitute(question=item["question"], options=format_options(item["options"])),
    )


if __name__ == "__main__":
    main()
