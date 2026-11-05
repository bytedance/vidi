"""Run CineMME open-ended QA inference with a vLLM/OpenAI-compatible API (local vLLM or OpenRouter)."""

from __future__ import annotations

import argparse

from evaluation.prompts import OEQ_PROMPT
from evaluation.utils import add_inference_args, run_requests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_inference_args(parser)
    args = parser.parse_args()
    run_requests(args, "question_id", lambda item: OEQ_PROMPT.substitute(question=item["question"]))


if __name__ == "__main__":
    main()
