"""Run CineMME grounding inference with a vLLM/OpenAI-compatible API (local vLLM or OpenRouter)."""

from __future__ import annotations

import argparse

from evaluation.prompts import GROUNDING_PROMPT, GROUNDING_PROMPT_API, JSON_SCHEMA_FOR_GROUNDING
from evaluation.utils import add_inference_args, is_openrouter, run_requests


def structured_extra_body(dialect: str) -> dict | None:
    if dialect == "none":
        return None
    if dialect == "structured_outputs":  # vLLM >= 0.10
        return {"structured_outputs": {"json": JSON_SCHEMA_FOR_GROUNDING}}
    if dialect == "guided_json":  # older vLLM
        return {"guided_json": JSON_SCHEMA_FOR_GROUNDING}
    if dialect == "json_schema":  # OpenAI-style hosted APIs, e.g. OpenRouter
        return {"response_format": {"type": "json_schema", "json_schema": {"name": "grounding", "schema": JSON_SCHEMA_FOR_GROUNDING}}}
    if dialect == "response_format":
        return {"response_format": {"type": "json_object"}}
    raise ValueError(f"Unsupported structured-output dialect: {dialect}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_inference_args(parser)
    parser.add_argument(
        "--structured-output",
        choices=["structured_outputs", "guided_json", "json_schema", "response_format", "none"],
        help="Default: response_format (JSON object) for openrouter.ai, else structured_outputs.",
    )
    parser.add_argument(
        "--prompt-variant",
        choices=["vllm", "api"],
        help="vllm: schema-constrained prompt; api: prompt with an inline format example for hosted APIs "
        "without reliable schema decoding. Default: api for openrouter.ai, else vllm.",
    )
    args = parser.parse_args()
    hosted = is_openrouter(args.api_base)
    dialect = args.structured_output or ("response_format" if hosted else "structured_outputs")
    prompt = GROUNDING_PROMPT_API if (args.prompt_variant or ("api" if hosted else "vllm")) == "api" else GROUNDING_PROMPT
    run_requests(args, "sample_id", lambda item: prompt, structured_extra_body(dialect))


if __name__ == "__main__":
    main()
