#!/usr/bin/env python3
"""Locate exact target-token occurrences in logged lm-eval prompts."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--model", default="/home/ubuntu/work/model/Qwen2.5-7B")
    return parser.parse_args()


def find_spans(values: list[int], pattern: list[int]) -> list[list[int]]:
    return [
        [start, start + len(pattern)]
        for start in range(len(values) - len(pattern) + 1)
        if values[start : start + len(pattern)] == pattern
    ]


def main() -> None:
    args = parse_args()
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    samples = payload.get("samples", {}).get(args.task, [])
    if not samples:
        raise RuntimeError(f"No logged samples for task {args.task}")
    for index, sample in enumerate(samples):
        target = str(sample["target"])
        argument = ast.literal_eval(sample["arguments"][0])
        prompt = argument[0] if isinstance(argument, tuple) else argument
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        target_ids = tokenizer.encode(target, add_special_tokens=False)
        print(
            json.dumps(
                {
                    "sample": index,
                    "target": target,
                    "input_tokens": len(input_ids),
                    "target_token_ids": target_ids,
                    "target_token_count": len(target_ids),
                    "spans": find_spans(input_ids, target_ids),
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
