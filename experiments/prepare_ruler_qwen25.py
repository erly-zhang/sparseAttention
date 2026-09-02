#!/usr/bin/env python3
"""Generate the official 13-task RULER matrix for Qwen2.5-7B."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


TASKS = [
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
]
LENGTHS = [4096, 8192, 16384, 32768, 65536, 131072]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--flexprefill_root", default="/home/ubuntu/work/FlexPrefill"
    )
    parser.add_argument(
        "--model", default="/home/ubuntu/work/model/Qwen2.5-7B"
    )
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    parser.add_argument(
        "--lengths", nargs="+", type=int, default=LENGTHS
    )
    parser.add_argument(
        "--output_root",
        default=(
            "/home/ubuntu/work/FlexPrefill/experiments/benchmark/"
            "ruler/data/qwen2_5"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    child_env = dict(os.environ)
    child_env["PATH"] = (
        f"{Path(sys.executable).parent}:{child_env.get('PATH', '')}"
    )
    child_env["PYTHONSAFEPATH"] = "1"
    compat_root = Path(__file__).resolve().parent / "ruler_compat"
    existing_pythonpath = child_env.get("PYTHONPATH", "")
    child_env["PYTHONPATH"] = (
        f"{compat_root}:{existing_pythonpath}"
        if existing_pythonpath
        else str(compat_root)
    )
    ruler_root = Path(args.flexprefill_root) / "experiments/benchmark/ruler"
    prepare = ruler_root / "data/prepare.py"
    output_root = Path(args.output_root)
    for length in args.lengths:
        save_dir = output_root / str(length)
        for task in args.tasks:
            output = save_dir / task / "validation.jsonl"
            if output.is_file() and sum(1 for _ in output.open()) == args.num_samples:
                continue
            command = [
                sys.executable,
                str(prepare),
                "--save_dir",
                str(save_dir),
                "--benchmark",
                "synthetic",
                "--task",
                task,
                "--tokenizer_path",
                args.model,
                "--tokenizer_type",
                "hf",
                "--max_seq_length",
                str(length),
                "--model_template_type",
                "base",
                "--num_samples",
                str(args.num_samples),
            ]
            print(" ".join(command), flush=True)
            subprocess.run(
                command,
                cwd=args.flexprefill_root,
                env=child_env,
                check=True,
            )
            if not output.is_file():
                raise RuntimeError(f"RULER generator did not create {output}")
            actual_rows = sum(1 for _ in output.open(encoding="utf-8"))
            if actual_rows != args.num_samples:
                raise RuntimeError(
                    f"{output} has {actual_rows} rows; "
                    f"expected {args.num_samples}"
                )


if __name__ == "__main__":
    main()
