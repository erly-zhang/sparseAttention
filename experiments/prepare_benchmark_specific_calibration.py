#!/usr/bin/env python3
"""Prepare leakage-free offline calibration records for each benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict


KV_TEMPLATE = (
    "Extract the value corresponding to the specified key in the JSON "
    "object below.\n\n{context}\n\n{input}"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", choices=["infinitebench", "ruler"])
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument(
        "--infinitebench_root",
        default=(
            "/home/ubuntu/work/FlexPrefill/experiments/benchmark/"
            "infinitebench"
        ),
    )
    parser.add_argument("--ruler_source")
    parser.add_argument("--ruler_task", default="niah_single_1")
    parser.add_argument("--ruler_length", type=int, default=12288)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[Dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: list[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def prepare_infinitebench(args: argparse.Namespace) -> Dict[str, Any]:
    source_root = Path(args.infinitebench_root)
    source_data = source_root / "data"
    source_rows = read_jsonl(source_data / "kv_retrieval.jsonl")
    calibration_source = source_rows[: args.num_samples]
    calibration = []
    exclusions = []
    for index, row in enumerate(calibration_source):
        prompt = KV_TEMPLATE.format(
            context=row["context"], input=row["input"]
        )
        sample_id = f"kv_retrieval-{index}"
        calibration.append(
            {
                "_id": sample_id,
                "prompt": prompt,
                "benchmark": "infinitebench",
                "task": "kv_retrieval",
                "source_index": index,
                "prompt_sha256": sha256_text(prompt),
            }
        )
        exclusions.append({"task": "kv_retrieval", "index": index, "id": sample_id})

    output_dir = Path(args.output_dir)
    filtered_data = output_dir / "filtered_data"
    filtered_data.mkdir(parents=True, exist_ok=True)
    for source in source_data.glob("*.jsonl"):
        target = filtered_data / source.name
        if source.name == "kv_retrieval.jsonl":
            write_jsonl(target, source_rows[args.num_samples :])
        elif not target.exists():
            os.link(source, target)

    task_dir = output_dir / "task_configs"
    task_dir.mkdir(parents=True, exist_ok=True)
    original_data_dir = "experiments/benchmark/infinitebench/data"
    for source in source_root.iterdir():
        if source.suffix == ".yaml":
            text = source.read_text(encoding="utf-8").replace(
                f"data_dir: {original_data_dir}",
                f"data_dir: {filtered_data}",
            )
            (task_dir / source.name).write_text(text, encoding="utf-8")
        elif source.suffix == ".py":
            shutil.copy2(source, task_dir / source.name)

    write_jsonl(output_dir / "calibration.jsonl", calibration)
    return {
        "benchmark": "infinitebench",
        "calibration_task": "kv_retrieval",
        "calibration_samples": calibration,
        "excluded_evaluation_samples": exclusions,
        "evaluation_count_before": len(source_rows),
        "evaluation_count_after": len(source_rows) - args.num_samples,
        "filtered_data_dir": str(filtered_data),
        "task_config_dir": str(task_dir),
    }


def prepare_ruler(args: argparse.Namespace) -> Dict[str, Any]:
    if not args.ruler_source:
        raise ValueError("--ruler_source is required for RULER")
    source = Path(args.ruler_source)
    source_rows = read_jsonl(source)
    if len(source_rows) != args.num_samples:
        raise ValueError(
            f"Expected {args.num_samples} RULER rows, found {len(source_rows)}"
        )
    calibration = []
    for row in source_rows:
        index = int(row["index"])
        prompt = str(row["input"])
        calibration.append(
            {
                "_id": (
                    f"ruler:{args.ruler_task}:{args.ruler_length}:{index}"
                ),
                "prompt": prompt,
                "benchmark": "ruler",
                "task": args.ruler_task,
                "sequence_length": args.ruler_length,
                "source_index": index,
                "prompt_sha256": sha256_text(prompt),
            }
        )
    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "calibration.jsonl", calibration)
    return {
        "benchmark": "ruler",
        "calibration_task": args.ruler_task,
        "calibration_length": args.ruler_length,
        "calibration_samples": calibration,
        "evaluation_sample_usage": False,
        "evaluation_lengths": [4096, 8192, 16384, 32768, 65536, 131072],
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = (
        prepare_infinitebench(args)
        if args.benchmark == "infinitebench"
        else prepare_ruler(args)
    )
    (output_dir / "calibration_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
