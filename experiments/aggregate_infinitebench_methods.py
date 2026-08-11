#!/usr/bin/env python3
"""Aggregate aligned InfiniteBench summaries with exact sparsity totals."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _task_score(summary: dict[str, Any], task: str) -> float:
    result = summary["official_lm_eval_results"][task]
    values = [
        float(value)
        for key, value in result.items()
        if key != "alias" and "stderr" not in key and isinstance(value, (int, float))
    ]
    if len(values) != 1:
        raise ValueError(f"Expected one score for {task}, found {values}")
    return values[0]


def aggregate_method(
    name: str,
    root: Path,
    overrides: dict[tuple[str, str], Path],
) -> dict[str, Any]:
    tasks: list[dict[str, Any]] = []
    totals = {
        "samples": 0,
        "selected_blocks": 0,
        "causal_blocks": 0,
        "selected_token_pairs": 0,
        "causal_token_pairs": 0,
        "input_tokens": 0.0,
        "generated_tokens": 0.0,
        "prefill_latency_sec": 0.0,
        "decode_latency_sec": 0.0,
        "total_latency_sec": 0.0,
        "weighted_score": 0.0,
    }
    has_pair_counts = True
    for summary_path in sorted(root.glob("*/summary.json")):
        task = summary_path.parent.name
        summary_path = overrides.get((name, task), summary_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        runtime = summary["runtime"]
        count = int(runtime["count"])
        alignment = summary.get("input_alignment", {})
        if (
            alignment.get("status") not in {"passed", "reference_created"}
            or alignment.get("count") != count
        ):
            raise ValueError(f"Input alignment failed for {name}/{task}")
        score = _task_score(summary, task)
        metrics_path = summary_path.parent / "online_metrics.jsonl"
        rows = [json.loads(line) for line in metrics_path.open(encoding="utf-8")]
        if len(rows) != count:
            raise ValueError(f"Metric count mismatch for {name}/{task}")
        selected_blocks = sum(int(row.get("selected_blocks") or 0) for row in rows)
        causal_blocks = sum(int(row.get("causal_blocks") or 0) for row in rows)
        pair_rows = all(
            row.get("selected_token_pairs") is not None
            and row.get("causal_token_pairs") is not None
            for row in rows
        )
        selected_pairs = (
            sum(int(row["selected_token_pairs"]) for row in rows) if pair_rows else None
        )
        causal_pairs = (
            sum(int(row["causal_token_pairs"]) for row in rows) if pair_rows else None
        )
        has_pair_counts = has_pair_counts and pair_rows
        tasks.append(
            {
                "task": task,
                "count": count,
                "score": score,
                "avg_prefill_latency_sec": runtime["avg_prefill_latency_sec"],
                "avg_decode_latency_sec": runtime["avg_decode_latency_sec"],
                "avg_total_latency_sec": runtime["avg_total_latency_sec"],
                "global_block_sparsity": (
                    1.0 - selected_blocks / causal_blocks if causal_blocks else None
                ),
                "global_token_sparsity": (
                    1.0 - selected_pairs / causal_pairs
                    if selected_pairs is not None and causal_pairs
                    else None
                ),
            }
        )
        totals["samples"] += count
        totals["selected_blocks"] += selected_blocks
        totals["causal_blocks"] += causal_blocks
        totals["selected_token_pairs"] += selected_pairs or 0
        totals["causal_token_pairs"] += causal_pairs or 0
        totals["input_tokens"] += count * float(runtime["avg_input_tokens"])
        totals["generated_tokens"] += count * float(runtime["avg_generated_tokens"])
        totals["prefill_latency_sec"] += count * float(runtime["avg_prefill_latency_sec"])
        totals["decode_latency_sec"] += count * float(runtime["avg_decode_latency_sec"])
        totals["total_latency_sec"] += count * float(runtime["avg_total_latency_sec"])
        totals["weighted_score"] += count * score

    samples = int(totals["samples"])
    if len(tasks) != 10 or samples != 3493:
        raise ValueError(f"Expected 10 tasks/3493 samples for {name}, got {len(tasks)}/{samples}")
    global_result = {
        "tasks": len(tasks),
        "samples": samples,
        "macro_task_score": sum(item["score"] for item in tasks) / len(tasks),
        "sample_weighted_score": totals["weighted_score"] / samples,
        "avg_input_tokens": totals["input_tokens"] / samples,
        "avg_generated_tokens": totals["generated_tokens"] / samples,
        "avg_prefill_latency_sec": totals["prefill_latency_sec"] / samples,
        "avg_decode_latency_sec": totals["decode_latency_sec"] / samples,
        "avg_total_latency_sec": totals["total_latency_sec"] / samples,
        "samples_per_hour": samples / totals["total_latency_sec"] * 3600.0,
        "selected_blocks": int(totals["selected_blocks"]),
        "causal_blocks": int(totals["causal_blocks"]),
        "global_block_sparsity": (
            1.0 - totals["selected_blocks"] / totals["causal_blocks"]
            if totals["causal_blocks"]
            else None
        ),
        "selected_token_pairs": (
            int(totals["selected_token_pairs"]) if has_pair_counts else None
        ),
        "causal_token_pairs": (
            int(totals["causal_token_pairs"]) if has_pair_counts else None
        ),
        "global_token_sparsity": (
            1.0 - totals["selected_token_pairs"] / totals["causal_token_pairs"]
            if has_pair_counts and totals["causal_token_pairs"]
            else None
        ),
    }
    return {
        "method": name,
        "root": str(root.resolve()),
        "validation": "passed/full",
        "global": global_result,
        "tasks": tasks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", action="append", nargs=2, metavar=("NAME", "ROOT"), required=True)
    parser.add_argument(
        "--task-override",
        action="append",
        nargs=3,
        metavar=("METHOD", "TASK", "TASK_DIR"),
        default=[],
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    overrides = {
        (method, task): Path(task_dir) / "summary.json"
        for method, task, task_dir in args.task_override
    }
    methods = [
        aggregate_method(name, Path(root), overrides) for name, root in args.method
    ]
    payload = {
        "benchmark": "InfiniteBench",
        "sparsity_definition": (
            "1 - sum(final-kernel causally valid selected token pairs) / "
            "sum(dense causal token pairs)"
        ),
        "score_aggregation": {
            "macro_task_score": "unweighted mean over 10 task scores",
            "sample_weighted_score": "mean weighted by each task sample count",
        },
        "methods": methods,
        "task_overrides": {
            f"{method}/{task}": str(path.parent.resolve())
            for (method, task), path in overrides.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    fields = ["method", *methods[0]["global"].keys()]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for method in methods:
            writer.writerow({"method": method["method"], **method["global"]})
    print(json.dumps({m["method"]: m["global"] for m in methods}, indent=2))


if __name__ == "__main__":
    main()
