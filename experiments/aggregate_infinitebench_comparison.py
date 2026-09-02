#!/usr/bin/env python3
"""Aggregate the aligned InfiniteBench method comparison."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.aggregate_infinitebench_compact import EXPECTED_COUNTS


WORK_ROOT = Path("/home/ubuntu/work")
OUTPUT_ROOT = WORK_ROOT / "experiments/outputs"
COMPACT_ROOT = (
    OUTPUT_ROOT
    / "infinitebench_shareprefill_ae3_compact_global_sparsity"
    / "shareprefill_ae3_compact"
)
HISA_ROOT = (
    OUTPUT_ROOT
    / "infinitebench_shareprefill_ae3_hisa_style"
    / "shareprefill_ae3_hisa"
)
UNIFIED_ROOT = OUTPUT_ROOT / "infinitebench_unified_math10_comparison"
FULL_ROOT = (
    OUTPUT_ROOT
    / "infinitebench_benchmark_specific_comparison"
    / "shareprefill_ae3_full"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def task_root(method: str, task: str) -> Path:
    if method == "shareprefill_ae3_compact":
        return COMPACT_ROOT / task
    if method == "shareprefill_ae3_hisa":
        return HISA_ROOT / task
    if method == "shareprefill_ae3_full":
        if task == "math_find":
            return UNIFIED_ROOT / method / task
        return FULL_ROOT / task
    return UNIFIED_ROOT / method / task


def task_score(summary: dict[str, Any], task: str) -> float:
    metrics = summary["official_lm_eval_results"][task]
    values = [
        float(value)
        for key, value in metrics.items()
        if key != "alias" and "stderr" not in key
    ]
    if len(values) != 1:
        raise RuntimeError(f"Expected one score for {task}, found {values}")
    return values[0]


def row_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["benchmark"],
        row["task"],
        row["input_tokens"],
        row["input_ids_sha256"],
    )


def main() -> None:
    methods = (
        "shareprefill_ae3_compact",
        "shareprefill_ae3_hisa",
        "shareprefill_ae3_full",
        "flexprefill",
        "minference",
    )
    aggregate: dict[str, Any] = {}
    reference_identities: dict[str, list[tuple[Any, ...]]] = {}

    for method in methods:
        sample_count = 0
        latency = {
            "prefill_latency_sec": 0.0,
            "decode_latency_sec": 0.0,
            "total_latency_sec": 0.0,
            "input_tokens": 0.0,
            "generated_tokens": 0.0,
        }
        scores = []
        task_results = []
        peak_memory = 0
        selected_blocks = 0
        causal_blocks = 0
        selected_pairs = 0
        causal_pairs = 0

        for task, expected_count in EXPECTED_COUNTS.items():
            root = task_root(method, task)
            summary = json.loads(
                (root / "summary.json").read_text(encoding="utf-8")
            )
            rows = read_jsonl(root / "online_metrics.jsonl")
            if len(rows) != expected_count:
                raise RuntimeError(
                    f"{method}/{task}: expected {expected_count}, "
                    f"found {len(rows)}"
                )
            identities = [row_identity(row) for row in rows]
            if method == methods[0]:
                reference_identities[task] = identities
            elif identities != reference_identities[task]:
                raise RuntimeError(f"{method}/{task}: input mismatch")

            score = task_score(summary, task)
            scores.append((score, expected_count))
            sample_count += expected_count
            for key in latency:
                latency[key] += sum(float(row[key]) for row in rows)
            peak_memory = max(
                peak_memory,
                max(int(row["peak_memory_bytes"]) for row in rows),
            )
            if "selected_blocks" in rows[0]:
                selected_blocks += sum(
                    int(row["selected_blocks"]) for row in rows
                )
                causal_blocks += sum(int(row["causal_blocks"]) for row in rows)
            if "selected_token_pairs" in rows[0]:
                selected_pairs += sum(
                    int(row["selected_token_pairs"]) for row in rows
                )
                causal_pairs += sum(
                    int(row["causal_token_pairs"]) for row in rows
                )
            task_results.append(
                {
                    "task": task,
                    "count": expected_count,
                    "score": score,
                    "avg_prefill_latency_sec": sum(
                        float(row["prefill_latency_sec"]) for row in rows
                    )
                    / expected_count,
                    "avg_total_latency_sec": sum(
                        float(row["total_latency_sec"]) for row in rows
                    )
                    / expected_count,
                }
            )

        result = {
            "count": sample_count,
            "input_alignment": "passed/full across all methods",
            "macro_task_score": sum(score for score, _ in scores)
            / len(scores),
            "sample_weighted_score": sum(
                score * count for score, count in scores
            )
            / sample_count,
            **{
                f"avg_{key}": value / sample_count
                for key, value in latency.items()
            },
            "max_peak_memory_bytes": peak_memory,
            "tasks": task_results,
        }
        if causal_blocks:
            result["global_block_keep_ratio"] = (
                selected_blocks / causal_blocks
            )
        if causal_pairs:
            pair_keep = selected_pairs / causal_pairs
            result["global_token_keep_ratio"] = pair_keep
            result["global_token_sparsity"] = 1.0 - pair_keep
        aggregate[method] = result

    compact_prefill = aggregate["shareprefill_ae3_compact"][
        "avg_prefill_latency_sec"
    ]
    compact_total = aggregate["shareprefill_ae3_compact"][
        "avg_total_latency_sec"
    ]
    aggregate["shareprefill_ae3_compact"]["relative_speed"] = {
        method: {
            "prefill_speedup": values["avg_prefill_latency_sec"]
            / compact_prefill,
            "total_speedup": values["avg_total_latency_sec"]
            / compact_total,
        }
        for method, values in aggregate.items()
        if method != "shareprefill_ae3_compact"
    }
    output = (
        OUTPUT_ROOT
        / "infinitebench_shareprefill_ae3_hisa_style"
        / "comparison_summary.json"
    )
    output.write_text(
        json.dumps(
            {
                "validation": {
                    "status": "passed",
                    "tasks": 10,
                    "samples_per_method": sum(EXPECTED_COUNTS.values()),
                    "input_identity": "exact across all four methods",
                },
                "methods": aggregate,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
