#!/usr/bin/env python3
"""Validate and aggregate a completed InfiniteBench Compact run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_COUNTS = {
    "longbook_sum_eng": 103,
    "longbook_qa_eng": 351,
    "longbook_choice_eng": 229,
    "longdialogue_qa_eng": 200,
    "longbook_qa_chn": 189,
    "code_debug": 394,
    "math_find": 350,
    "passkey": 590,
    "number_string": 590,
    "kv_retrieval": 497,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--method", default="shareprefill_ae3_compact")
    args = parser.parse_args()

    method_root = args.root / args.method
    count_keys = (
        "selected_token_pairs",
        "causal_token_pairs",
        "compacted_key_tokens",
        "candidate_key_tokens",
        "selected_blocks",
        "causal_blocks",
    )
    latency_keys = (
        "input_tokens",
        "generated_tokens",
        "prefill_latency_sec",
        "decode_latency_sec",
        "total_latency_sec",
    )
    counts = {key: 0 for key in count_keys}
    totals = {key: 0.0 for key in latency_keys}
    task_rows = []
    scores = []

    for task, expected_count in EXPECTED_COUNTS.items():
        task_root = method_root / task
        summary = json.loads(
            (task_root / "summary.json").read_text(encoding="utf-8")
        )
        rows = read_jsonl(task_root / "online_metrics.jsonl")
        alignment = summary["input_alignment"]
        if len(rows) != expected_count:
            raise RuntimeError(
                f"{task}: expected {expected_count} rows, found {len(rows)}"
            )
        if summary["runtime"]["count"] != expected_count:
            raise RuntimeError(f"{task}: summary count mismatch")
        if alignment.get("status") != "passed":
            raise RuntimeError(f"{task}: input alignment did not pass")
        if alignment.get("alignment_scope") != "full":
            raise RuntimeError(f"{task}: input alignment is not full")

        for key in count_keys:
            counts[key] += sum(int(row[key]) for row in rows)
        for key in latency_keys:
            totals[key] += sum(float(row[key]) for row in rows)

        score = task_score(summary, task)
        scores.append((score, expected_count))
        runtime = summary["runtime"]
        task_rows.append(
            {
                "task": task,
                "count": expected_count,
                "score": score,
                "global_token_keep_ratio": runtime[
                    "global_token_keep_ratio"
                ],
                "global_token_sparsity": runtime["global_token_sparsity"],
                "avg_block_keep_ratio": runtime["avg_block_keep_ratio"],
                "within_selected_block_token_keep_ratio": runtime[
                    "within_selected_block_token_keep_ratio"
                ],
                "avg_prefill_latency_sec": runtime[
                    "avg_prefill_latency_sec"
                ],
                "avg_decode_latency_sec": runtime["avg_decode_latency_sec"],
                "avg_total_latency_sec": runtime["avg_total_latency_sec"],
                "input_alignment": alignment,
            }
        )

    total_samples = sum(EXPECTED_COUNTS.values())
    pair_keep = counts["selected_token_pairs"] / counts["causal_token_pairs"]
    block_keep = counts["selected_blocks"] / counts["causal_blocks"]
    within_keep = (
        counts["compacted_key_tokens"] / counts["candidate_key_tokens"]
    )
    aggregate = {
        "benchmark": "InfiniteBench",
        "method": args.method,
        "validation": {
            "status": "passed",
            "tasks": len(EXPECTED_COUNTS),
            "samples": total_samples,
            "all_input_alignments": "passed/full",
        },
        "global": {
            **counts,
            "global_token_keep_ratio": pair_keep,
            "global_token_sparsity": 1.0 - pair_keep,
            "global_block_keep_ratio": block_keep,
            "global_block_sparsity": 1.0 - block_keep,
            "global_within_selected_block_token_keep_ratio": within_keep,
            "global_within_selected_block_token_sparsity": 1.0 - within_keep,
            "macro_task_score": sum(score for score, _ in scores)
            / len(scores),
            "sample_weighted_score": sum(
                score * count for score, count in scores
            )
            / total_samples,
            **{
                f"avg_{key}": value / total_samples
                for key, value in totals.items()
            },
        },
        "tasks": task_rows,
    }
    output = args.output or args.root / "aggregate_summary.json"
    output.write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
