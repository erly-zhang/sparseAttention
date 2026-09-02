#!/usr/bin/env python3
"""Validate and summarize the small InfiniteBench hybrid-layer scan."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


START_LAYERS = (0, 6, 10, 14, 15, 16, 17, 18, 19, 20, 21, 22, 28)
TASKS = ("passkey", "number_string")
METHOD = "shareprefill_ae3_token_block_auto_hybrid"
NUM_LAYERS = 28


def task_score(summary: dict, task: str) -> float:
    result = summary["official_lm_eval_results"][task]
    values = [
        value
        for key, value in result.items()
        if key.startswith("get_score_one_") and "stderr" not in key
    ]
    if len(values) != 1:
        raise ValueError(f"Expected one score for {task}, got {values}")
    return float(values[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    rows = []
    hashes_by_task: dict[str, list[str]] = {}
    for start_layer in START_LAYERS:
        task_rows = []
        for task in TASKS:
            task_dir = args.root / f"start_{start_layer}" / METHOD / task
            summary = json.loads((task_dir / "summary.json").read_text())
            metrics = [
                json.loads(line)
                for line in (task_dir / "online_metrics.jsonl").read_text().splitlines()
            ]
            if len(metrics) != 3 or summary["runtime"]["count"] != 3:
                raise ValueError(f"Incomplete output for start={start_layer} {task}")
            if summary["input_alignment"]["status"] != "passed":
                raise ValueError(f"Alignment failed for start={start_layer} {task}")
            configured_start = summary["method_metadata"]["online_config"][
                "target_top_p_start_layer_zero_based"
            ]
            if configured_start != start_layer:
                raise ValueError(
                    f"Metadata start mismatch: {configured_start} != {start_layer}"
                )
            hashes = [metric["input_ids_sha256"] for metric in metrics]
            reference_hashes = hashes_by_task.setdefault(task, hashes)
            if hashes != reference_hashes:
                raise ValueError(f"Input hash mismatch for start={start_layer} {task}")
            for metric in metrics:
                layer_stats = metric["per_layer_selector_stats"]
                for layer in range(NUM_LAYERS):
                    expected = (
                        "fixed_top_k" if layer < start_layer else "probability_top_p"
                    )
                    modes = layer_stats[str(layer)]["target_selection_mode_rows"]
                    if set(modes) != {expected}:
                        raise ValueError(
                            f"Mode mismatch start={start_layer} task={task} "
                            f"layer={layer}: {modes}"
                        )
            runtime = summary["runtime"]
            task_rows.append(
                {
                    "task": task,
                    "score": task_score(summary, task),
                    "prefill": float(runtime["avg_prefill_latency_sec"]),
                    "total": float(runtime["avg_total_latency_sec"]),
                    "selected_pairs": int(runtime["selected_token_pairs"]),
                    "causal_pairs": int(runtime["causal_token_pairs"]),
                }
            )

        selected_pairs = sum(row["selected_pairs"] for row in task_rows)
        causal_pairs = sum(row["causal_pairs"] for row in task_rows)
        rows.append(
            {
                "start_layer_zero_based": start_layer,
                "first_top_p_layer_one_based": (
                    start_layer + 1 if start_layer < NUM_LAYERS else "none"
                ),
                "fixed_layer_count": start_layer,
                "top_p_layer_count": NUM_LAYERS - start_layer,
                "passkey_score": task_rows[0]["score"],
                "number_string_score": task_rows[1]["score"],
                "macro_score": sum(row["score"] for row in task_rows) / len(task_rows),
                "avg_prefill_latency_sec": sum(row["prefill"] for row in task_rows)
                / len(task_rows),
                "avg_total_latency_sec": sum(row["total"] for row in task_rows)
                / len(task_rows),
                "global_token_sparsity": 1.0 - selected_pairs / causal_pairs,
                "input_alignment": "passed",
                "layer_modes": "passed",
            }
        )

    output_json = args.root / "layer_scan_summary.json"
    output_csv = args.root / "layer_scan_summary.csv"
    output_json.write_text(json.dumps(rows, indent=2) + "\n")
    with output_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(
        "start  first_top_p  passkey  number  macro  prefill  total  sparsity"
    )
    for row in rows:
        print(
            f"{row['start_layer_zero_based']:>5}  "
            f"{str(row['first_top_p_layer_one_based']):>11}  "
            f"{100 * row['passkey_score']:>7.2f}  "
            f"{100 * row['number_string_score']:>6.2f}  "
            f"{100 * row['macro_score']:>5.2f}  "
            f"{row['avg_prefill_latency_sec']:>7.3f}  "
            f"{row['avg_total_latency_sec']:>6.3f}  "
            f"{100 * row['global_token_sparsity']:>8.2f}%"
        )


if __name__ == "__main__":
    main()
