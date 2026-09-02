#!/usr/bin/env python3
"""Validate and summarize the Dense-to-Top-p AutoBlock layer scan."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


START_LAYERS = (0, 6, 10, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24)
TASKS = ("passkey", "number_string")
METHOD = "shareprefill_ae3_token_block_auto_dense_topp_mass"
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
        sparse_layer_count = NUM_LAYERS - start_layer
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
            online_config = summary["method_metadata"]["online_config"]
            if online_config["target_top_p_start_layer_zero_based"] != start_layer:
                raise ValueError(f"Metadata start mismatch for {start_layer} {task}")
            if online_config["dense_layers"] != list(range(start_layer)):
                raise ValueError(f"Dense prefix mismatch for {start_layer} {task}")
            hashes = [metric["input_ids_sha256"] for metric in metrics]
            reference_hashes = hashes_by_task.setdefault(task, hashes)
            if hashes != reference_hashes:
                raise ValueError(f"Input hash mismatch for start={start_layer} {task}")

            for metric in metrics:
                layer_stats = metric["per_layer_selector_stats"]
                expected_layers = {str(layer) for layer in range(start_layer, NUM_LAYERS)}
                if set(layer_stats) != expected_layers:
                    raise ValueError(
                        f"Sparse-layer mismatch start={start_layer} task={task}"
                    )
                for layer in range(start_layer, NUM_LAYERS):
                    modes = layer_stats[str(layer)]["target_selection_mode_rows"]
                    if set(modes) != {"probability_top_p"}:
                        raise ValueError(
                            f"Top-p mode mismatch start={start_layer} task={task} "
                            f"layer={layer}: {modes}"
                        )

            runtime = summary["runtime"]
            sparse_selected_pairs = int(runtime["selected_token_pairs"])
            sparse_causal_pairs = int(runtime["causal_token_pairs"])
            dense_causal_pairs = (
                sparse_causal_pairs * start_layer // sparse_layer_count
            )
            task_rows.append(
                {
                    "task": task,
                    "score": task_score(summary, task),
                    "prefill": float(runtime["avg_prefill_latency_sec"]),
                    "total": float(runtime["avg_total_latency_sec"]),
                    "overall_selected_pairs": sparse_selected_pairs + dense_causal_pairs,
                    "overall_causal_pairs": sparse_causal_pairs + dense_causal_pairs,
                    "suffix_mass_coverage": float(
                        runtime["target_probability_coverage_ratio"]
                    ),
                    "suffix_count_coverage": float(
                        runtime["target_mask_coverage_ratio"]
                    ),
                }
            )

        selected_pairs = sum(row["overall_selected_pairs"] for row in task_rows)
        causal_pairs = sum(row["overall_causal_pairs"] for row in task_rows)
        rows.append(
            {
                "start_layer_zero_based": start_layer,
                "first_top_p_layer_one_based": start_layer + 1,
                "dense_layer_count": start_layer,
                "top_p_layer_count": sparse_layer_count,
                "passkey_score": task_rows[0]["score"],
                "number_string_score": task_rows[1]["score"],
                "macro_score": sum(row["score"] for row in task_rows) / 2,
                "avg_prefill_latency_sec": sum(row["prefill"] for row in task_rows) / 2,
                "avg_total_latency_sec": sum(row["total"] for row in task_rows) / 2,
                "overall_pair_sparsity": 1.0 - selected_pairs / causal_pairs,
                "suffix_target_mass_coverage": sum(
                    row["suffix_mass_coverage"] for row in task_rows
                ) / 2,
                "suffix_target_count_coverage": sum(
                    row["suffix_count_coverage"] for row in task_rows
                ) / 2,
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

    print("start first passkey number macro prefill sparsity mass_cov count_cov")
    for row in rows:
        print(
            f"{row['start_layer_zero_based']:>5} "
            f"{row['first_top_p_layer_one_based']:>5} "
            f"{100 * row['passkey_score']:>7.2f} "
            f"{100 * row['number_string_score']:>6.2f} "
            f"{100 * row['macro_score']:>6.2f} "
            f"{row['avg_prefill_latency_sec']:>7.3f} "
            f"{100 * row['overall_pair_sparsity']:>8.2f}% "
            f"{100 * row['suffix_target_mass_coverage']:>8.2f}% "
            f"{100 * row['suffix_target_count_coverage']:>8.2f}%"
        )


if __name__ == "__main__":
    main()
