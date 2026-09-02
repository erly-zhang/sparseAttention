#!/usr/bin/env python3
"""Summarize representative/member TopK fidelity and answer-span survival."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


TASKS = ("passkey", "number_string")
COUNT_FIELDS = (
    "selector_rows",
    "member_target_tokens",
    "representative_target_tokens",
    "representative_overlap_tokens",
    "final_overlap_tokens",
    "answer_valid_slots",
    "answer_member_target_slots",
    "answer_representative_target_slots",
    "answer_final_kernel_slots",
)


def parse_mapping(value: str) -> tuple[str, Path]:
    try:
        name, path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected NAME=METHOD_ROOT") from error
    return name, Path(path).resolve()


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else float("nan")


def empty_totals() -> dict[str, float]:
    return {field: 0.0 for field in COUNT_FIELDS}


def add_record(target: dict[str, float], values: dict[str, Any]) -> None:
    for field in COUNT_FIELDS[:5]:
        target[field] += float(values.get(field, 0.0))
    for range_values in values.get("watched_key_ranges", {}).values():
        target["answer_valid_slots"] += float(
            range_values.get("valid_slots", 0.0)
        )
        target["answer_member_target_slots"] += float(
            range_values.get("member_target_slots", 0.0)
        )
        target["answer_representative_target_slots"] += float(
            range_values.get("representative_target_slots", 0.0)
        )
        target["answer_final_kernel_slots"] += float(
            range_values.get("final_kernel_slots", 0.0)
        )


def finalize(totals: dict[str, float]) -> dict[str, float]:
    own = totals["member_target_tokens"]
    representative = totals["representative_target_tokens"]
    overlap = totals["representative_overlap_tokens"]
    final_overlap = totals["final_overlap_tokens"]
    union = own + representative - overlap
    answer_valid = totals["answer_valid_slots"]
    return {
        **totals,
        "representative_recall_of_member_topk": ratio(overlap, own),
        "representative_member_jaccard": ratio(overlap, union),
        "final_kernel_recall_of_member_topk": ratio(final_overlap, own),
        "block_projection_recovery_over_representative": ratio(
            final_overlap - overlap, own
        ),
        "answer_member_topk_rate": ratio(
            totals["answer_member_target_slots"], answer_valid
        ),
        "answer_representative_topk_rate": ratio(
            totals["answer_representative_target_slots"], answer_valid
        ),
        "answer_final_kernel_rate": ratio(
            totals["answer_final_kernel_slots"], answer_valid
        ),
    }


def summarize_task(path: Path) -> dict[str, Any]:
    metrics_path = path / "online_metrics.jsonl"
    rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise RuntimeError(f"No rows in {metrics_path}")
    aggregate = empty_totals()
    per_layer: dict[int, dict[str, float]] = defaultdict(empty_totals)
    same_kv = empty_totals()
    cross_kv = empty_totals()
    representative_heads = empty_totals()
    projection_target = 0.0
    projection_covered = 0.0
    resolved_ranges = []
    for row in rows:
        projection_target += float(row.get("target_mask_tokens", 0.0))
        projection_covered += float(row.get("covered_target_tokens", 0.0))
        resolved_ranges.append(row.get("resolved_watched_key_ranges", []))
        layer_stats = row.get("member_mask_fidelity_stats", {})
        for layer_text, members in layer_stats.items():
            layer = int(layer_text)
            representative_kv = {
                int(values["representative_head"]): int(values["kv_head"])
                for values in members.values()
                if int(values.get("is_representative", 0))
            }
            for values in members.values():
                if int(values.get("is_representative", 0)):
                    add_record(representative_heads, values)
                    continue
                add_record(aggregate, values)
                add_record(per_layer[layer], values)
                rep_head = int(values["representative_head"])
                target = (
                    same_kv
                    if int(values["kv_head"]) == representative_kv[rep_head]
                    else cross_kv
                )
                add_record(target, values)
    return {
        "count": len(rows),
        "input_hashes": [row.get("input_ids_sha256") for row in rows],
        "resolved_watched_key_ranges": resolved_ranges,
        "global_token_sparsity": ratio(
            sum(float(row.get("causal_token_pairs", 0.0)) for row in rows)
            - sum(float(row.get("selected_token_pairs", 0.0)) for row in rows),
            sum(float(row.get("causal_token_pairs", 0.0)) for row in rows),
        ),
        "representative_target_coverage_after_projection": ratio(
            projection_covered, projection_target
        ),
        "non_representative_members": finalize(aggregate),
        "same_kv_members": finalize(same_kv),
        "cross_kv_members": finalize(cross_kv),
        "representative_heads": finalize(representative_heads),
        "per_layer": {
            str(layer): finalize(values)
            for layer, values in sorted(per_layer.items())
        },
    }


def percent(value: float) -> str:
    return "--" if math.isnan(value) else f"{100.0 * value:.2f}%"


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# AutoBlock K=3 Member-Mask Fidelity",
        "",
        "Primary fidelity metrics exclude each group's representative head, "
        "whose self-overlap is one by construction.",
        "",
        "| Model | Task | Rep recall of member TopK | Rep/member Jaccard | "
        "Final-kernel recall of member TopK | Answer in member TopK | "
        "Answer in rep TopK | Answer in final kernel | Pair sparsity |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, model_data in report["models"].items():
        for task, task_data in model_data.items():
            values = task_data["non_representative_members"]
            lines.append(
                "| "
                + " | ".join(
                    (
                        model,
                        task,
                        percent(values["representative_recall_of_member_topk"]),
                        percent(values["representative_member_jaccard"]),
                        percent(values["final_kernel_recall_of_member_topk"]),
                        percent(values["answer_member_topk_rate"]),
                        percent(values["answer_representative_topk_rate"]),
                        percent(values["answer_final_kernel_rate"]),
                        percent(task_data["global_token_sparsity"]),
                    )
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## KV-family split",
            "",
            "| Model | Task | Member relation | Rep recall | Jaccard | "
            "Final-kernel recall |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for model, model_data in report["models"].items():
        for task, task_data in model_data.items():
            for key, label in (
                ("same_kv_members", "same KV head"),
                ("cross_kv_members", "different KV head"),
            ):
                values = task_data[key]
                lines.append(
                    "| "
                    + " | ".join(
                        (
                            model,
                            task,
                            label,
                            percent(values["representative_recall_of_member_topk"]),
                            percent(values["representative_member_jaccard"]),
                            percent(values["final_kernel_recall_of_member_topk"]),
                        )
                    )
                    + " |"
                )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_layer_csv(report: dict[str, Any], path: Path) -> None:
    fields = [
        "model",
        "task",
        "layer",
        "representative_recall_of_member_topk",
        "representative_member_jaccard",
        "final_kernel_recall_of_member_topk",
        "answer_member_topk_rate",
        "answer_representative_topk_rate",
        "answer_final_kernel_rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for model, model_data in report["models"].items():
            for task, task_data in model_data.items():
                for layer, values in task_data["per_layer"].items():
                    writer.writerow(
                        {
                            "model": model,
                            "task": task,
                            "layer": layer,
                            **{field: values[field] for field in fields[3:]},
                        }
                    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=parse_mapping, required=True)
    parser.add_argument("--task", action="append")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    tasks = tuple(args.task or TASKS)
    report = {
        "definition": {
            "target": "fixed TopK=8192 per layer/head/128-query tile",
            "primary_population": "non-representative member Q heads",
            "representative_recall": "member TopK positions also in representative TopK",
            "final_recall": "member TopK positions present in final whole-block kernel mask",
        },
        "models": {
            name: {
                task: summarize_task(root / task)
                for task in tasks
            }
            for name, root in args.run
        },
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "member_mask_fidelity_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_markdown(report, output_dir / "member_mask_fidelity_summary.md")
    write_layer_csv(report, output_dir / "member_mask_fidelity_by_layer.csv")


if __name__ == "__main__":
    main()
