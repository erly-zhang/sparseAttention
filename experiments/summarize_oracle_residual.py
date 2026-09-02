#!/usr/bin/env python3
"""Aggregate the Llama AutoBlock oracle-residual diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


METHOD = "shareprefill_ae3_token_block_auto_oracle_residual"
IDENTITY_FIELDS = {
    "group_index",
    "representative_head",
    "kv_head",
    "is_representative",
}
COUNT_FIELDS = (
    "selector_rows",
    "member_top8192_tokens",
    "shared_target_tokens",
    "shared_kernel_tokens",
    "shared_overlap_tokens",
    "shared_kernel_overlap_tokens",
    "residual_tokens",
    "residual_new_hits",
    "final_target_tokens",
    "final_kernel_tokens",
    "final_overlap_tokens",
    "kernel_overlap_tokens",
    "residual_shared_overlap_tokens",
    "residual_kernel_missing_tokens",
    "residual_causal_violation_tokens",
    "score_abs_difference_sum",
    "score_abs_difference_count",
)


def ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def official_accuracy(summary: dict[str, Any], task: str) -> float:
    results = summary["official_lm_eval_results"][task]
    values = [
        float(value)
        for key, value in results.items()
        if key != "alias"
        and "stderr" not in key
        and isinstance(value, (int, float))
    ]
    if len(values) != 1:
        raise ValueError(f"Expected one official score for {task}, got {results}")
    return values[0]


def add_counts(target: dict[str, float], values: dict[str, Any]) -> None:
    for field in COUNT_FIELDS:
        target[field] += float(values.get(field, 0.0))


def derived(values: dict[str, float]) -> dict[str, float | None]:
    denominator = values["member_top8192_tokens"]
    return {
        "shared_target_recall": ratio(values["shared_overlap_tokens"], denominator),
        "shared_kernel_recall": ratio(
            values["shared_kernel_overlap_tokens"], denominator
        ),
        "final_target_recall": ratio(values["final_overlap_tokens"], denominator),
        "final_kernel_recall": ratio(values["kernel_overlap_tokens"], denominator),
        "delta_target_recall": ratio(
            values["final_overlap_tokens"] - values["shared_overlap_tokens"],
            denominator,
        ),
        "delta_kernel_recall": ratio(
            values["kernel_overlap_tokens"]
            - values["shared_kernel_overlap_tokens"],
            denominator,
        ),
        "residual_precision": ratio(
            values["residual_new_hits"], values["residual_tokens"]
        ),
        "mean_abs_member_rep_score_difference": ratio(
            values["score_abs_difference_sum"],
            values["score_abs_difference_count"],
        ),
    }


def flatten_member_rows(
    rows: Iterable[dict[str, Any]],
) -> tuple[dict[str, float], dict[tuple[int, int], dict[str, Any]]]:
    aggregate: dict[str, float] = defaultdict(float)
    layer_heads: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        for layer_text, members in row.get("member_mask_fidelity_stats", {}).items():
            layer = int(layer_text)
            for head_text, values in members.items():
                if int(values["is_representative"]):
                    continue
                head = int(head_text)
                add_counts(aggregate, values)
                target = layer_heads.setdefault(
                    (layer, head),
                    {
                        "layer": layer,
                        "head": head,
                        **{field: values[field] for field in IDENTITY_FIELDS},
                        **{field: 0.0 for field in COUNT_FIELDS},
                    },
                )
                add_counts(target, values)
    return dict(aggregate), layer_heads


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    errors: list[str] = []

    for config_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if config_dir.name in {"logs"}:
            continue
        for task in ("kv_retrieval", "passkey"):
            task_dir = config_dir / METHOD / task
            summary_path = task_dir / "summary.json"
            metrics_path = task_dir / "online_metrics.jsonl"
            if not summary_path.is_file() or not metrics_path.is_file():
                continue
            summary = json.loads(summary_path.read_text())
            rows = read_jsonl(metrics_path)
            alignment = summary["input_alignment"]
            if len(rows) != 12 or summary["runtime"]["count"] != 12:
                errors.append(f"{config_dir.name}/{task}: expected 12 rows")
            if alignment["status"] != "passed":
                errors.append(f"{config_dir.name}/{task}: alignment failed")
            aggregate, layer_heads = flatten_member_rows(rows)
            metrics = derived(aggregate)
            representative_residual_tokens = sum(
                float(values.get("residual_tokens", 0.0))
                for row in rows
                for members in row.get("member_mask_fidelity_stats", {}).values()
                for values in members.values()
                if int(values["is_representative"])
            )
            selector_rows = sum(
                float(values.get("selector_rows", 0.0))
                for row in rows
                for values in row.get("per_layer_selector_stats", {}).values()
            )
            shared = int(
                summary["method_metadata"]["online_config"]["shared_topk_budget"]
            )
            residual = int(
                summary["method_metadata"]["online_config"][
                    "oracle_residual_tokens_per_nonrepresentative_head_row"
                ]
            )
            watched_ranges = summary["runtime"].get(
                "watched_key_range_stats", {}
            )
            watched_valid_slots = sum(
                float(values.get("valid_token_slots", 0.0))
                for values in watched_ranges.values()
            )
            watched_target_slots = sum(
                float(values.get("target_token_slots", 0.0))
                for values in watched_ranges.values()
            )
            watched_kernel_slots = sum(
                float(values.get("selected_token_slots", 0.0))
                for values in watched_ranges.values()
            )
            result = {
                "config": config_dir.name,
                "task": task,
                "shared_budget": shared,
                "residual_budget": residual,
                "nominal_budget": shared + residual,
                "samples": len(rows),
                "alignment_status": alignment["status"],
                "alignment_scope": alignment["alignment_scope"],
                "accuracy": official_accuracy(summary, task),
                "avg_prefill_latency_sec": summary["runtime"][
                    "avg_prefill_latency_sec"
                ],
                "avg_decode_latency_sec": summary["runtime"][
                    "avg_decode_latency_sec"
                ],
                "avg_total_latency_sec": summary["runtime"][
                    "avg_total_latency_sec"
                ],
                "global_token_sparsity": summary["runtime"][
                    "global_token_sparsity"
                ],
                "mean_target_keys_per_head_query_tile": ratio(
                    float(summary["runtime"]["target_mask_tokens"]), selector_rows
                ),
                "mean_kernel_keys_per_head_query_tile": ratio(
                    float(summary["runtime"]["compacted_key_tokens"]), selector_rows
                ),
                "watched_answer_range_target_keep_ratio": ratio(
                    watched_target_slots, watched_valid_slots
                ),
                "watched_answer_range_kernel_keep_ratio": ratio(
                    watched_kernel_slots, watched_valid_slots
                ),
                "residual_new_hits": aggregate["residual_new_hits"],
                "residual_tokens": aggregate["residual_tokens"],
                "mean_residual_tokens_per_nonrepresentative_row": ratio(
                    aggregate["residual_tokens"], aggregate["selector_rows"]
                ),
                "representative_residual_tokens": representative_residual_tokens,
                "residual_shared_overlap_tokens": aggregate[
                    "residual_shared_overlap_tokens"
                ],
                "residual_kernel_missing_tokens": aggregate[
                    "residual_kernel_missing_tokens"
                ],
                "residual_causal_violation_tokens": aggregate[
                    "residual_causal_violation_tokens"
                ],
                **metrics,
            }
            output_rows.append(result)
            for values in layer_heads.values():
                layer_rows.append(
                    {
                        "config": config_dir.name,
                        "task": task,
                        **values,
                        **derived(values),
                    }
                )
            for sample_index, row in enumerate(rows):
                manifests.append(
                    {
                        "config": config_dir.name,
                        "task": task,
                        "sample_index_within_run": sample_index,
                        "call_index": row.get("call_index"),
                        "input_tokens": row["input_tokens"],
                        "input_ids_sha256": row["input_ids_sha256"],
                        "dump_files": [
                            str(path)
                            for path in sorted(
                                (
                                    config_dir
                                    / "oracle_mask_dump"
                                    / task
                                ).glob(f"sample_{sample_index:03d}_layer_*.pt")
                            )
                        ],
                    }
                )

    if not output_rows:
        raise SystemExit("No completed oracle-residual task directories found")
    for row in output_rows:
        if row["residual_budget"] and row["residual_tokens"] <= 0:
            errors.append(f"{row['config']}/{row['task']}: no residual tokens")
        for field in (
            "representative_residual_tokens",
            "residual_shared_overlap_tokens",
            "residual_kernel_missing_tokens",
            "residual_causal_violation_tokens",
        ):
            if row[field] != 0:
                errors.append(
                    f"{row['config']}/{row['task']}: {field}={row[field]}"
                )
    summary_json = {
        "root": str(root),
        "method": METHOD,
        "results": output_rows,
        "validation_errors": errors,
    }
    (root / "oracle_residual_summary.json").write_text(
        json.dumps(summary_json, indent=2, sort_keys=True) + "\n"
    )
    with (root / "oracle_residual_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    with (root / "oracle_residual_layer_head.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(layer_rows[0]))
        writer.writeheader()
        writer.writerows(layer_rows)
    (root / "oracle_residual_sample_manifest.json").write_text(
        json.dumps(manifests, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary_json, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
