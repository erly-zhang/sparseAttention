#!/usr/bin/env python3
"""Summarize answer-token selector mass and retention for fixed/top-p AutoBlock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


METHODS = {
    "fixed8192": "shareprefill_ae3_token_block_auto_fixed_mass_profile",
    "topp90": "shareprefill_ae3_token_block_auto_topp",
}
TASK_RANGES = {
    "passkey": ("48:53", "58:63"),
    "number_string": ("46:56", "61:71"),
}
BASE_FIELDS = (
    "probability_mass",
    "selector_rows",
    "valid_token_slots",
    "target_token_slots",
    "selected_token_slots",
    "sink_token_slots",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def combine_ranges(
    watched: Mapping[str, Mapping[str, Any]], ranges: tuple[str, ...]
) -> dict[str, float]:
    selector_rows = max(float(watched[name]["selector_rows"]) for name in ranges)
    probability_mass = sum(float(watched[name]["probability_mass"]) for name in ranges)
    valid_slots = sum(float(watched[name]["valid_token_slots"]) for name in ranges)
    target_slots = sum(float(watched[name]["target_token_slots"]) for name in ranges)
    selected_slots = sum(
        float(watched[name]["selected_token_slots"]) for name in ranges
    )
    sink_slots = sum(float(watched[name]["sink_token_slots"]) for name in ranges)
    return {
        "probability_mass_per_selector_row": probability_mass / selector_rows,
        "target_keep_ratio": target_slots / valid_slots,
        "final_kernel_keep_ratio": selected_slots / valid_slots,
        "sink_protection_ratio": sink_slots / valid_slots,
        "final_beyond_sink_ratio": (selected_slots - sink_slots) / valid_slots,
    }


def aggregate_layer_rows(
    rows: list[Mapping[str, Any]], ranges: tuple[str, ...]
) -> dict[str, dict[str, float]]:
    totals: dict[str, dict[str, dict[str, float]]] = {}
    for row in rows:
        for layer, layer_values in row["per_layer_selector_stats"].items():
            watched = layer_values["watched_key_ranges"]
            layer_totals = totals.setdefault(layer, {})
            for range_name in ranges:
                range_totals = layer_totals.setdefault(
                    range_name, {field: 0.0 for field in BASE_FIELDS}
                )
                for field in BASE_FIELDS:
                    range_totals[field] += float(watched[range_name][field])
    return {
        layer: combine_ranges(watched, ranges)
        for layer, watched in sorted(totals.items(), key=lambda item: int(item[0]))
    }


def main() -> None:
    args = parse_args()
    report: dict[str, Any] = {"root": str(args.root), "tasks": {}}
    for task, ranges in TASK_RANGES.items():
        task_report: dict[str, Any] = {}
        for method_name, method_dir in METHODS.items():
            task_dir = args.root / method_dir / task
            summary = json.loads((task_dir / "summary.json").read_text())
            runtime = summary["runtime"]
            rows = [
                json.loads(line)
                for line in (task_dir / "online_metrics.jsonl").read_text().splitlines()
                if line.strip()
            ]
            combined = combine_ranges(runtime["watched_key_range_stats"], ranges)
            answer_tokens = sum(
                int(end) - int(start)
                for start, end in (value.split(":") for value in ranges)
            )
            uniform_mass = answer_tokens / float(runtime["avg_input_tokens"])
            block_rows = {
                str(size): int(count)
                for size, count in runtime["chosen_block_size_rows"].items()
            }
            block_total = sum(block_rows.values())
            task_report[method_name] = {
                "count": int(runtime["count"]),
                "input_alignment": summary["input_alignment"],
                "answer_ranges": list(ranges),
                "answer_token_count": answer_tokens,
                "uniform_probability_mass": uniform_mass,
                **combined,
                "probability_concentration_vs_uniform": (
                    combined["probability_mass_per_selector_row"] / uniform_mass
                ),
                "global_token_sparsity": float(runtime["global_token_sparsity"]),
                "chosen_block_size_fraction": {
                    size: count / block_total for size, count in block_rows.items()
                },
                "per_layer": aggregate_layer_rows(rows, ranges),
            }
        report["tasks"][task] = task_report
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
