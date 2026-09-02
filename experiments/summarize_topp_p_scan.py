#!/usr/bin/env python3
"""Select the pure top-p run closest to a fixed-selector compute budget."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan_root", type=Path, required=True)
    parser.add_argument("--fixed_metrics", type=Path, required=True)
    parser.add_argument("--selected_output", type=Path, required=True)
    return parser.parse_args()


def kernel_keys_per_row(values: dict[str, Any]) -> float:
    rows = sum(int(count) for count in values["chosen_block_size_rows"].values())
    return float(values["compacted_key_tokens"]) / rows


def load_first_jsonl(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.loads(next(stream))


def main() -> None:
    args = parse_args()
    fixed = load_first_jsonl(args.fixed_metrics)
    fixed_sparsity = float(fixed["global_token_sparsity"])
    fixed_kernel_keys = kernel_keys_per_row(fixed)

    rows = []
    for summary_path in sorted(args.scan_root.glob("p_*/**/passkey/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        runtime = summary["runtime"]
        top_p = float(summary["run_args"]["target_token_top_p"])
        sparsity = float(runtime["global_token_sparsity"])
        kernel_keys = kernel_keys_per_row(runtime)
        rows.append(
            {
                "top_p": top_p,
                "global_token_sparsity": sparsity,
                "kernel_keys_per_row": kernel_keys,
                "sparsity_abs_error": abs(sparsity - fixed_sparsity),
                "kernel_keys_abs_error": abs(kernel_keys - fixed_kernel_keys),
                "summary_path": str(summary_path),
            }
        )
    if not rows:
        raise RuntimeError(f"No completed scan summaries under {args.scan_root}")

    rows.sort(
        key=lambda row: (
            row["sparsity_abs_error"],
            row["kernel_keys_abs_error"],
        )
    )
    selected = rows[0]
    payload = {
        "fixed": {
            "global_token_sparsity": fixed_sparsity,
            "kernel_keys_per_row": fixed_kernel_keys,
        },
        "selected_top_p": selected["top_p"],
        "selected": selected,
        "candidates": sorted(rows, key=lambda row: row["top_p"]),
    }
    args.scan_root.mkdir(parents=True, exist_ok=True)
    (args.scan_root / "scan_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "| top-p | pair sparsity | kernel keys/row | sparsity error | key error |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in payload["candidates"]:
        lines.append(
            f"| {row['top_p']:.3f} | {row['global_token_sparsity']:.4%} | "
            f"{row['kernel_keys_per_row']:.1f} | {row['sparsity_abs_error']:.4%} | "
            f"{row['kernel_keys_abs_error']:.1f} |"
        )
    (args.scan_root / "scan_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    args.selected_output.write_text(f"{selected['top_p']:.3f}\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
