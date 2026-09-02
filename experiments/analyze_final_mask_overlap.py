#!/usr/bin/env python3
"""Compare final kernel key-token sets from FlexPrefill and AutoBlock dumps."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from experiments.baseline_sparsity import decode_array


def load_dump(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            token_hash = str(row["input_ids_sha256"])
            if token_hash in rows:
                raise ValueError(f"Duplicate input hash in {path}: {token_hash}")
            rows[token_hash] = row
    return rows


def merge_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def interval_size(intervals: list[tuple[int, int]]) -> int:
    return sum(end - start for start, end in intervals)


def intersection_size(
    left: list[tuple[int, int]], right: list[tuple[int, int]]
) -> int:
    i = j = total = 0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        if end > start:
            total += end - start
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


def auto_masks(sample: dict[str, Any]) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    seq_len = int(sample["input_tokens"])
    chunk_size = 32
    num_chunks = (seq_len + chunk_size - 1) // chunk_size
    expected_layers = set(range(32))
    observed_layers: set[int] = set()
    for selector in sample["selectors"]:
        if selector["kind"] != "token_compacted_final_key_runs":
            continue
        layer = int(selector["layer"])
        observed_layers.add(layer)
        offsets = decode_array(selector["run_row_offsets"]).astype(np.int64)
        starts = decode_array(selector["run_starts"]).astype(np.int64)
        lengths = decode_array(selector["run_lengths"]).astype(np.int64)
        head_to_group = decode_array(selector["head_to_group"]).astype(np.int64)
        row_shape = tuple(int(value) for value in selector["row_shape"])
        if len(row_shape) != 3 or row_shape[0] != 1:
            raise ValueError(f"Unsupported AutoBlock row shape: {row_shape}")
        _, num_groups, num_query_blocks = row_shape
        if offsets.size != num_groups * num_query_blocks + 1:
            raise ValueError("AutoBlock run offsets do not match row shape")
        if offsets[-1] != starts.size or starts.size != lengths.size:
            raise ValueError("AutoBlock run arrays are inconsistent")
        if int(lengths.sum()) != int(selector["selected_key_slots"]):
            raise ValueError("AutoBlock RLE does not preserve selected slot count")
        ends = starts + lengths
        if np.any(starts % chunk_size):
            raise ValueError("AutoBlock run start is not aligned to 32 tokens")
        if np.any((ends % chunk_size != 0) & (ends != seq_len)):
            raise ValueError("AutoBlock run end is not aligned to 32 tokens")
        run_counts = np.diff(offsets)
        run_rows = np.repeat(np.arange(run_counts.size), run_counts)
        diff = np.zeros((run_counts.size, num_chunks + 1), dtype=np.int16)
        chunk_starts = starts // chunk_size
        chunk_ends = (ends + chunk_size - 1) // chunk_size
        np.add.at(diff, (run_rows, chunk_starts), 1)
        np.add.at(diff, (run_rows, chunk_ends), -1)
        group_mask = np.cumsum(diff[:, :-1], axis=-1) > 0
        group_mask = group_mask.reshape(num_groups, num_query_blocks, num_chunks)
        result[layer] = group_mask[head_to_group]
    if observed_layers != expected_layers:
        raise ValueError(
            f"AutoBlock dump has layers {sorted(observed_layers)}, expected 0..31"
        )
    return result


def flex_masks(sample: dict[str, Any]) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    expected_layers = set(range(32))
    observed_layers: set[int] = set()
    seq_len = int(sample["input_tokens"])
    for selector in sample["selectors"]:
        if selector["kind"] != "flexprefill_final_block_ids":
            continue
        layer = int(selector["layer"])
        observed_layers.add(layer)
        block_size = int(selector["block_size"])
        num_blocks = int(selector["num_blocks"])
        vertical = decode_array(selector["vertical_indices"]).astype(np.int64)
        slash = decode_array(selector["slash_indices"]).astype(np.int64)
        extra_offsets = decode_array(selector["extra_head_offsets"]).astype(np.int64)
        extra_ids = decode_array(selector["extra_block_ids"]).astype(np.int64)
        if vertical.shape[0] != 1 or slash.shape[:2] != vertical.shape[:2]:
            raise ValueError("Unsupported FlexPrefill selector shape")
        num_heads = int(vertical.shape[1])
        if extra_offsets.size != num_heads + 1:
            raise ValueError("FlexPrefill extra offsets do not match head count")
        block_mask = np.zeros(
            (num_heads, num_blocks, num_blocks), dtype=np.bool_
        )
        causal = np.tri(num_blocks, num_blocks, dtype=np.bool_)
        for head in range(num_heads):
            vertical_keys = np.unique(vertical[0, head])
            vertical_keys = vertical_keys[
                (vertical_keys >= 0) & (vertical_keys < num_blocks)
            ]
            block_mask[head][:, vertical_keys] = causal[:, vertical_keys]
            key_blocks = np.arange(num_blocks, dtype=np.int64)
            for slash_offset in np.unique(slash[0, head]):
                query_blocks = key_blocks + int(slash_offset)
                valid = (query_blocks >= 0) & (query_blocks < num_blocks)
                block_mask[head, query_blocks[valid], key_blocks[valid]] = True
            begin, end = int(extra_offsets[head]), int(extra_offsets[head + 1])
            extras = extra_ids[begin:end]
            extras = extras[(extras >= 0) & (extras < num_blocks * num_blocks)]
            query_blocks, key_blocks = np.divmod(extras, num_blocks)
            valid = key_blocks <= query_blocks
            block_mask[head, query_blocks[valid], key_blocks[valid]] = True
        if block_size % 32:
            raise ValueError("FlexPrefill block size is not divisible by 32")
        result[layer] = np.repeat(block_mask, block_size // 32, axis=-1)[
            ..., : (seq_len + 31) // 32
        ]
    if observed_layers != expected_layers:
        raise ValueError(
            f"FlexPrefill dump has layers {sorted(observed_layers)}, expected 0..31"
        )
    return result


def compare_masks(
    left: dict[int, np.ndarray],
    right: dict[int, np.ndarray],
    *,
    seq_len: int,
) -> dict[str, Any]:
    if left.keys() != right.keys():
        missing_left = len(right.keys() - left.keys())
        missing_right = len(left.keys() - right.keys())
        raise ValueError(
            f"Selector row mismatch: missing_left={missing_left}, "
            f"missing_right={missing_right}"
        )
    totals = defaultdict(int)
    layers: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    last_width = seq_len - 32 * ((seq_len - 1) // 32)

    def weighted_slots(mask: np.ndarray) -> int:
        total = int(mask.sum(dtype=np.int64)) * 32
        if last_width != 32:
            total -= int(mask[..., -1].sum(dtype=np.int64)) * (32 - last_width)
        return total

    for layer in sorted(left):
        left_mask = left[layer]
        right_mask = right[layer]
        if left_mask.shape != right_mask.shape:
            raise ValueError(
                f"Layer {layer} mask shape mismatch: "
                f"{left_mask.shape} versus {right_mask.shape}"
            )
        left_slots = weighted_slots(left_mask)
        right_slots = weighted_slots(right_mask)
        intersection = weighted_slots(left_mask & right_mask)
        union = left_slots + right_slots - intersection
        rows = int(np.prod(left_mask.shape[:-1]))
        for bucket in (totals, layers[layer]):
            bucket["left_slots"] += left_slots
            bucket["right_slots"] += right_slots
            bucket["intersection_slots"] += intersection
            bucket["union_slots"] += union
            bucket["selector_rows"] += rows

    def finalize(values: dict[str, int]) -> dict[str, float | int]:
        left_slots = values["left_slots"]
        right_slots = values["right_slots"]
        intersection = values["intersection_slots"]
        union = values["union_slots"]
        return {
            **dict(values),
            "jaccard": intersection / union if union else 1.0,
            "left_covered_by_right": intersection / left_slots if left_slots else 1.0,
            "right_covered_by_left": intersection / right_slots if right_slots else 1.0,
            "mean_left_keys_per_row": left_slots / values["selector_rows"],
            "mean_right_keys_per_row": right_slots / values["selector_rows"],
            "mean_intersection_keys_per_row": intersection / values["selector_rows"],
        }

    return {
        "overall": finalize(totals),
        "layers": {str(layer): finalize(values) for layer, values in sorted(layers.items())},
    }


def selected_pair_count(masks: dict[int, np.ndarray], seq_len: int) -> int:
    query_tile = 128
    key_chunk = 32
    num_query = (seq_len + query_tile - 1) // query_tile
    num_key = (seq_len + key_chunk - 1) // key_chunk
    q_start = np.arange(num_query, dtype=np.int64)[:, None] * query_tile
    q_end = np.minimum(q_start + query_tile, seq_len)
    k_start = np.arange(num_key, dtype=np.int64)[None, :] * key_chunk
    k_end = np.minimum(k_start + key_chunk, seq_len)

    def prefix(query: np.ndarray) -> np.ndarray:
        width = np.maximum(k_end - k_start, 0)
        steps = np.maximum(query - k_start, 0)
        triangular_steps = np.minimum(steps, width)
        return (
            triangular_steps * (triangular_steps + 1) // 2
            + np.maximum(steps - width, 0) * width
        )

    weights = prefix(q_end) - prefix(q_start)
    return sum(
        int((mask * weights[None, :, :]).sum(dtype=np.int64))
        for mask in masks.values()
    )


def load_metrics(path: Path) -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    result = {str(row["input_ids_sha256"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate input hash in {path}")
    return result


def combine_comparisons(items: list[dict[str, Any]]) -> dict[str, Any]:
    combined = defaultdict(int)
    layer_combined: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for item in items:
        for key in ("left_slots", "right_slots", "intersection_slots", "union_slots", "selector_rows"):
            combined[key] += int(item["overall"][key])
        for layer, values in item["layers"].items():
            for key in ("left_slots", "right_slots", "intersection_slots", "union_slots", "selector_rows"):
                layer_combined[layer][key] += int(values[key])

    def finalize(values: dict[str, int]) -> dict[str, float | int]:
        intersection = values["intersection_slots"]
        left_slots = values["left_slots"]
        right_slots = values["right_slots"]
        union = values["union_slots"]
        rows = values["selector_rows"]
        return {
            **dict(values),
            "jaccard": intersection / union if union else 1.0,
            "left_covered_by_right": intersection / left_slots if left_slots else 1.0,
            "right_covered_by_left": intersection / right_slots if right_slots else 1.0,
            "mean_left_keys_per_row": left_slots / rows,
            "mean_right_keys_per_row": right_slots / rows,
            "mean_intersection_keys_per_row": intersection / rows,
        }

    return {
        "overall": finalize(combined),
        "layers": {
            layer: finalize(values)
            for layer, values in sorted(layer_combined.items(), key=lambda item: int(item[0]))
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flex", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--protected", type=Path, required=True)
    parser.add_argument("--flex-metrics", type=Path, required=True)
    parser.add_argument("--original-metrics", type=Path, required=True)
    parser.add_argument("--protected-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    flex = load_dump(args.flex)
    original = load_dump(args.original)
    protected = load_dump(args.protected)
    metric_sets = {
        "flexprefill": load_metrics(args.flex_metrics),
        "original_autoblock": load_metrics(args.original_metrics),
        "target_protected_autoblock": load_metrics(args.protected_metrics),
    }
    common = set(flex) & set(original) & set(protected)
    if common != set(flex) or common != set(original) or common != set(protected):
        raise ValueError(
            "Input hashes differ: "
            f"flex={len(flex)}, original={len(original)}, "
            f"protected={len(protected)}, common={len(common)}"
        )
    for name, metrics in metric_sets.items():
        if set(metrics) != common:
            raise ValueError(
                f"Metric hashes differ for {name}: metrics={len(metrics)}, "
                f"common={len(common)}"
            )

    flex_original = []
    flex_protected = []
    original_protected = []
    sample_rows = []
    pair_validation_totals = defaultdict(lambda: defaultdict(int))
    for token_hash in sorted(common):
        lengths = {
            int(flex[token_hash]["input_tokens"]),
            int(original[token_hash]["input_tokens"]),
            int(protected[token_hash]["input_tokens"]),
        }
        if len(lengths) != 1:
            raise ValueError(f"Input length mismatch for {token_hash}")
        seq_len = lengths.pop()
        flex_mask = flex_masks(flex[token_hash])
        original_mask = auto_masks(original[token_hash])
        protected_mask = auto_masks(protected[token_hash])
        mask_sets = {
            "flexprefill": flex_mask,
            "original_autoblock": original_mask,
            "target_protected_autoblock": protected_mask,
        }
        for name, masks in mask_sets.items():
            reconstructed_pairs = selected_pair_count(masks, seq_len)
            recorded_pairs = int(
                metric_sets[name][token_hash]["selected_token_pairs"]
            )
            if reconstructed_pairs != recorded_pairs:
                raise ValueError(
                    f"Pair mismatch for {name} {token_hash}: "
                    f"{reconstructed_pairs} != {recorded_pairs}"
                )
            pair_validation_totals[name]["reconstructed"] += reconstructed_pairs
            pair_validation_totals[name]["recorded"] += recorded_pairs
        first = compare_masks(flex_mask, original_mask, seq_len=seq_len)
        second = compare_masks(flex_mask, protected_mask, seq_len=seq_len)
        third = compare_masks(original_mask, protected_mask, seq_len=seq_len)
        flex_original.append(first)
        flex_protected.append(second)
        original_protected.append(third)
        sample_rows.append(
            {
                "input_ids_sha256": token_hash,
                "input_tokens": seq_len,
                "flex_vs_original_jaccard": first["overall"]["jaccard"],
                "original_covered_by_flex": first["overall"]["right_covered_by_left"],
                "flex_covered_by_original": first["overall"]["left_covered_by_right"],
                "flex_vs_protected_jaccard": second["overall"]["jaccard"],
                "protected_covered_by_flex": second["overall"]["right_covered_by_left"],
                "flex_covered_by_protected": second["overall"]["left_covered_by_right"],
            }
        )

    report = {
        "schema_version": 1,
        "definition": (
            "Exact final legal key-token slots per aligned sample, layer, Q head, "
            "and 128-token query tile. Ratios are micro-aggregated from summed slots."
        ),
        "sample_count": len(common),
        "input_hashes": sorted(common),
        "pair_count_validation": {
            name: {
                **dict(values),
                "exact_match": values["reconstructed"] == values["recorded"],
            }
            for name, values in pair_validation_totals.items()
        },
        "comparisons": {
            "flexprefill_vs_original_autoblock": combine_comparisons(flex_original),
            "flexprefill_vs_target_protected_autoblock": combine_comparisons(flex_protected),
            "original_vs_target_protected_autoblock": combine_comparisons(original_protected),
        },
        "samples": sample_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    csv_path = args.output.with_suffix(".layers.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "comparison",
                "layer",
                "jaccard",
                "left_covered_by_right",
                "right_covered_by_left",
                "mean_left_keys_per_row",
                "mean_right_keys_per_row",
                "mean_intersection_keys_per_row",
            ],
        )
        writer.writeheader()
        for name, comparison in report["comparisons"].items():
            for layer, values in comparison["layers"].items():
                writer.writerow(
                    {"comparison": name, "layer": layer, **{key: values[key] for key in writer.fieldnames[2:]}}
                )
    print(json.dumps({
        "sample_count": report["sample_count"],
        "comparisons": {
            name: values["overall"] for name, values in report["comparisons"].items()
        },
        "output": str(args.output),
        "layer_csv": str(csv_path),
    }, indent=2))


if __name__ == "__main__":
    main()
