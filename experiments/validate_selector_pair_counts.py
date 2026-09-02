#!/usr/bin/env python3
"""Cross-check reconstructed selector masks against recorded causal pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.analyze_final_mask_overlap import auto_masks, flex_masks, load_dump


def pair_weights(seq_len: int, query_tile: int = 128, key_chunk: int = 32) -> np.ndarray:
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

    return prefix(q_end) - prefix(q_start)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["flex", "auto"], required=True)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    args = parser.parse_args()

    dumps = load_dump(args.dump)
    metric_rows = [json.loads(line) for line in args.metrics.read_text().splitlines()]
    metric_by_hash = {row["input_ids_sha256"]: row for row in metric_rows}
    if set(dumps) != set(metric_by_hash):
        raise ValueError("Dump and metric hashes differ")
    reconstructed_total = 0
    recorded_total = 0
    for token_hash, sample in dumps.items():
        seq_len = int(sample["input_tokens"])
        weights = pair_weights(seq_len)
        masks = flex_masks(sample) if args.kind == "flex" else auto_masks(sample)
        reconstructed = sum(
            int((mask * weights[None, :, :]).sum(dtype=np.int64))
            for mask in masks.values()
        )
        recorded = int(metric_by_hash[token_hash]["selected_token_pairs"])
        if reconstructed != recorded:
            raise ValueError(
                f"Pair mismatch for {token_hash}: {reconstructed} != {recorded}"
            )
        reconstructed_total += reconstructed
        recorded_total += recorded
    print(json.dumps({
        "kind": args.kind,
        "samples": len(dumps),
        "reconstructed_selected_token_pairs": reconstructed_total,
        "recorded_selected_token_pairs": recorded_total,
        "exact_match": reconstructed_total == recorded_total,
    }, indent=2))


if __name__ == "__main__":
    main()
