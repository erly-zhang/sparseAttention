#!/usr/bin/env python3
"""Check a FlexPrefill selector dump through the official transform."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.baseline_sparsity import (
    BaselineSparsityInstrumentation,
    decode_array,
)
from flex_prefill.ops.flex_prefill_attention import transform_veritcal_slash_idx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    args = parser.parse_args()
    sample = json.loads(args.dump.read_text().splitlines()[0])
    metric = json.loads(args.metrics.read_text().splitlines()[0])
    seq_len = int(sample["input_tokens"])
    total_pairs = 0
    total_blocks = 0
    layer_pairs = []
    for selector in sample["selectors"]:
        block_size = int(selector["block_size"])
        vertical = torch.from_numpy(
            decode_array(selector["vertical_indices"]).astype(np.int64)
        )
        slash = torch.from_numpy(
            decode_array(selector["slash_indices"]).astype(np.int64)
        )
        extras_offsets = decode_array(selector["extra_head_offsets"]).astype(np.int64)
        extras = decode_array(selector["extra_block_ids"]).astype(np.int64)
        blocks = transform_veritcal_slash_idx(
            vertical, slash, int(selector["num_blocks"])
        )
        current = 0
        for head, base in enumerate(blocks[0]):
            begin, end = int(extras_offsets[head]), int(extras_offsets[head + 1])
            extra = torch.from_numpy(extras[begin:end])
            final = torch.unique(torch.cat((base.to(torch.int64), extra)))
            pairs, blocks_count = BaselineSparsityInstrumentation._block_pair_count(
                final, seq_len, block_size
            )
            current += int(pairs.item())
            total_blocks += int(blocks_count.item())
        total_pairs += current
        layer_pairs.append((int(selector["layer"]), current))
    print(json.dumps({
        "official_reconstructed_pairs": total_pairs,
        "recorded_pairs": int(metric["selected_token_pairs"]),
        "official_reconstructed_blocks": total_blocks,
        "recorded_blocks": int(metric["selected_blocks"]),
        "exact_match": total_pairs == int(metric["selected_token_pairs"]),
        "layer_pairs": layer_pairs,
    }, indent=2))


if __name__ == "__main__":
    main()
