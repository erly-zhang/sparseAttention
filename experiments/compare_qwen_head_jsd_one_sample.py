#!/usr/bin/env python3
"""Compare within-layer Q-head attention similarity on one shared prompt."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


WORK = Path("/home/ubuntu/work")
sys.path.insert(0, str(WORK))

from experiments.run_shared_layer_mask_experiment import (  # noqa: E402
    collect_last_q_attentions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--data",
        default=str(
            WORK
            / "experiments/data/infinitebench_benchmark_specific_calibration"
            / "filtered_data/kv_retrieval.jsonl"
        ),
    )
    parser.add_argument("--source-row", type=int, default=0)
    parser.add_argument("--last-q", type=int, default=32)
    parser.add_argument("--max-input-length", type=int, default=130944)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--token-chunk-size", type=int, default=512)
    return parser.parse_args()


def load_source(path: Path, row_index: int) -> dict:
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index == row_index:
                return json.loads(line)
    raise IndexError(f"No source row {row_index} in {path}")


def build_prompt(row: dict) -> str:
    return (
        "Extract the value corresponding to the specified key in the JSON "
        "object below.\n\n"
        f"{row['context']}\n\n{row['input']}"
    )


def tensor_sha256(input_ids: torch.Tensor) -> str:
    values = input_ids.detach().to(device="cpu", dtype=torch.int64).numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def pairwise_jsd(
    layer_attention: torch.Tensor,
    *,
    device: torch.device,
    token_chunk_size: int,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean normalized JSD and the project's mean sqrt-JSD distance."""

    probabilities = layer_attention.to(device=device, dtype=torch.float32)
    probabilities.clamp_min_(0)
    probabilities.add_(eps)
    probabilities.div_(probabilities.sum(dim=-1, keepdim=True).clamp_min(eps))
    heads, queries, sequence_length = probabilities.shape
    per_query_jsd = torch.zeros(
        (heads, heads, queries), dtype=torch.float32, device=device
    )
    for start in range(0, sequence_length, token_chunk_size):
        chunk = probabilities[..., start : start + token_chunk_size]
        p = chunk[:, None, :, :]
        q = chunk[None, :, :, :]
        midpoint = 0.5 * (p + q)
        per_query_jsd.add_(
            0.5
            * (
                (p * (torch.log(p) - torch.log(midpoint))).sum(dim=-1)
                + (q * (torch.log(q) - torch.log(midpoint))).sum(dim=-1)
            )
        )
    normalized = torch.clamp(per_query_jsd / math.log(2.0), 0.0, 1.0)
    raw_matrix = normalized.mean(dim=-1)
    sqrt_matrix = torch.sqrt(normalized).mean(dim=-1)
    diagonal = torch.arange(heads, device=device)
    raw_matrix[diagonal, diagonal] = 0.0
    sqrt_matrix[diagonal, diagonal] = 0.0
    return raw_matrix.cpu(), sqrt_matrix.cpu()


def summarize_matrix(
    raw: torch.Tensor,
    sqrt_distance: torch.Tensor,
    *,
    num_kv_heads: int,
) -> dict:
    num_heads = raw.shape[0]
    upper = torch.triu_indices(num_heads, num_heads, offset=1)
    raw_values = raw[upper[0], upper[1]]
    sqrt_values = sqrt_distance[upper[0], upper[1]]
    q_per_kv = num_heads // num_kv_heads
    kv_family = torch.arange(num_heads) // q_per_kv
    same_kv = kv_family[upper[0]] == kv_family[upper[1]]
    cross_kv = ~same_kv

    def stats(values: torch.Tensor) -> dict:
        return {
            "mean": float(values.mean()),
            "median": float(values.median()),
            "p90": float(torch.quantile(values, 0.90)),
            "max": float(values.max()),
        }

    return {
        "pair_count": int(raw_values.numel()),
        "normalized_jsd": stats(raw_values),
        "sqrt_jsd_distance": stats(sqrt_values),
        "similarity_one_minus_sqrt_jsd_mean": float(1.0 - sqrt_values.mean()),
        "same_kv_normalized_jsd_mean": float(raw_values[same_kv].mean()),
        "cross_kv_normalized_jsd_mean": float(raw_values[cross_kv].mean()),
        "same_kv_sqrt_jsd_mean": float(sqrt_values[same_kv].mean()),
        "cross_kv_sqrt_jsd_mean": float(sqrt_values[cross_kv].mean()),
    }


def main() -> None:
    args = parse_args()
    output = args.output_root / args.label
    output.mkdir(parents=True, exist_ok=True)
    row = load_source(Path(args.data), args.source_row)
    prompt = build_prompt(row)
    prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    encoded_for_manifest = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_input_length,
    )
    manifest = {
        "label": args.label,
        "model": args.model,
        "source_data": args.data,
        "source_row": args.source_row,
        "source_id": row.get("id"),
        "prompt_chars": len(prompt),
        "prompt_sha256": prompt_sha,
        "context_sha256": hashlib.sha256(row["context"].encode()).hexdigest(),
        "input_sha256": hashlib.sha256(row["input"].encode()).hexdigest(),
        "input_tokens": int(encoded_for_manifest["input_ids"].shape[1]),
        "input_ids_sha256": tensor_sha256(encoded_for_manifest["input_ids"]),
        "last_q": args.last_q,
        "max_input_length": args.max_input_length,
        "chat_template_applied": False,
        "metric": {
            "primary": "mean_q JSD(p_hq || p_gq) / ln(2)",
            "project_distance": "mean_q sqrt(JSD(p_hq || p_gq) / ln(2))",
            "range": [0.0, 1.0],
            "lower_is_more_similar": True,
        },
    }
    del encoded_for_manifest
    (output / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    started = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        trust_remote_code=True,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model.eval()
    collector_args = argparse.Namespace(
        max_input_length=args.max_input_length,
        last_q=args.last_q,
        chunk_size=args.chunk_size,
    )
    attentions, sequence_length, input_ids, attention_mask = (
        collect_last_q_attentions(model, tokenizer, prompt, collector_args)
    )
    actual_hash = tensor_sha256(input_ids)
    if actual_hash != manifest["input_ids_sha256"]:
        raise RuntimeError("Collector token IDs differ from the input manifest")

    num_layers = int(model.config.num_hidden_layers)
    num_heads = int(model.config.num_attention_heads)
    num_kv_heads = int(model.config.num_key_value_heads)
    if attentions.shape != (num_layers, num_heads, args.last_q, sequence_length):
        raise RuntimeError(f"Unexpected attention shape: {attentions.shape}")

    raw_matrices = []
    sqrt_matrices = []
    layer_rows = []
    for layer in range(num_layers):
        raw, sqrt_distance = pairwise_jsd(
            attentions[layer],
            device=torch.device("cuda:0"),
            token_chunk_size=args.token_chunk_size,
        )
        raw_matrices.append(raw.numpy())
        sqrt_matrices.append(sqrt_distance.numpy())
        layer_rows.append(
            {
                "layer": layer,
                **summarize_matrix(
                    raw, sqrt_distance, num_kv_heads=num_kv_heads
                ),
            }
        )
        torch.cuda.empty_cache()

    raw_stack = np.stack(raw_matrices)
    sqrt_stack = np.stack(sqrt_matrices)
    np.savez_compressed(
        output / "pairwise_jsd_matrices.npz",
        normalized_jsd=raw_stack,
        sqrt_jsd_distance=sqrt_stack,
    )
    with (output / "per_layer_jsd.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "layer",
                "pair_count",
                "normalized_jsd_mean",
                "normalized_jsd_median",
                "normalized_jsd_p90",
                "normalized_jsd_max",
                "sqrt_jsd_distance_mean",
                "sqrt_jsd_distance_median",
                "sqrt_jsd_distance_p90",
                "sqrt_jsd_distance_max",
                "similarity_one_minus_sqrt_jsd_mean",
                "same_kv_normalized_jsd_mean",
                "cross_kv_normalized_jsd_mean",
                "same_kv_sqrt_jsd_mean",
                "cross_kv_sqrt_jsd_mean",
            ],
        )
        writer.writeheader()
        for row_stats in layer_rows:
            writer.writerow(
                {
                    "layer": row_stats["layer"],
                    "pair_count": row_stats["pair_count"],
                    **{
                        f"normalized_jsd_{name}": value
                        for name, value in row_stats["normalized_jsd"].items()
                    },
                    **{
                        f"sqrt_jsd_distance_{name}": value
                        for name, value in row_stats["sqrt_jsd_distance"].items()
                    },
                    **{
                        key: value
                        for key, value in row_stats.items()
                        if key
                        in {
                            "similarity_one_minus_sqrt_jsd_mean",
                            "same_kv_normalized_jsd_mean",
                            "cross_kv_normalized_jsd_mean",
                            "same_kv_sqrt_jsd_mean",
                            "cross_kv_sqrt_jsd_mean",
                        }
                    },
                }
            )

    upper = np.triu_indices(num_heads, k=1)
    all_raw = raw_stack[:, upper[0], upper[1]].reshape(-1)
    all_sqrt = sqrt_stack[:, upper[0], upper[1]].reshape(-1)
    q_per_kv = num_heads // num_kv_heads
    family = np.arange(num_heads) // q_per_kv
    same_pair = family[upper[0]] == family[upper[1]]
    same_all = np.tile(same_pair, num_layers)
    summary = {
        **manifest,
        "sequence_length": sequence_length,
        "num_layers": num_layers,
        "num_q_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "q_heads_per_kv_head": q_per_kv,
        "query_scope": f"last_{args.last_q}_causal_rows",
        "total_head_pairs": int(all_raw.size),
        "overall_normalized_jsd_mean": float(all_raw.mean()),
        "overall_normalized_jsd_median": float(np.median(all_raw)),
        "overall_normalized_jsd_p90": float(np.quantile(all_raw, 0.90)),
        "overall_sqrt_jsd_distance_mean": float(all_sqrt.mean()),
        "overall_similarity_one_minus_sqrt_jsd_mean": float(1.0 - all_sqrt.mean()),
        "same_kv_normalized_jsd_mean": float(all_raw[same_all].mean()),
        "cross_kv_normalized_jsd_mean": float(all_raw[~same_all].mean()),
        "same_kv_sqrt_jsd_mean": float(all_sqrt[same_all].mean()),
        "cross_kv_sqrt_jsd_mean": float(all_sqrt[~same_all].mean()),
        "elapsed_sec": time.time() - started,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "per_layer": layer_rows,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "per_layer"}, indent=2))

    del attentions, input_ids, attention_mask, model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
