#!/usr/bin/env python3
"""
Graph2Vec clustering for attention heads within each layer.

Pipeline (aligned with run_ab_similarity_experiment.py):
1. Build directional similarity on last query row + K%% mass rule.
2. Binarize: 1 if similarity >= threshold.
3. For each head, extract a directed ego graph from the binary adjacency.
4. Graph2Vec (directed WL subtrees + Doc2Vec) embed each ego graph.
5. KMeans on embeddings (k=2 by default).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

CLUSTER_DIR = Path(__file__).resolve().parent
WORK_ROOT = CLUSTER_DIR.parents[1]
ATTN_SCRIPTS = WORK_ROOT / "attention_map" / "scripts"
for path in (CLUSTER_DIR, ATTN_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from graph2vec_head import graph2vec_head_clustering
from layer_head_similarity import (
    binarize_similarity,
    compute_layer_directional_similarity,
    discover_sample_dirs,
    infer_sample_dims,
    partition_edge_stats,
    sample_slug,
)


def binary_cache_name(k_percent: float, threshold: float) -> str:
    k_tag = str(k_percent).replace(".", "p")
    thr_tag = str(threshold).replace(".", "p")
    return f"similarity_binary_k{k_tag}_thr{thr_tag}.npy"


def directed_graph_stats(binary: np.ndarray) -> tuple[np.ndarray, int, float]:
    ad = binary.astype(bool).copy()
    np.fill_diagonal(ad, False)
    n = binary.shape[0]
    directed_edge_count = int(ad.sum())
    binary_density = float(directed_edge_count / (n * (n - 1))) if n > 1 else 0.0
    return ad, directed_edge_count, binary_density


def load_or_compute_binary(
    sample_dir: Path,
    layer: int,
    *,
    cache_dir: Path | None,
    k_percent: float,
    threshold: float,
) -> np.ndarray:
    cache_name = binary_cache_name(k_percent, threshold)
    if cache_dir is not None:
        cached = cache_dir / f"layer_{layer:02d}" / cache_name
        if cached.is_file():
            return np.load(cached)

    num_layers, num_heads, seq_len, is_last_row = infer_sample_dims(sample_dir)
    q = seq_len - 1
    sim = compute_layer_directional_similarity(
        sample_dir,
        layer,
        num_heads=num_heads,
        seq_len=seq_len,
        q=q,
        k_percent=k_percent,
        is_last_row=is_last_row,
    )
    return binarize_similarity(sim, threshold)


def process_sample(
    sample_dir: Path,
    out_dir: Path,
    *,
    layer_ids: List[int],
    k_percent: float,
    threshold: float,
    cache_dir: Path | None,
    k_clusters: int,
    ego_radius: int,
    vector_size: int,
    wl_iterations: int,
    epochs: int,
    random_state: int,
) -> Dict:
    num_layers, num_heads, _, _ = infer_sample_dims(sample_dir)
    if layer_ids == [-1]:
        layer_ids = list(range(num_layers))

    sample_out = out_dir
    sample_out.mkdir(parents=True, exist_ok=True)
    cache_name = binary_cache_name(k_percent, threshold)

    summary: Dict = {
        "sample_dir": str(sample_dir),
        "k_percent": float(k_percent),
        "threshold": float(threshold),
        "num_heads": int(num_heads),
        "method": "Graph2Vec",
        "layers": {},
    }

    for layer in layer_ids:
        if layer < 0 or layer >= num_layers:
            continue

        print(f"[{sample_dir.name} layer {layer:02d}] Graph2Vec...", flush=True)
        binary = load_or_compute_binary(
            sample_dir,
            layer,
            cache_dir=cache_dir,
            k_percent=k_percent,
            threshold=threshold,
        )

        result = graph2vec_head_clustering(
            binary,
            k=k_clusters,
            ego_radius=ego_radius,
            vector_size=vector_size,
            wl_iterations=wl_iterations,
            epochs=epochs,
            random_state=random_state,
        )

        ad, directed_edge_count, binary_density = directed_graph_stats(binary)

        graph2vec_labels = np.array(result["graph2vec_labels"], dtype=int)
        row_labels = np.array(result["binary_row_labels"], dtype=int)
        row_col_labels = np.array(result["binary_row_col_labels"], dtype=int)
        embeddings = result["graph2vec_embeddings"]

        layer_out = sample_out / f"layer_{layer:02d}"
        layer_out.mkdir(parents=True, exist_ok=True)
        np.save(layer_out / cache_name, binary)
        embedding_path = layer_out / "graph2vec_embeddings.npy"
        np.save(embedding_path, embeddings)

        layer_summary = {
            "graph2vec_labels": result["graph2vec_labels"],
            "binary_row_labels": result["binary_row_labels"],
            "binary_row_col_labels": result["binary_row_col_labels"],
            "graph2vec_embeddings_path": str(embedding_path.name),
            "graph2vec_embedding_shape": list(embeddings.shape),
            "stats": {
                "Graph2Vec": partition_edge_stats(ad, graph2vec_labels),
                "BinaryRow-KMeans": partition_edge_stats(ad, row_labels),
                "BinaryRowCol-KMeans": partition_edge_stats(ad, row_col_labels),
            },
            "binary_density": binary_density,
            "directed_edge_count": directed_edge_count,
            "meta": result["meta"],
        }
        (layer_out / "graph2vec_clustering.json").write_text(
            json.dumps(layer_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        summary["layers"][f"layer_{layer:02d}"] = {
            "graph2vec_labels": result["graph2vec_labels"],
            "binary_row_labels": result["binary_row_labels"],
            "binary_row_col_labels": result["binary_row_col_labels"],
            "stats": layer_summary["stats"],
            "binary_density": binary_density,
            "directed_edge_count": directed_edge_count,
        }

        print(
            f"[Layer {layer:02d}] edges={directed_edge_count}, density={binary_density:.4f}",
            flush=True,
        )
        print(f"[Layer {layer:02d}] Graph2Vec labels: {result['graph2vec_labels']}", flush=True)
        print(f"[Layer {layer:02d}] BinaryRow labels: {result['binary_row_labels']}", flush=True)
        print(
            f"[Layer {layer:02d}] BinaryRowCol labels: {result['binary_row_col_labels']}",
            flush=True,
        )
        print(
            f"  Graph2Vec within_ratio={layer_summary['stats']['Graph2Vec']['within_ratio']:.3f}  "
            f"BinaryRow within_ratio={layer_summary['stats']['BinaryRow-KMeans']['within_ratio']:.3f}  "
            f"BinaryRowCol within_ratio={layer_summary['stats']['BinaryRowCol-KMeans']['within_ratio']:.3f}",
            flush=True,
        )

    (sample_out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Graph2Vec head clustering per layer")
    p.add_argument(
        "--root_dir",
        type=str,
        default=str(WORK_ROOT / "attention_map/outputs_longbench_v2_7b_32k"),
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=str(WORK_ROOT / "attention_map/analysis/layer_head_graph2vec_7b_32k"),
    )
    p.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Optional dir with precomputed layer_XX/similarity_binary_k*_thr*.npy per sample slug",
    )
    p.add_argument("--k_percent", type=float, default=95.0)
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--layers", type=str, default="all")
    p.add_argument("--k_clusters", type=int, default=2)
    p.add_argument("--ego_radius", type=int, default=2)
    p.add_argument("--vector_size", type=int, default=32)
    p.add_argument("--wl_iterations", type=int, default=2)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def parse_layer_ids(layers_arg: str, num_layers: int | None = None) -> List[int]:
    if layers_arg == "all":
        return list(range(num_layers)) if num_layers else [-1]
    return [int(x.strip()) for x in layers_arg.split(",") if x.strip()]


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir)
    out_dir = Path(args.out_dir)
    cache_root = Path(args.cache_dir) if args.cache_dir else None
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_dirs = discover_sample_dirs(root_dir)
    if not sample_dirs:
        raise FileNotFoundError(f"No samples under {root_dir}")

    batch_summary: Dict = {
        "root_dir": str(root_dir),
        "method": "Graph2Vec",
        "k_percent": args.k_percent,
        "threshold": args.threshold,
        "samples": {},
    }

    for sample_dir in sample_dirs:
        slug = sample_slug(sample_dir, root_dir)
        cache_dir = cache_root / slug if cache_root else None
        num_layers, _, _, _ = infer_sample_dims(sample_dir)
        layer_ids = parse_layer_ids(args.layers, num_layers)

        print(f"\n=== {slug} ===", flush=True)
        summary = process_sample(
            sample_dir,
            out_dir / slug,
            layer_ids=layer_ids,
            k_percent=args.k_percent,
            threshold=args.threshold,
            cache_dir=cache_dir if cache_dir and cache_dir.is_dir() else None,
            k_clusters=args.k_clusters,
            ego_radius=args.ego_radius,
            vector_size=args.vector_size,
            wl_iterations=args.wl_iterations,
            epochs=args.epochs,
            random_state=args.seed,
        )
        batch_summary["samples"][slug] = {
            "sample_dir": str(sample_dir),
            "layer_count": len(summary["layers"]),
        }

    (out_dir / "batch_summary.json").write_text(
        json.dumps(batch_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nDone. Outputs: {out_dir}")


if __name__ == "__main__":
    main()
