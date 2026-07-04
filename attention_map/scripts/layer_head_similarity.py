"""Shared utilities for per-layer head similarity (aligned with run_ab_similarity_experiment)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np


def keys_for_k_percent_mass(row: np.ndarray, q: int, k_percent: float) -> np.ndarray:
    """K%% mass coverage on causal row q (same rule as run_ab_similarity_experiment)."""
    causal = row[: q + 1]
    total = float(causal.sum())
    if total <= 0:
        return np.array([], dtype=np.int32)

    target = (k_percent / 100.0) * total
    order = np.argsort(-causal)
    chosen: List[int] = []
    cum = 0.0
    for k in order:
        cum += float(causal[k])
        chosen.append(int(k))
        if cum >= target:
            break
    return np.asarray(chosen, dtype=np.int32)


def infer_sample_dims(sample_dir: Path) -> Tuple[int, int, int, bool]:
    layers = sorted(sample_dir.glob("layer_*"))
    if not layers:
        raise FileNotFoundError(f"No layer_* under {sample_dir}")
    num_layers = len(layers)
    heads = sorted(layers[0].glob("head_*.npy"))
    if not heads:
        raise FileNotFoundError(f"No head_*.npy under {layers[0]}")
    num_heads = len(heads)

    raw = np.load(heads[0], mmap_mode="r")
    if raw.ndim == 2 and raw.shape[0] == raw.shape[1] and raw.shape[0] > 1:
        seq_len, is_last_row = int(raw.shape[0]), False
    elif raw.ndim == 2 and raw.shape[0] == 1:
        seq_len, is_last_row = int(raw.shape[1]), True
    elif raw.ndim == 1:
        seq_len, is_last_row = int(raw.shape[0]), True
    else:
        raise ValueError(f"Unexpected head shape {raw.shape} at {heads[0]}")
    return num_layers, num_heads, seq_len, is_last_row


def load_head_row(
    sample_dir: Path,
    layer: int,
    head: int,
    q: int,
    *,
    is_last_row: bool,
) -> np.ndarray:
    path = sample_dir / f"layer_{layer:02d}" / f"head_{head:02d}.npy"
    attn = np.load(path, mmap_mode="r")
    if is_last_row:
        return np.asarray(attn, dtype=np.float32).reshape(-1)
    return np.asarray(attn[q], dtype=np.float32)


def compute_layer_directional_similarity(
    sample_dir: Path,
    layer: int,
    *,
    num_heads: int,
    seq_len: int,
    q: int,
    k_percent: float,
    is_last_row: bool,
) -> np.ndarray:
    """S[i,j] = coverage of head j on keys selected from head i (last query row)."""
    sim = np.zeros((num_heads, num_heads), dtype=np.float32)
    selected_keys: List[np.ndarray] = []

    for head_i in range(num_heads):
        row_i = load_head_row(sample_dir, layer, head_i, q, is_last_row=is_last_row)
        selected_keys.append(keys_for_k_percent_mass(row_i, q, k_percent))

    for head_i in range(num_heads):
        keys = selected_keys[head_i]
        if keys.size == 0:
            continue
        for head_j in range(num_heads):
            row_j = load_head_row(sample_dir, layer, head_j, q, is_last_row=is_last_row)
            total = float(row_j[: q + 1].sum())
            if total <= 0:
                continue
            sim[head_i, head_j] = float(row_j[keys].sum()) / total
    return sim


def binarize_similarity(sim: np.ndarray, threshold: float) -> np.ndarray:
    """1 if entry >= threshold (same as run_ab_similarity_experiment)."""
    return (sim >= threshold).astype(np.uint8)


def discover_sample_dirs(root_dir: Path) -> List[Path]:
    samples: List[Path] = []
    for layer_dir in sorted(root_dir.rglob("layer_*")):
        if not layer_dir.is_dir():
            continue
        sample_dir = layer_dir.parent
        if sample_dir in samples:
            continue
        if list(layer_dir.glob("head_*.npy")):
            samples.append(sample_dir)
    return sorted(samples)


def sample_slug(sample_dir: Path, root_dir: Path | None = None) -> str:
    if root_dir is not None:
        try:
            return sample_dir.relative_to(root_dir).as_posix().replace("/", "__")
        except ValueError:
            pass
    return sample_dir.name


def partition_edge_stats(ad: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Summarize within/between edge counts for a partition."""
    ad = ad.astype(bool)
    labels = np.asarray(labels)
    n = ad.shape[0]
    within = between = 0
    for i in range(n):
        for j in range(n):
            if not ad[i, j]:
                continue
            if labels[i] == labels[j]:
                within += 1
            else:
                between += 1
    total = within + between
    return {
        "within_edges": float(within),
        "between_edges": float(between),
        "total_directed_edges": float(total),
        "within_ratio": float(within / total) if total else 0.0,
        "cluster1_size": float(np.sum(labels == 1)),
        "cluster2_size": float(np.sum(labels == 2)),
    }
