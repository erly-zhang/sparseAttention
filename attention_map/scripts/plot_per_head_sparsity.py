#!/usr/bin/env python3
"""Visualize total_selected_tokens from per_head_sparsity.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot total_selected_tokens sparsity stats")
    p.add_argument(
        "--json_path",
        type=str,
        default="/home/ubuntu/work/attention_map/analysis/ab_similarity_experiment_k95/per_head_sparsity.json",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory (default: same dir as JSON)",
    )
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args()


def load_totals(
    payload: Dict,
    sample_key: str,
) -> Tuple[np.ndarray, int, int]:
    rows: List[Dict] = payload[sample_key]
    n_heads_total = len(rows)
    num_layers = max(r["layer"] for r in rows) + 1
    num_heads = max(r["head"] for r in rows) + 1
    if num_layers * num_heads != n_heads_total:
        raise ValueError(
            f"{sample_key}: expected {num_layers}*{num_heads}={num_layers * num_heads} "
            f"heads, got {n_heads_total}"
        )
    grid = np.zeros((num_layers, num_heads), dtype=np.float64)
    for r in rows:
        grid[r["layer"], r["head"]] = r["total_selected_tokens"]
    return grid, num_layers, num_heads


def plot_layer_head_heatmap(
    grid: np.ndarray,
    out_png: Path,
    *,
    title: str,
    dpi: int,
) -> None:
    num_layers, num_heads = grid.shape
    fig_h = max(6.0, min(14.0, num_layers * 0.28))
    fig_w = max(5.0, min(10.0, num_heads * 0.45))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    vmax = float(grid.max()) if grid.size else 1.0
    im = ax.imshow(
        grid,
        aspect="auto",
        cmap="YlOrRd",
        vmin=0,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.set_title(title, fontsize=11)
    ax.set_xticks(np.arange(num_heads))
    ax.set_yticks(np.arange(0, num_layers, max(1, num_layers // 18)))
    plt.colorbar(im, ax=ax, label="total_selected_tokens", fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def plot_global_layer_major(
    grid: np.ndarray,
    out_png: Path,
    *,
    title: str,
    dpi: int,
) -> None:
    """One row heatmap: layer-major flat order (like similarity matrices)."""
    flat = grid.reshape(-1)
    n = flat.size
    side = int(np.ceil(np.sqrt(n)))
    pad = side * side - n
    if pad:
        flat = np.concatenate([flat, np.full(pad, np.nan)])
    mat = flat.reshape(side, side)

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(mat, aspect="equal", cmap="YlOrRd", interpolation="nearest")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("column (layer-major index)")
    ax.set_ylabel("row (layer-major index)")
    num_heads = grid.shape[1]
    if n > num_heads:
        ticks = np.arange(0, n + 1, num_heads)
        ticks = ticks[ticks <= n]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([str(t // num_heads) for t in ticks], fontsize=7)
        ax.set_yticklabels([str(t // num_heads) for t in ticks], fontsize=7)
    plt.colorbar(im, ax=ax, label="total_selected_tokens", fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def plot_histogram_compare(
    a: np.ndarray,
    b: np.ndarray,
    out_png: Path,
    *,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(
        min(a.min(), b.min()),
        max(a.max(), b.max()),
        40,
    )
    ax.hist(a.reshape(-1), bins=bins, alpha=0.55, label="A", color="#2166ac", density=True)
    ax.hist(b.reshape(-1), bins=bins, alpha=0.55, label="B", color="#b2182b", density=True)
    ax.set_xlabel("total_selected_tokens per head")
    ax.set_ylabel("density")
    ax.set_title("Distribution of total_selected_tokens (per head)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def plot_scatter_ab(
    a: np.ndarray,
    b: np.ndarray,
    out_png: Path,
    *,
    dpi: int,
) -> None:
    x = a.reshape(-1)
    y = b.reshape(-1)
    lim = max(x.max(), y.max()) * 1.02
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(x, y, s=12, alpha=0.45, c="#404040", edgecolors="none")
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, alpha=0.6, label="y=x")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("A: total_selected_tokens")
    ax.set_ylabel("B: total_selected_tokens")
    ax.set_title("Per-head total_selected_tokens: A vs B")
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def plot_diff_heatmap(
    grid_a: np.ndarray,
    grid_b: np.ndarray,
    out_png: Path,
    *,
    title: str,
    dpi: int,
) -> None:
    diff = grid_b.astype(np.float64) - grid_a.astype(np.float64)
    vmax = float(np.max(np.abs(diff))) or 1.0
    num_layers, num_heads = diff.shape
    fig_h = max(6.0, min(14.0, num_layers * 0.28))
    fig_w = max(5.0, min(10.0, num_heads * 0.45))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(
        diff,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.set_title(title, fontsize=11)
    ax.set_xticks(np.arange(num_heads))
    ax.set_yticks(np.arange(0, num_layers, max(1, num_layers // 18)))
    plt.colorbar(im, ax=ax, label="B − A", fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def plot_layer_mean(
    grid_a: np.ndarray,
    grid_b: np.ndarray,
    out_png: Path,
    *,
    dpi: int,
) -> None:
    layers = np.arange(grid_a.shape[0])
    mean_a = grid_a.mean(axis=1)
    mean_b = grid_b.mean(axis=1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(layers, mean_a, label="A", color="#2166ac", linewidth=1.5)
    ax.plot(layers, mean_b, label="B", color="#b2182b", linewidth=1.5)
    ax.set_xlabel("layer")
    ax.set_ylabel("mean total_selected_tokens (over heads)")
    ax.set_title("Layer-wise mean total_selected_tokens")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    json_path = Path(args.json_path)
    out_dir = Path(args.out_dir) if args.out_dir else json_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    meta = payload.get("meta", {})
    k = meta.get("k_percent", "?")
    q0 = meta.get("q_start", "?")

    grid_a, num_layers, num_heads = load_totals(payload, "A")
    grid_b, _, _ = load_totals(payload, "B")

    prefix = out_dir / "total_selected_tokens"
    plot_layer_head_heatmap(
        grid_a,
        Path(f"{prefix}_A_layer_head.png"),
        title=f"Sample A: total_selected_tokens (K={k}%, q≥{q0})",
        dpi=args.dpi,
    )
    plot_layer_head_heatmap(
        grid_b,
        Path(f"{prefix}_B_layer_head.png"),
        title=f"Sample B: total_selected_tokens (K={k}%, q≥{q0})",
        dpi=args.dpi,
    )
    plot_global_layer_major(
        grid_a,
        Path(f"{prefix}_A_global_flat.png"),
        title=f"Sample A: total_selected_tokens (layer-major layout)",
        dpi=args.dpi,
    )
    plot_global_layer_major(
        grid_b,
        Path(f"{prefix}_B_global_flat.png"),
        title=f"Sample B: total_selected_tokens (layer-major layout)",
        dpi=args.dpi,
    )
    plot_histogram_compare(
        grid_a,
        grid_b,
        Path(f"{prefix}_histogram.png"),
        dpi=args.dpi,
    )
    plot_scatter_ab(
        grid_a,
        grid_b,
        Path(f"{prefix}_scatter_AB.png"),
        dpi=args.dpi,
    )
    plot_diff_heatmap(
        grid_a,
        grid_b,
        Path(f"{prefix}_diff_B_minus_A.png"),
        title="B − A: total_selected_tokens per (layer, head)",
        dpi=args.dpi,
    )
    plot_layer_mean(
        grid_a,
        grid_b,
        Path(f"{prefix}_layer_mean.png"),
        dpi=args.dpi,
    )

    summary = payload.get("summary", {})
    print(f"Wrote PNGs under {out_dir}:")
    for name in [
        "total_selected_tokens_A_layer_head.png",
        "total_selected_tokens_B_layer_head.png",
        "total_selected_tokens_A_global_flat.png",
        "total_selected_tokens_B_global_flat.png",
        "total_selected_tokens_histogram.png",
        "total_selected_tokens_scatter_AB.png",
        "total_selected_tokens_diff_B_minus_A.png",
        "total_selected_tokens_layer_mean.png",
    ]:
        print(f"  - {name}")
    if summary:
        print("Summary:", json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
