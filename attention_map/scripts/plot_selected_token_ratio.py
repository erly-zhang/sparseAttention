#!/usr/bin/env python3
"""Analyze the selected-token ratio under the K% attention-mass rule.

For each head we already know how many tokens are needed to cover K% (e.g. 95%)
of the attention mass on the (last) query row. This script reports and
visualizes that count as a *fraction of the total tokens* the query can attend
to, i.e. selected_tokens / (q + 1). When only the last query row is used,
q + 1 == seq_len, so the ratio answers: "to keep 95% of attention score, what
fraction of all tokens must be selected?"

Input: per_head_sparsity.json produced by run_ab_similarity_experiment.py
(each head already carries `avg_selected_ratio`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot selected-token ratio under K% mass rule")
    p.add_argument(
        "--json_path",
        type=str,
        default=(
            "/home/ubuntu/work/attention_map/analysis/"
            "synthetic_ab_experiment_k95/per_head_sparsity.json"
        ),
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory (default: same dir as JSON)",
    )
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args()


def load_ratio_grid(payload: Dict, sample_key: str) -> Tuple[np.ndarray, int, int]:
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
        grid[r["layer"], r["head"]] = r["avg_selected_ratio"]
    return grid, num_layers, num_heads


def ratio_stats(grid: np.ndarray) -> Dict[str, float]:
    flat = grid.reshape(-1)
    return {
        "mean_ratio_over_heads": float(flat.mean()),
        "median_ratio_over_heads": float(np.median(flat)),
        "std_ratio_over_heads": float(flat.std()),
        "min_ratio_over_heads": float(flat.min()),
        "max_ratio_over_heads": float(flat.max()),
        "p90_ratio_over_heads": float(np.percentile(flat, 90)),
    }


def plot_ratio_layer_head(
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
    im = ax.imshow(
        grid,
        aspect="auto",
        cmap="YlGnBu",
        vmin=0,
        vmax=float(grid.max()) if grid.size else 1.0,
        interpolation="nearest",
    )
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.set_title(title, fontsize=11)
    ax.set_xticks(np.arange(num_heads))
    ax.set_yticks(np.arange(0, num_layers, max(1, num_layers // 18)))
    plt.colorbar(im, ax=ax, label="selected-token ratio", fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def plot_ratio_histogram(
    a: np.ndarray,
    b: np.ndarray,
    out_png: Path,
    *,
    k_percent,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0.0, max(a.max(), b.max()), 40)
    ax.hist(a.reshape(-1), bins=bins, alpha=0.55, label="A", color="#2166ac", density=True)
    ax.hist(b.reshape(-1), bins=bins, alpha=0.55, label="B", color="#b2182b", density=True)
    ax.axvline(float(a.mean()), color="#2166ac", linestyle="--", linewidth=1.2)
    ax.axvline(float(b.mean()), color="#b2182b", linestyle="--", linewidth=1.2)
    ax.set_xlabel(f"selected-token ratio (fraction of total tokens to keep {k_percent}% mass)")
    ax.set_ylabel("density")
    ax.set_title(f"Per-head selected-token ratio (K={k_percent}%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def plot_ratio_layer_mean(
    grid_a: np.ndarray,
    grid_b: np.ndarray,
    out_png: Path,
    *,
    dpi: int,
) -> None:
    layers = np.arange(grid_a.shape[0])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(layers, grid_a.mean(axis=1), label="A", color="#2166ac", linewidth=1.5)
    ax.plot(layers, grid_b.mean(axis=1), label="B", color="#b2182b", linewidth=1.5)
    ax.set_xlabel("layer")
    ax.set_ylabel("mean selected-token ratio (over heads)")
    ax.set_title("Layer-wise mean selected-token ratio")
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
    seq_len = meta.get("seq_len")

    grid_a, num_layers, num_heads = load_ratio_grid(payload, "A")
    grid_b, _, _ = load_ratio_grid(payload, "B")

    prefix = out_dir / "selected_token_ratio"
    plot_ratio_layer_head(
        grid_a,
        Path(f"{prefix}_A_layer_head.png"),
        title=f"Sample A: selected-token ratio (K={k}%, q≥{q0})",
        dpi=args.dpi,
    )
    plot_ratio_layer_head(
        grid_b,
        Path(f"{prefix}_B_layer_head.png"),
        title=f"Sample B: selected-token ratio (K={k}%, q≥{q0})",
        dpi=args.dpi,
    )
    plot_ratio_histogram(
        grid_a,
        grid_b,
        Path(f"{prefix}_histogram.png"),
        k_percent=k,
        dpi=args.dpi,
    )
    plot_ratio_layer_mean(
        grid_a,
        grid_b,
        Path(f"{prefix}_layer_mean.png"),
        dpi=args.dpi,
    )

    stats = {
        "meta": {
            "k_percent": k,
            "q_start": q0,
            "seq_len": seq_len,
            "num_heads_total": int(num_layers * num_heads),
            "note": (
                "selected-token ratio per head = selected_tokens / (q + 1). "
                "It answers: to retain K% of attention mass, what fraction of all "
                "tokens must be kept? mean_ratio_over_heads averages this across heads."
            ),
        },
        "A": ratio_stats(grid_a),
        "B": ratio_stats(grid_b),
    }
    out_json = Path(f"{prefix}_summary.json")
    out_json.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Wrote selected-token ratio analysis under {out_dir}:")
    for name in [
        "selected_token_ratio_A_layer_head.png",
        "selected_token_ratio_B_layer_head.png",
        "selected_token_ratio_histogram.png",
        "selected_token_ratio_layer_mean.png",
        "selected_token_ratio_summary.json",
    ]:
        print(f"  - {name}")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
