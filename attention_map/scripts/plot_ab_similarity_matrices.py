#!/usr/bin/env python3
"""Plot directional/binary similarity matrices from saved .npy files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from attention_map.similarity import plot_similarity_heatmap


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--analysis_dir",
        default="/home/ubuntu/work/attention_map/analysis/synthetic_ab_experiment_k95",
    )
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args()


def plot_binary(matrix: np.ndarray, out_png: Path, *, title: str, num_heads: int, dpi: int) -> None:
    plot_similarity_heatmap(
        matrix.astype(np.float32),
        out_png,
        title=title,
        cmap="gray_r",
        num_heads=num_heads,
        dpi=dpi,
    )


def main() -> None:
    args = parse_args()
    analysis_dir = Path(args.analysis_dir)
    summary_path = analysis_dir / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        num_heads = int(summary["num_heads"])
        k_percent = float(summary["k_percent"])
        threshold = float(summary["threshold"])
        q0 = int(summary["q_start"])
    else:
        num_heads = 16
        k_percent = 95.0
        threshold = 0.9
        q0 = 1024

    specs = [
        ("M_AA_directional.npy", f"A-A directional similarity, K={k_percent}%, q>={q0}", "viridis", False),
        ("M_AB_directional.npy", f"A-B directional similarity, K={k_percent}%, q>={q0}", "viridis", False),
        ("M_BB_directional.npy", f"B-B directional similarity, K={k_percent}%, q>={q0}", "viridis", False),
        ("M_AA_binary.npy", f"A-A binary mask >= {threshold}", "gray_r", True),
        ("M_AB_binary.npy", f"A-B binary mask >= {threshold}", "gray_r", True),
        ("M_BB_binary.npy", f"B-B binary mask >= {threshold}", "gray_r", True),
    ]

    for npy_name, title, cmap, is_binary in specs:
        npy_path = analysis_dir / npy_name
        if not npy_path.is_file():
            print(f"Skip missing: {npy_path}")
            continue
        matrix = np.load(npy_path)
        out_png = npy_path.with_suffix(".png")
        if is_binary:
            plot_binary(matrix, out_png, title=title, num_heads=num_heads, dpi=args.dpi)
        else:
            plot_similarity_heatmap(
                matrix,
                out_png,
                title=title,
                cmap=cmap,
                num_heads=num_heads,
                dpi=args.dpi,
            )
        print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
