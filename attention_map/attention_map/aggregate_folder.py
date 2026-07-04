#!/usr/bin/env python3
"""Sum all attention .npy files in a folder and plot one heatmap."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, PowerNorm

CMAP_NAME = "Greys"


def get_display_cmap():
    cmap = plt.get_cmap(CMAP_NAME).copy()
    cmap.set_bad(color="white")
    return cmap


def load_square(path: Path) -> np.ndarray:
    a = np.squeeze(np.load(path))
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"Expected square [L,L], got {a.shape} at {path}")
    return a.astype(np.float32)


def apply_causal_mask(A: np.ndarray) -> np.ma.MaskedArray:
    n = A.shape[0]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    return np.ma.array(A, mask=mask)


def dynamic_color_limits(
    data: np.ma.MaskedArray,
    *,
    vmin_percentile: float = 1.0,
    vmax_percentile: float = 90.0,
) -> tuple[float, float, dict]:
    """
    按有效像素（下三角）的分位数动态设定 vmin/vmax。
    聚合 attention 往往只有少数位置接近 1，用 p90~p95 作 vmax 可显著增强对比度。
    """
    valid = data.compressed().astype(np.float64)
    valid = valid[valid > 0]  # 忽略严格 0，避免 log 尺度异常
    if valid.size == 0:
        return 0.0, 1.0, {}

    vmin = float(np.percentile(valid, vmin_percentile))
    vmax = float(np.percentile(valid, vmax_percentile))
    if vmax <= vmin:
        vmax = float(np.max(valid))
        vmin = float(np.min(valid))
    if vmax <= vmin:
        vmax = vmin + 1e-8

    stats = {
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "mean": float(np.mean(valid)),
        "p50": float(np.percentile(valid, 50)),
        "p90": float(np.percentile(valid, 90)),
        "p95": float(np.percentile(valid, 95)),
        "p99": float(np.percentile(valid, 99)),
        "vmin_used": vmin,
        "vmax_used": vmax,
        "vmin_percentile": vmin_percentile,
        "vmax_percentile": vmax_percentile,
    }
    return vmin, vmax, stats


def choose_norm(vmin: float, vmax: float, norm: str):
    """根据数据跨度选择线性 / 对数 / 幂次归一化。"""
    if norm == "linear":
        return None
    span = vmax / max(vmin, 1e-12)
    if norm == "log" or (norm == "auto" and span > 50):
        return LogNorm(vmin=max(vmin, 1e-8), vmax=vmax)
    if norm == "power" or (norm == "auto" and span > 10):
        return PowerNorm(gamma=0.45, vmin=vmin, vmax=vmax)
    return None


def aggregate_folder(
    folder: Path,
    out_png: Path,
    *,
    dpi: int = 150,
    figsize: tuple[float, float] = (10, 10),
    vmin_percentile: float = 1.0,
    vmax_percentile: float = 90.0,
    norm: str = "auto",
) -> dict:
    files = sorted(folder.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files in {folder}")

    total = None
    L = None
    for f in files:
        a = load_square(f)
        if total is None:
            L = a.shape[0]
            total = np.zeros((L, L), dtype=np.float64)
        elif a.shape[0] != L:
            raise ValueError(f"Shape mismatch: {f} has {a.shape}, expected ({L},{L})")
        total += a

    assert total is not None
    count = len(files)
    summed = total.astype(np.float32)

    data = apply_causal_mask(summed)
    vmin, vmax, color_stats = dynamic_color_limits(
        data,
        vmin_percentile=vmin_percentile,
        vmax_percentile=vmax_percentile,
    )
    mnorm = choose_norm(vmin, vmax, norm)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        data,
        aspect="equal",
        cmap=get_display_cmap(),
        norm=mnorm,
        vmin=None if mnorm is not None else vmin,
        vmax=None if mnorm is not None else vmax,
        interpolation="nearest",
    )
    ax.set_xlabel("Key position (token index)")
    ax.set_ylabel("Query position (token index)")
    norm_label = norm if norm != "auto" else (
        "log" if isinstance(mnorm, LogNorm) else ("power" if isinstance(mnorm, PowerNorm) else "linear")
    )
    ax.set_title(
        f"Sum of {count} attention maps — {folder.name}\n"
        f"color scale: {norm_label}, p{vmin_percentile:g}–p{vmax_percentile:g} "
        f"→ [{vmin:.2e}, {vmax:.2e}] (darker = higher)",
        fontsize=11,
    )
    plt.colorbar(im, ax=ax, label="summed attention weight")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)

    return {
        "folder": str(folder),
        "num_maps": count,
        "seq_len": int(L),
        "sum_max": float(summed.max()),
        "sum_min": float(summed.min()),
        "color_stats": color_stats,
        "out_png": str(out_png),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sum attention maps in a folder and plot heatmap")
    p.add_argument(
        "--folder",
        type=str,
        default="/home/ubuntu/work/attention_map/outputs/4k/pattern_sorted/StreamLLM(A-shape)",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output png path. Default: <folder>/aggregated_sum_heatmap.png",
    )
    p.add_argument(
        "--vmax_percentile",
        type=float,
        default=90.0,
        help="色标上限分位数（越小对比度越强，默认 p90）",
    )
    p.add_argument(
        "--vmin_percentile",
        type=float,
        default=1.0,
        help="色标下限分位数（默认 p1）",
    )
    p.add_argument(
        "--norm",
        type=str,
        choices=["auto", "linear", "log", "power"],
        default="auto",
        help="颜色归一化：auto 按数据跨度自动选 log/power/linear",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    folder = Path(args.folder)
    out_png = Path(args.out) if args.out else (folder / "aggregated_sum_heatmap.png")
    info = aggregate_folder(
        folder,
        out_png,
        vmin_percentile=args.vmin_percentile,
        vmax_percentile=args.vmax_percentile,
        norm=args.norm,
    )
    print(info)


if __name__ == "__main__":
    main()
