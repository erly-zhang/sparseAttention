#!/usr/bin/env python3
"""
Aggregate attention .npy files and plot heatmaps.

Main uses:
  1. Single folder mean/sum heatmap:
     python aggregate_folder_modified.py --folder /path/to/StreamLLM --mode mean

  2. Multiple folders with shared color scale:
     python aggregate_folder_modified.py \
       --folders /path/to/StreamLLM /path/to/VerticalSlash /path/to/Blockwise \
       --out /path/to/mean_shared_scale.png \
       --mode mean --shared_scale --norm log

  3. Difference from global mean for multiple folders:
     python aggregate_folder_modified.py \
       --folders /path/to/StreamLLM /path/to/VerticalSlash /path/to/Blockwise \
       --out /path/to/diff_from_global.png \
       --mode mean --diff_global

Input:
  Each folder contains many square [L,L] .npy attention maps.

Important changes from earlier version:
  - Default aggregation is mean, not sum.
  - Multiple folders can be plotted in one figure with a shared color scale.
  - Optional diff-from-global visualization highlights class-specific differences.
  - Upper triangle is masked by default for causal attention.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, PowerNorm, TwoSlopeNorm


DEFAULT_CMAP = "Greys"
DIFF_CMAP = "RdBu_r"


# -----------------------------
# Loading / aggregation
# -----------------------------

def load_square(path: Path) -> np.ndarray:
    arr = np.squeeze(np.load(path))
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Expected square [L,L], got {arr.shape} at {path}")
    return arr.astype(np.float32, copy=False)


def causal_masked(A: np.ndarray, *, mask_upper: bool = True) -> np.ma.MaskedArray:
    if not mask_upper:
        return np.ma.array(A)
    n = A.shape[0]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    return np.ma.array(A, mask=mask)


def get_cmap(name: str) -> plt.Colormap:
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad(color="white")
    return cmap


def aggregate_folder(folder: Path, *, mode: str = "mean", pattern: str = "*.npy") -> Tuple[np.ndarray, Dict[str, object]]:
    files = sorted(folder.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} in {folder}")

    total: Optional[np.ndarray] = None
    L: Optional[int] = None
    for path in files:
        A = load_square(path)
        if total is None:
            L = int(A.shape[0])
            total = np.zeros((L, L), dtype=np.float64)
        elif A.shape[0] != L:
            raise ValueError(f"Shape mismatch: {path} has {A.shape}, expected ({L},{L})")
        total += A

    assert total is not None and L is not None
    count = len(files)
    if mode == "mean":
        agg = (total / count).astype(np.float32)
    elif mode == "sum":
        agg = total.astype(np.float32)
    else:
        raise ValueError("mode must be 'mean' or 'sum'")

    info = {
        "folder": str(folder),
        "name": folder.name,
        "num_maps": int(count),
        "seq_len": int(L),
        "mode": mode,
        "min": float(np.min(agg)),
        "max": float(np.max(agg)),
        "mean": float(np.mean(agg)),
    }
    return agg, info


# -----------------------------
# Color scale helpers
# -----------------------------

def positive_percentile_limits(
    arrays: Sequence[np.ma.MaskedArray],
    *,
    pmin: float,
    pmax: float,
) -> Tuple[float, float, Dict[str, float]]:
    vals = []
    for data in arrays:
        x = data.compressed().astype(np.float64)
        x = x[x > 0]
        if x.size:
            vals.append(x)
    if not vals:
        return 0.0, 1.0, {}

    all_vals = np.concatenate(vals)
    vmin = float(np.percentile(all_vals, pmin))
    vmax = float(np.percentile(all_vals, pmax))
    if vmax <= vmin:
        vmin = float(np.min(all_vals))
        vmax = float(np.max(all_vals))
    if vmax <= vmin:
        vmax = vmin + 1e-8

    stats = {
        "min_positive": float(np.min(all_vals)),
        "max": float(np.max(all_vals)),
        "mean_positive": float(np.mean(all_vals)),
        "p50": float(np.percentile(all_vals, 50)),
        "p90": float(np.percentile(all_vals, 90)),
        "p95": float(np.percentile(all_vals, 95)),
        "p99": float(np.percentile(all_vals, 99)),
        "vmin_used": vmin,
        "vmax_used": vmax,
        "vmin_percentile": float(pmin),
        "vmax_percentile": float(pmax),
    }
    return vmin, vmax, stats


def diff_symmetric_limit(arrays: Sequence[np.ma.MaskedArray], *, percentile: float) -> Tuple[float, Dict[str, float]]:
    vals = []
    for data in arrays:
        x = data.compressed().astype(np.float64)
        if x.size:
            vals.append(np.abs(x))
    if not vals:
        return 1.0, {}
    all_abs = np.concatenate(vals)
    vmax = float(np.percentile(all_abs, percentile))
    if vmax <= 0:
        vmax = float(np.max(all_abs))
    if vmax <= 0:
        vmax = 1e-8
    return vmax, {
        "abs_max": float(np.max(all_abs)),
        "abs_p90": float(np.percentile(all_abs, 90)),
        "abs_p95": float(np.percentile(all_abs, 95)),
        "abs_p99": float(np.percentile(all_abs, 99)),
        "abs_percentile_used": float(percentile),
        "vmax_used": vmax,
    }


def build_norm(norm: str, vmin: float, vmax: float):
    if norm == "linear":
        return None
    if norm == "log":
        return LogNorm(vmin=max(vmin, 1e-12), vmax=max(vmax, max(vmin, 1e-12) * 1.0001))
    if norm == "power":
        return PowerNorm(gamma=0.45, vmin=vmin, vmax=vmax)
    if norm == "auto":
        span = vmax / max(vmin, 1e-12)
        if span > 50:
            return LogNorm(vmin=max(vmin, 1e-12), vmax=vmax)
        if span > 10:
            return PowerNorm(gamma=0.45, vmin=vmin, vmax=vmax)
        return None
    raise ValueError(f"Unknown norm: {norm}")


def norm_name(norm_obj, requested: str) -> str:
    if requested != "auto":
        return requested
    if isinstance(norm_obj, LogNorm):
        return "log"
    if isinstance(norm_obj, PowerNorm):
        return "power"
    return "linear"


# -----------------------------
# Plotting
# -----------------------------

def plot_single(
    folder: Path,
    out_png: Path,
    *,
    mode: str,
    pattern: str,
    mask_upper: bool,
    cmap_name: str,
    norm: str,
    vmin_percentile: float,
    vmax_percentile: float,
    dpi: int,
    figsize: Tuple[float, float],
) -> Dict[str, object]:
    agg, info = aggregate_folder(folder, mode=mode, pattern=pattern)
    data = causal_masked(agg, mask_upper=mask_upper)
    vmin, vmax, color_stats = positive_percentile_limits([data], pmin=vmin_percentile, pmax=vmax_percentile)
    norm_obj = build_norm(norm, vmin, vmax)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        data,
        aspect="equal",
        cmap=get_cmap(cmap_name),
        norm=norm_obj,
        vmin=None if norm_obj is not None else vmin,
        vmax=None if norm_obj is not None else vmax,
        interpolation="nearest",
    )
    ax.set_xlabel("key")
    ax.set_ylabel("query")
    used_norm = norm_name(norm_obj, norm)
    ax.set_title(
        f"{folder.name}\n{mode} of {info['num_maps']} maps; "
        f"{used_norm}, p{vmin_percentile:g}-p{vmax_percentile:g}: [{vmin:.2e}, {vmax:.2e}]"
    )
    plt.colorbar(im, ax=ax, label=f"{mode} attention")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)

    info.update({"out_png": str(out_png), "color_stats": color_stats, "norm_used": used_norm})
    return info


def plot_multiple(
    folders: Sequence[Path],
    out_png: Path,
    *,
    mode: str,
    pattern: str,
    mask_upper: bool,
    cmap_name: str,
    norm: str,
    shared_scale: bool,
    diff_global: bool,
    vmin_percentile: float,
    vmax_percentile: float,
    diff_percentile: float,
    dpi: int,
    figsize_per_panel: Tuple[float, float],
) -> Dict[str, object]:
    aggs: List[np.ndarray] = []
    infos: List[Dict[str, object]] = []
    for folder in folders:
        agg, info = aggregate_folder(folder, mode=mode, pattern=pattern)
        aggs.append(agg)
        infos.append(info)

    shapes = {A.shape for A in aggs}
    if len(shapes) != 1:
        raise ValueError(f"All folders must have same attention shape, got {shapes}")

    if diff_global:
        counts = np.asarray([float(info["num_maps"]) for info in infos], dtype=np.float64)
        # Weighted global mean over all maps, not simple mean over classes.
        global_mean = sum(A.astype(np.float64) * c for A, c in zip(aggs, counts)) / float(counts.sum())
        plot_arrays = [(A.astype(np.float64) - global_mean).astype(np.float32) for A in aggs]
        masked = [causal_masked(A, mask_upper=mask_upper) for A in plot_arrays]
        vmax, color_stats = diff_symmetric_limit(masked, percentile=diff_percentile)
        norm_obj = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        cmap = get_cmap(DIFF_CMAP)
        colorbar_label = f"delta from global {mode}"
        title_prefix = f"Difference from global {mode}"
    else:
        plot_arrays = aggs
        masked = [causal_masked(A, mask_upper=mask_upper) for A in plot_arrays]
        if shared_scale:
            vmin, vmax, color_stats = positive_percentile_limits(masked, pmin=vmin_percentile, pmax=vmax_percentile)
            norm_obj = build_norm(norm, vmin, vmax)
            norms = [norm_obj] * len(masked)
            vmins = [None if norm_obj is not None else vmin] * len(masked)
            vmaxs = [None if norm_obj is not None else vmax] * len(masked)
        else:
            color_stats = {}
            norms, vmins, vmaxs = [], [], []
            for data in masked:
                vmin, vmax, _ = positive_percentile_limits([data], pmin=vmin_percentile, pmax=vmax_percentile)
                nobj = build_norm(norm, vmin, vmax)
                norms.append(nobj)
                vmins.append(None if nobj is not None else vmin)
                vmaxs.append(None if nobj is not None else vmax)
        cmap = get_cmap(cmap_name)
        colorbar_label = f"{mode} attention"
        if shared_scale:
            title_prefix = f"Shared scale {mode}"
        else:
            title_prefix = f"Per-panel scale {mode}"

    n = len(folders)
    fig_w = figsize_per_panel[0] * n
    fig_h = figsize_per_panel[1]
    fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h), squeeze=False)
    axes_list = axes[0]

    last_im = None
    if diff_global:
        for ax, folder, data, info in zip(axes_list, folders, masked, infos):
            last_im = ax.imshow(data, aspect="equal", cmap=cmap, norm=norm_obj, interpolation="nearest")
            ax.set_title(f"{folder.name}\n{info['num_maps']} maps")
            ax.set_xlabel("key")
            ax.set_ylabel("query")
    else:
        for idx, (ax, folder, data, info) in enumerate(zip(axes_list, folders, masked, infos)):
            if shared_scale:
                im_norm = norm_obj
                im_vmin = None if im_norm is not None else vmin
                im_vmax = None if im_norm is not None else vmax
            else:
                im_norm = norms[idx]
                im_vmin = vmins[idx]
                im_vmax = vmaxs[idx]
            last_im = ax.imshow(
                data,
                aspect="equal",
                cmap=cmap,
                norm=im_norm,
                vmin=im_vmin,
                vmax=im_vmax,
                interpolation="nearest",
            )
            ax.set_title(f"{folder.name}\n{mode} of {info['num_maps']} maps")
            ax.set_xlabel("key")
            ax.set_ylabel("query")

    if last_im is not None:
        fig.colorbar(last_im, ax=axes_list.tolist(), shrink=0.72, label=colorbar_label)

    if diff_global:
        fig.suptitle(f"{title_prefix}; symmetric p{diff_percentile:g}", y=0.98)
    else:
        used_norm = norm_name(norm_obj, norm) if shared_scale else norm
        if shared_scale:
            fig.suptitle(
                f"{title_prefix}; {used_norm}, p{vmin_percentile:g}-p{vmax_percentile:g}: "
                f"[{color_stats.get('vmin_used', 0):.2e}, {color_stats.get('vmax_used', 1):.2e}]",
                y=0.98,
            )
        else:
            fig.suptitle(title_prefix, y=0.98)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return {
        "folders": [str(x) for x in folders],
        "infos": infos,
        "mode": mode,
        "shared_scale": bool(shared_scale),
        "diff_global": bool(diff_global),
        "color_stats": color_stats,
        "out_png": str(out_png),
    }


# -----------------------------
# CLI
# -----------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Aggregate and visualize attention maps")
    p.add_argument("--folder", type=str, default=None, help="Single folder containing .npy maps")
    p.add_argument("--folders", type=str, nargs="*", default=None, help="Multiple folders to plot side-by-side")
    p.add_argument("--out", type=str, default=None, help="Output png path")
    p.add_argument("--mode", choices=["mean", "sum"], default="mean", help="Aggregate by mean or sum. Default: mean")
    p.add_argument("--pattern", type=str, default="*.npy", help="Glob pattern inside each folder. Default: *.npy")
    p.add_argument("--norm", choices=["auto", "linear", "log", "power"], default="auto", help="Color normalization")
    p.add_argument("--cmap", type=str, default=DEFAULT_CMAP, help="Colormap for non-difference plots")
    p.add_argument("--shared_scale", action="store_true", help="Use one color scale across multiple folders")
    p.add_argument("--diff_global", action="store_true", help="Plot each folder mean minus weighted global mean")
    p.add_argument("--vmin_percentile", type=float, default=1.0, help="Lower color percentile for non-diff plots")
    p.add_argument("--vmax_percentile", type=float, default=95.0, help="Upper color percentile for non-diff plots")
    p.add_argument("--diff_percentile", type=float, default=99.0, help="Symmetric abs percentile for diff plots")
    p.add_argument("--no_mask_upper", action="store_true", help="Do not mask upper triangle")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--figsize", type=str, default=None, help="Single figure size W,H. Default: 8,8")
    p.add_argument("--figsize_per_panel", type=str, default=None, help="Multiple plot panel size W,H. Default: 5,5")
    p.add_argument("--json_out", type=str, default=None, help="Optional JSON stats output")
    return p


def parse_size(value: Optional[str], default: Tuple[float, float]) -> Tuple[float, float]:
    if not value:
        return default
    if "," not in value:
        raise ValueError("Size must be formatted as W,H")
    w, h = value.split(",", 1)
    return float(w), float(h)


def main() -> None:
    args = build_argparser().parse_args()
    mask_upper = not args.no_mask_upper

    folders: List[Path] = []
    if args.folders:
        folders.extend(Path(x) for x in args.folders)
    if args.folder:
        folders.append(Path(args.folder))

    if not folders:
        raise SystemExit("Please provide --folder or --folders")

    if len(folders) == 1:
        folder = folders[0]
        out_png = Path(args.out) if args.out else folder / f"aggregated_{args.mode}_heatmap.png"
        info = plot_single(
            folder,
            out_png,
            mode=args.mode,
            pattern=args.pattern,
            mask_upper=mask_upper,
            cmap_name=args.cmap,
            norm=args.norm,
            vmin_percentile=args.vmin_percentile,
            vmax_percentile=args.vmax_percentile,
            dpi=args.dpi,
            figsize=parse_size(args.figsize, (8.0, 8.0)),
        )
    else:
        out_png = Path(args.out) if args.out else folders[0].parent / (
            "diff_from_global.png" if args.diff_global else f"aggregated_{args.mode}_shared.png"
        )
        info = plot_multiple(
            folders,
            out_png,
            mode=args.mode,
            pattern=args.pattern,
            mask_upper=mask_upper,
            cmap_name=args.cmap,
            norm=args.norm,
            shared_scale=args.shared_scale or args.diff_global,
            diff_global=args.diff_global,
            vmin_percentile=args.vmin_percentile,
            vmax_percentile=args.vmax_percentile,
            diff_percentile=args.diff_percentile,
            dpi=args.dpi,
            figsize_per_panel=parse_size(args.figsize_per_panel, (5.0, 5.0)),
        )

    if args.json_out:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
