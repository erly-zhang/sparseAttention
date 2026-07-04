#!/usr/bin/env python3
"""
Visualize one attention map per pattern_sorted subfolder (full resolution by default).

Typical use after sort_by_pattern:
  pattern_sorted/
    StreamLLM(A-shape)/layer_00_head_00.npy
    Vertical-Slash/...
    Block-wise/...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from attention_map.sort_by_pattern import PATTERN_TO_DIR
from attention_map.visualize import (
    apply_causal_mask_for_display,
    downsample_square,
    robust_vmax,
)

PATTERN_FOLDERS: List[str] = list(PATTERN_TO_DIR.values())

# 高对比 sequential：低=深蓝/青，高=黄（常见 heatmap 风格，比 Greys 更易分辨）
DEFAULT_CMAP = "viridis"


def get_pattern_cmap(name: str = DEFAULT_CMAP):
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad(color="white")
    return cmap


def _parse_layer_head(stem: str) -> Tuple[int, int]:
    # layer_05_head_10
    parts = stem.split("_")
    return int(parts[1]), int(parts[3])


def _best_from_classification(
    cls: dict, layer: int, head: int
) -> Tuple[Optional[str], Optional[float], dict]:
    key = f"{layer:02d}"
    h = str(head)
    results = cls.get("results", {})
    if key not in results or h not in results[key]:
        return None, None, {}
    best = results[key][h].get("best", {})
    return best.get("pattern"), best.get("score"), best.get("params", {})


def _resolve_pick(
    folder: Path,
    filename: Optional[str],
) -> Path:
    if filename:
        path = folder / filename
        if not path.exists():
            raise FileNotFoundError(f"Pick not found: {path}")
        return path
    files = sorted(folder.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files in {folder}")
    return files[0]


def _load_classification(path: Optional[Path]) -> dict:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def plot_single_map(
    attn: np.ndarray,
    *,
    title: str,
    out_png: Path,
    downsample_max: Optional[int],
    dpi: int,
    cmap_name: str = DEFAULT_CMAP,
) -> dict:
    attn = np.squeeze(attn).astype(np.float32)
    if attn.ndim != 2 or attn.shape[0] != attn.shape[1]:
        raise ValueError(f"Expected square [L,L], got {attn.shape}")

    L = int(attn.shape[0])
    display = attn
    if downsample_max is not None and downsample_max > 0 and L > downsample_max:
        display = downsample_square(attn, max_size=downsample_max)

    data = apply_causal_mask_for_display(display)
    vmax = robust_vmax(data)
    cmap = get_pattern_cmap(cmap_name)

    inches = max(display.shape[0] / dpi, 1.0)
    fig, ax = plt.subplots(figsize=(inches, inches), dpi=dpi)
    ax.imshow(
        data,
        aspect="equal",
        cmap=cmap,
        vmin=0,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("key position (token index)")
    ax.set_ylabel("query position (token index)")
    plt.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04, label="attention")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)

    return {
        "display_shape": list(display.shape),
        "seq_len": L,
        "downsample_max": downsample_max,
        "cmap": cmap_name,
        "out_png": str(out_png),
    }


def visualize_pattern_picks(
    pattern_sorted_dir: Path,
    out_dir: Path,
    *,
    picks: Optional[Dict[str, str]] = None,
    classification_json: Optional[Path] = None,
    downsample_max: Optional[int] = None,
    dpi: int = 150,
    combined: bool = True,
    cmap_name: str = DEFAULT_CMAP,
) -> List[dict]:
    """
    Plot one .npy per pattern folder.

    picks: optional map folder_name -> filename, e.g.
      {"StreamLLM(A-shape)": "layer_00_head_00.npy", ...}
    If omitted, uses the first sorted *.npy in each folder.
    """
    pattern_sorted_dir = Path(pattern_sorted_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cls = _load_classification(classification_json)
    picks = picks or {}
    summary: List[dict] = []
    panel_data = []

    for folder_name in PATTERN_FOLDERS:
        sub = pattern_sorted_dir / folder_name
        if not sub.is_dir():
            raise FileNotFoundError(f"Missing pattern folder: {sub}")

        npy_path = _resolve_pick(sub, picks.get(folder_name))
        attn = np.load(npy_path)
        layer, head = _parse_layer_head(npy_path.stem)
        pattern, score, params = _best_from_classification(cls, layer, head)

        title = f"{folder_name}\n{npy_path.name}\nL{layer} H{head}"
        if pattern:
            title += f"\n{pattern} score={score:.4f}"

        safe_name = folder_name.replace("/", "-")
        out_png = out_dir / f"{safe_name}_{npy_path.stem}.png"
        info = plot_single_map(
            attn,
            title=title,
            out_png=out_png,
            downsample_max=downsample_max,
            dpi=dpi,
            cmap_name=cmap_name,
        )
        info.update(
            {
                "category": folder_name,
                "file": str(npy_path),
                "layer": layer,
                "head": head,
                "best_pattern": pattern,
                "best_score": score,
                "best_params": params,
            }
        )
        summary.append(info)
        panel_data.append((folder_name, npy_path, layer, head, pattern, score, attn))

    if combined and panel_data:
        _save_combined(
            panel_data,
            out_dir / "three_patterns_combined.png",
            downsample_max,
            dpi,
            cmap_name,
        )

    selection_path = out_dir / "selection.json"
    selection_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def _save_combined(
    panel_data: list,
    out_png: Path,
    downsample_max: Optional[int],
    dpi: int,
    cmap_name: str = DEFAULT_CMAP,
) -> None:
    cmap = get_pattern_cmap(cmap_name)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, (folder_name, npy_path, layer, head, pattern, score, attn) in zip(
        axes, panel_data
    ):
        attn = np.squeeze(attn).astype(np.float32)
        L = attn.shape[0]
        display = attn
        if downsample_max is not None and downsample_max > 0 and L > downsample_max:
            display = downsample_square(attn, max_size=downsample_max)
        data = apply_causal_mask_for_display(display)
        vmax = robust_vmax(data)
        ax.imshow(
            data,
            aspect="equal",
            cmap=cmap,
            vmin=0,
            vmax=vmax,
            interpolation="nearest",
        )
        label = pattern or folder_name
        ax.set_title(
            f"{folder_name}\n{npy_path.name}\nL={L} {label} {score:.3f}",
            fontsize=8,
        )
        ax.set_xlabel("key")
        ax.set_ylabel("query")

    ds_note = "full resolution" if not downsample_max else f"downsample max={downsample_max}"
    fig.suptitle(
        f"One sample per pattern ({ds_note}, causal mask, cmap={cmap_name})",
        fontsize=11,
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot one full attention heatmap per pattern_sorted category"
    )
    p.add_argument(
        "--pattern_sorted_dir",
        type=str,
        required=True,
        help="Directory with StreamLLM(A-shape)/, Vertical-Slash/, Block-wise/",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Default: <pattern_sorted_dir>/viz_one_per_pattern",
    )
    p.add_argument(
        "--classification_json",
        type=str,
        default=None,
        help="pattern_classification.json for titles (optional)",
    )
    p.add_argument(
        "--stream_file",
        type=str,
        default=None,
        help="e.g. layer_00_head_00.npy (default: first .npy in folder)",
    )
    p.add_argument(
        "--vertical_file",
        type=str,
        default=None,
        help="e.g. layer_00_head_12.npy",
    )
    p.add_argument(
        "--block_file",
        type=str,
        default=None,
        help="e.g. layer_05_head_10.npy",
    )
    p.add_argument(
        "--downsample_max",
        type=int,
        default=0,
        help="Max side length; 0 = full resolution (default)",
    )
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument(
        "--cmap",
        type=str,
        default=DEFAULT_CMAP,
        help="Matplotlib colormap (default: viridis, blue-green-yellow). "
        "Also try: plasma, inferno, turbo, YlGnBu",
    )
    p.add_argument(
        "--no_combined",
        action="store_true",
        help="Skip three_patterns_combined.png",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pattern_sorted_dir = Path(args.pattern_sorted_dir)
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else pattern_sorted_dir / "viz_one_per_pattern"
    )
    classification_json = (
        Path(args.classification_json)
        if args.classification_json
        else None
    )

    picks = {}
    if args.stream_file:
        picks["StreamLLM(A-shape)"] = args.stream_file
    if args.vertical_file:
        picks["Vertical-Slash"] = args.vertical_file
    if args.block_file:
        picks["Block-wise"] = args.block_file

    downsample = args.downsample_max if args.downsample_max > 0 else None

    summary = visualize_pattern_picks(
        pattern_sorted_dir,
        out_dir,
        picks=picks or None,
        classification_json=classification_json,
        downsample_max=downsample,
        dpi=args.dpi,
        combined=not args.no_combined,
        cmap_name=args.cmap,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote plots under {out_dir}")


if __name__ == "__main__":
    main()
