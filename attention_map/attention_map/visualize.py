#!/usr/bin/env python3
"""Visualize saved attention maps as heatmaps (darker = higher value)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


# 关键修复：
# "Greys" 才是 0=白、1=黑，也就是 darker=higher
# "Greys_r" 会反过来，导致接近 0 的 attention 被画成黑色
CMAP_NAME = "Greys"


def get_display_cmap():
    cmap = plt.get_cmap(CMAP_NAME).copy()
    cmap.set_bad(color="white")
    return cmap


def load_head_attn(sample_dir: Path, layer: int, head: int) -> np.ndarray:
    path = sample_dir / f"layer_{layer:02d}" / f"head_{head:02d}.npy"
    return np.load(path).astype(np.float32)


def load_sample_meta(sample_dir: Path) -> dict:
    meta_path = sample_dir / "sample_meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))

    manifest_path = sample_dir / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    return {}


def discover_samples(task_dir: Path) -> List[Path]:
    samples = sorted(
        p for p in task_dir.iterdir()
        if p.is_dir() and p.name.startswith("sample_")
    )
    if not samples:
        raise FileNotFoundError(f"No sample_* directories under {task_dir}")
    return samples


def is_full_matrix(attn: np.ndarray) -> bool:
    """True if saved as square [seq, seq] attention matrix."""
    attn = np.squeeze(attn)
    return attn.ndim == 2 and attn.shape[0] > 1 and attn.shape[0] == attn.shape[1]


def to_2d(attn: np.ndarray) -> np.ndarray:
    attn = np.squeeze(attn)
    if attn.ndim == 1:
        return attn[np.newaxis, :]
    return attn


def to_square(attn: np.ndarray) -> np.ndarray:
    attn = np.squeeze(attn)
    if attn.ndim != 2 or attn.shape[0] != attn.shape[1]:
        raise ValueError(f"Expected square attention matrix, got shape={attn.shape}")
    return attn


def detect_mode(sample_dir: Path) -> str:
    meta = load_sample_meta(sample_dir)
    qs = meta.get("query_slice")

    if qs == "all":
        return "full"
    if qs == "last":
        return "last"

    attn = load_head_attn(sample_dir, 0, 0)
    return "full" if is_full_matrix(attn) else "last"


def get_dims(sample_dir: Path) -> Tuple[int, int, int, str]:
    meta = load_sample_meta(sample_dir)
    mode = detect_mode(sample_dir)

    if "num_layers" in meta and "num_attention_heads" in meta:
        seq_len = int(meta["seq_len"])
        return int(meta["num_layers"]), int(meta["num_attention_heads"]), seq_len, mode

    layers = sorted(sample_dir.glob("layer_*"))
    num_layers = len(layers)

    if num_layers == 0:
        raise FileNotFoundError(f"No layer_* directories under {sample_dir}")

    num_heads = len(list(layers[0].glob("head_*.npy")))

    attn = np.squeeze(load_head_attn(sample_dir, 0, 0))
    if mode == "full":
        seq_len = attn.shape[0]
    else:
        seq_len = attn.shape[-1]

    return num_layers, num_heads, seq_len, mode


def compute_global_vmax(
    sample_dir: Path,
    num_layers: int,
    num_heads: int,
    layers_subset: Optional[List[int]] = None,
) -> float:
    layers = layers_subset if layers_subset is not None else list(range(num_layers))
    vmax = 0.0

    for layer in layers:
        for head in range(num_heads):
            attn = load_head_attn(sample_dir, layer, head)
            vmax = max(vmax, float(np.max(attn)))

    return vmax if vmax > 0 else 1.0


def layer_mean_map(sample_dir: Path, layer: int, num_heads: int) -> np.ndarray:
    stack = np.stack(
        [to_square(load_head_attn(sample_dir, layer, h)) for h in range(num_heads)],
        axis=0,
    )
    return stack.mean(axis=0)


def apply_causal_mask_for_display(attn: np.ndarray) -> np.ma.MaskedArray:
    """
    Mask upper triangle for causal attention.
    Masked area will be displayed as white.
    """
    n = attn.shape[0]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    return np.ma.array(attn, mask=mask)


def vmax_of_array(data: np.ndarray) -> float:
    if np.ma.isMaskedArray(data):
        val = float(data.max())
    else:
        val = float(np.max(data))
    return val if val > 0 else 1.0


def robust_vmax(data: np.ndarray, percentile: float = 99.5) -> float:
    """
    Use percentile scaling instead of raw max.

    Attention values are usually very small.
    If one or several entries are much larger than others,
    using max as vmax will make the rest of the map almost invisible.
    """
    if np.ma.isMaskedArray(data):
        vals = data.compressed()
    else:
        vals = np.asarray(data).ravel()

    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return 1.0

    val = float(np.percentile(vals, percentile))

    if val > 0:
        return val

    max_val = float(np.max(vals))
    return max_val if max_val > 0 else 1.0


def downsample_square(attn: np.ndarray, max_size: int = 128) -> np.ndarray:
    n = attn.shape[0]
    if n <= max_size:
        return attn

    step = int(np.ceil(n / max_size))
    return attn[::step, ::step]


def plot_heatmap(
    attn: np.ndarray,
    ax: plt.Axes,
    vmax: Optional[float] = None,
    title: str = "",
    *,
    square: bool = False,
    causal_mask: bool = False,
    mark_index: Optional[int] = None,
    vmin: float = 0.0,
) -> None:
    data = to_square(attn) if square else to_2d(attn)

    if causal_mask and data.ndim == 2 and data.shape[0] == data.shape[1]:
        data = apply_causal_mask_for_display(data)

    if vmax is None:
        vmax = robust_vmax(data)

    is_square_2d = (
        data.ndim == 2
        and data.shape[0] == data.shape[1]
        and data.shape[0] > 1
    )

    im = ax.imshow(
        data,
        aspect="equal" if square and is_square_2d else "auto",
        cmap=get_display_cmap(),
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    ax.set_title(title, fontsize=8)

    if data.ndim == 1 or (data.ndim == 2 and data.shape[0] == 1):
        ax.set_yticks([])
        ax.set_xlabel("key", fontsize=6)
    elif square:
        ax.set_xlabel("key", fontsize=6)
        ax.set_ylabel("query", fontsize=6)

    if mark_index is not None and data.ndim == 2:
        idx = min(max(0, mark_index), data.shape[0] - 1)
        ax.axvline(idx, color="red", linewidth=0.6, alpha=0.8)

        if data.shape[0] > 1:
            ax.axhline(idx, color="red", linewidth=0.6, alpha=0.8)

    return im


def _default_layers(num_layers: int) -> List[int]:
    return sorted(
        {
            0,
            num_layers // 4,
            num_layers // 2,
            3 * num_layers // 4,
            num_layers - 1,
        }
    )


def visualize_sample_last(
    sample_dir: Path,
    out_dir: Path,
    num_layers: int,
    num_heads: int,
    seq_len: int,
    layers_to_plot: List[int],
    answer_pos: Optional[int],
) -> None:
    global_vmax = compute_global_vmax(sample_dir, num_layers, num_heads)

    layer_mean = np.zeros((num_layers, seq_len), dtype=np.float32)

    for layer in range(num_layers):
        stack = np.stack(
            [
                to_2d(load_head_attn(sample_dir, layer, head)).squeeze()
                for head in range(num_heads)
            ],
            axis=0,
        )
        layer_mean[layer] = stack.mean(axis=0)

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(
        layer_mean,
        aspect="auto",
        cmap=get_display_cmap(),
        vmin=0.0,
        vmax=robust_vmax(layer_mean),
        interpolation="nearest",
    )

    ax.set_xlabel("Key position")
    ax.set_ylabel("Layer index")
    ax.set_title(f"Mean attention over heads, last query row — {sample_dir.name}")

    if answer_pos is not None:
        ax.axvline(answer_pos, color="red", linewidth=1.0, label="answer token")
        ax.legend(loc="upper right", fontsize=8)

    plt.colorbar(im, ax=ax, label="attention weight")
    fig.tight_layout()
    fig.savefig(out_dir / "overview_layers_mean_heads.png", dpi=150)
    plt.close(fig)

    for layer in layers_to_plot:
        fig, axes = plt.subplots(4, 4, figsize=(16, 4))
        fig.suptitle(f"Layer {layer} — last-query row per head", fontsize=12)

        for head in range(num_heads):
            ax = axes[head // 4, head % 4]
            attn = load_head_attn(sample_dir, layer, head)
            plot_heatmap(
                attn,
                ax,
                vmax=global_vmax,
                title=f"head {head}",
                mark_index=answer_pos,
            )

        fig.tight_layout()
        fig.savefig(out_dir / f"layer_{layer:02d}_all_heads.png", dpi=150)
        plt.close(fig)

    head_max = np.zeros((num_layers, num_heads), dtype=np.float32)

    for layer in range(num_layers):
        for head in range(num_heads):
            head_max[layer, head] = to_2d(load_head_attn(sample_dir, layer, head)).max()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(
        head_max,
        aspect="auto",
        cmap=get_display_cmap(),
        vmin=0,
        vmax=robust_vmax(head_max),
    )

    ax.set_xlabel("Head index")
    ax.set_ylabel("Layer index")
    ax.set_title(f"Max attention per layer/head — {sample_dir.name}")
    plt.colorbar(im, ax=ax, label="max weight")
    fig.tight_layout()
    fig.savefig(out_dir / "overview_layer_head_max.png", dpi=150)
    plt.close(fig)


def visualize_sample_full(
    sample_dir: Path,
    out_dir: Path,
    num_layers: int,
    num_heads: int,
    layers_to_plot: List[int],
    answer_pos: Optional[int],
    grid_downsample: int,
) -> None:
    # 1. 所有层的 mean-over-head 方阵概览
    cols = 6
    rows = int(np.ceil(num_layers / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_2d(axes)

    cmap = get_display_cmap()

    for idx in range(rows * cols):
        ax = axes[idx // cols, idx % cols]

        if idx >= num_layers:
            ax.axis("off")
            continue

        mean_map = layer_mean_map(sample_dir, idx, num_heads)
        disp = downsample_square(mean_map, grid_downsample)
        disp = apply_causal_mask_for_display(disp)

        ax.imshow(
            disp,
            aspect="equal",
            cmap=cmap,
            vmin=0,
            vmax=robust_vmax(disp),
            interpolation="nearest",
        )

        ax.set_title(f"L{idx}", fontsize=7)
        ax.axis("off")

    fig.suptitle(
        f"All layers — mean over heads, square attention — {sample_dir.name}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "overview_all_layers_mean_square_grid.png", dpi=150)
    plt.close(fig)

    # 2. 指定层：mean-over-head 方阵 + 每个 head 的完整方阵
    for layer in layers_to_plot:
        mean_map = layer_mean_map(sample_dir, layer, num_heads)

        fig, ax = plt.subplots(figsize=(8, 8))
        plot_heatmap(
            mean_map,
            ax,
            vmax=robust_vmax(mean_map),
            title=f"Layer {layer} — mean over heads [seq × seq]",
            square=True,
            causal_mask=True,
            mark_index=answer_pos,
        )

        plt.colorbar(
            ax.images[0],
            ax=ax,
            fraction=0.046,
            pad=0.04,
            label="attention",
        )

        fig.tight_layout()
        fig.savefig(out_dir / f"layer_{layer:02d}_mean_heads_square.png", dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(4, 4, figsize=(20, 20))
        fig.suptitle(
            f"Layer {layer} — full square attention per head "
            f"(causal, darker=higher)",
            fontsize=12,
        )

        for head in range(num_heads):
            ax = axes[head // 4, head % 4]
            attn = load_head_attn(sample_dir, layer, head)

            # 每个 head 单独用 robust_vmax 自动缩放
            plot_heatmap(
                attn,
                ax,
                vmax=None,
                title=f"head {head}",
                square=True,
                causal_mask=True,
                mark_index=answer_pos,
            )

        fig.tight_layout()
        fig.savefig(out_dir / f"layer_{layer:02d}_all_heads_square.png", dpi=150)
        plt.close(fig)

        # 3. 从 full matrix 中取最后一个 query row，方便观察 long-context retrieval pattern
        last_rows = []
        row_vmax = 0.0

        for head in range(num_heads):
            row = to_square(load_head_attn(sample_dir, layer, head))[-1:]
            last_rows.append(row)
            row_vmax = max(row_vmax, float(row.max()))

        row_vmax = row_vmax if row_vmax > 0 else 1.0

        fig, axes = plt.subplots(4, 4, figsize=(16, 4))
        fig.suptitle(f"Layer {layer} — last query row from full matrix", fontsize=12)

        for head in range(num_heads):
            ax = axes[head // 4, head % 4]
            plot_heatmap(
                last_rows[head],
                ax,
                vmax=row_vmax,
                title=f"head {head}",
                mark_index=answer_pos,
            )

        fig.tight_layout()
        fig.savefig(out_dir / f"layer_{layer:02d}_all_heads_last_row.png", dpi=150)
        plt.close(fig)

    # 4. layer × head 的最大 attention 值
    head_max = np.zeros((num_layers, num_heads), dtype=np.float32)

    for layer in range(num_layers):
        for head in range(num_heads):
            head_max[layer, head] = to_square(
                load_head_attn(sample_dir, layer, head)
            ).max()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(
        head_max,
        aspect="auto",
        cmap=get_display_cmap(),
        vmin=0,
        vmax=robust_vmax(head_max),
    )

    ax.set_xlabel("Head index")
    ax.set_ylabel("Layer index")
    ax.set_title(f"Max attention per layer/head — full matrix — {sample_dir.name}")

    plt.colorbar(im, ax=ax, label="max weight")
    fig.tight_layout()
    fig.savefig(out_dir / "overview_layer_head_max.png", dpi=150)
    plt.close(fig)


def visualize_sample(
    sample_dir: Path,
    out_dir: Path,
    layers_to_plot: Optional[List[int]] = None,
    grid_downsample: int = 128,
) -> None:
    num_layers, num_heads, seq_len, mode = get_dims(sample_dir)
    meta = load_sample_meta(sample_dir)
    answer_pos = meta.get("token_position_answer")

    if layers_to_plot is None:
        layers_to_plot = _default_layers(num_layers)
    else:
        layers_to_plot = sorted(
            set(min(max(0, layer), num_layers - 1) for layer in layers_to_plot)
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    mode_note = {
        "mode": mode,
        "seq_len": seq_len,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "layers_to_plot": layers_to_plot,
    }

    with (out_dir / "viz_meta.json").open("w", encoding="utf-8") as f:
        json.dump(mode_note, f, indent=2)

    if mode == "full":
        visualize_sample_full(
            sample_dir=sample_dir,
            out_dir=out_dir,
            num_layers=num_layers,
            num_heads=num_heads,
            layers_to_plot=layers_to_plot,
            answer_pos=answer_pos,
            grid_downsample=grid_downsample,
        )
    else:
        visualize_sample_last(
            sample_dir=sample_dir,
            out_dir=out_dir,
            num_layers=num_layers,
            num_heads=num_heads,
            seq_len=seq_len,
            layers_to_plot=layers_to_plot,
            answer_pos=answer_pos,
        )

    print(f"Saved visualizations ({mode} mode) to {out_dir}")


def visualize_task_dir(
    task_dir: Path,
    vis_root: Optional[Path] = None,
    **kwargs,
) -> None:
    samples = discover_samples(task_dir)

    for sample_dir in samples:
        out = (vis_root or task_dir / "visualizations") / sample_dir.name
        visualize_sample(sample_dir, out, **kwargs)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Heatmap visualization for saved attention maps"
    )

    p.add_argument(
        "--input_dir",
        type=str,
        default="/home/ubuntu/work/attention_map/outputs/4k/niah_single_1",
        help="Task dir containing sample_* dirs, or one single sample_* dir",
    )

    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Default: <input_dir>/visualizations/<sample_name>",
    )

    p.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated layer indices for detailed plots, e.g. 0,17,35",
    )

    p.add_argument(
        "--grid_downsample",
        type=int,
        default=128,
        help="Max side length for all-layers overview grid in full-matrix mode",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)

    layers = None
    if args.layers:
        layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]

    kwargs = {
        "layers_to_plot": layers,
        "grid_downsample": args.grid_downsample,
    }

    if input_dir.name.startswith("sample_"):
        out_dir = (
            Path(args.output_dir)
            if args.output_dir
            else input_dir.parent / "visualizations" / input_dir.name
        )
        visualize_sample(input_dir, out_dir, **kwargs)
    else:
        vis_root = (
            Path(args.output_dir)
            if args.output_dir
            else input_dir / "visualizations"
        )

        for sample_dir in discover_samples(input_dir):
            visualize_sample(sample_dir, vis_root / sample_dir.name, **kwargs)


if __name__ == "__main__":
    main()