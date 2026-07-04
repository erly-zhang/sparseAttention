#!/usr/bin/env python3
"""
Directional attention-head similarity via K% mass coverage.

For head i and head j, on each query row q (causal: keys k <= q):

1. On head i row q, select key indices S_q whose cumulative attention mass
   reaches K% of the row total (greedy: descending attention within k <= q).
2. On head j row q, compute coverage(q) = sum_{k in S_q} A_j[q,k] / sum_{k<=q} A_j[q,k].
3. sim(i -> j) = mean_{q >= q_start} coverage(q).

sim(i -> j) is generally asymmetric. We also save sym(i,j) = (sim(i->j) + sim(j->i)) / 2.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np


def q_start_default(L: int, q_start: Optional[int]) -> int:
    if q_start is not None:
        return max(0, min(int(q_start), L - 1))
    return min(2500, L // 2)


def load_head(sample_dir: Path, layer: int, head: int) -> np.ndarray:
    p = sample_dir / f"layer_{layer:02d}" / f"head_{head:02d}.npy"
    a = np.squeeze(np.load(p))
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"Expected square [L,L] at {p}, got {a.shape}")
    return a.astype(np.float32, copy=False)


def _causal_row(row: np.ndarray, q: int) -> np.ndarray:
    """Row for query q, keys 0..q only."""
    return row[: q + 1]


def keys_for_k_percent_mass(row: np.ndarray, q: int, k_percent: float) -> np.ndarray:
    """
    Greedy top-mass keys on causal row q of head i until K%% of row sum is covered.
    Returns 1D array of key indices (subset of [0, q]).
    """
    causal = _causal_row(row, q)
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


def row_coverage_on_head(
    row_j: np.ndarray,
    q: int,
    keys: np.ndarray,
) -> float:
    """Fraction of head j row-q mass that lies on keys selected from head i."""
    causal = _causal_row(row_j, q)
    total = float(causal.sum())
    if total <= 0:
        return 0.0
    if keys.size == 0:
        return 0.0
    covered = float(causal[keys].sum())
    return covered / total


def directional_similarity(
    A_i: np.ndarray,
    A_j: np.ndarray,
    *,
    k_percent: float,
    q0: int,
) -> float:
    """
    sim(i -> j): mean query-row coverage on head j using K%% masks from head i.
    """
    L = A_i.shape[0]
    if A_j.shape != (L, L):
        raise ValueError(f"Shape mismatch: {A_i.shape} vs {A_j.shape}")

    scores: List[float] = []
    for q in range(q0, L):
        keys = keys_for_k_percent_mass(A_i[q], q, k_percent)
        scores.append(row_coverage_on_head(A_j[q], q, keys))

    if not scores:
        return 0.0
    return float(np.mean(scores))


def load_all_heads(sample_dir: Path) -> Tuple[np.ndarray, int, int, int]:
    """Stack all layer/head maps as [N, L, L] with N = num_layers * num_heads."""
    num_layers, num_heads, L = infer_sample_dims(sample_dir)
    N = num_layers * num_heads
    maps = np.empty((N, L, L), dtype=np.float32)
    for layer in range(num_layers):
        for head in range(num_heads):
            idx = layer * num_heads + head
            maps[idx] = load_head(sample_dir, layer, head)
    return maps, num_layers, num_heads, L


def global_directional_matrix(
    maps: np.ndarray,
    *,
    k_percent: float,
    q0: int,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Full [N, N] directional similarity across all stacked heads.

    For fixed source head i, vectorize coverage over all target heads j per query q.
    """
    N, L, _ = maps.shape
    n_q = L - q0
    if n_q <= 0:
        return np.zeros((N, N), dtype=np.float32)

    keys_by_i_q: List[List[np.ndarray]] = []
    iterator = range(N)
    if show_progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="precompute keys", unit="head")
        except ImportError:
            pass

    for i in iterator:
        row_keys: List[np.ndarray] = []
        for q in range(q0, L):
            row_keys.append(keys_for_k_percent_mass(maps[i, q], q, k_percent))
        keys_by_i_q.append(row_keys)

    directional = np.zeros((N, N), dtype=np.float32)
    iterator_i = range(N)
    if show_progress:
        try:
            from tqdm import tqdm

            iterator_i = tqdm(iterator_i, desc="sim(i->j)", unit="src")
        except ImportError:
            pass

    inv_nq = 1.0 / n_q
    for i in iterator_i:
        accum = np.zeros(N, dtype=np.float64)
        for q_idx, q in enumerate(range(q0, L)):
            keys = keys_by_i_q[i][q_idx]
            causal = maps[:, q, : q + 1]
            totals = causal.sum(axis=1)
            if keys.size == 0:
                continue
            covered = causal[:, keys].sum(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                cov = np.divide(
                    covered,
                    totals,
                    out=np.zeros(N, dtype=np.float64),
                    where=totals > 0,
                )
            accum += cov
        directional[i, :] = (accum * inv_nq).astype(np.float32)
    return directional


def cross_sample_directional_matrix(
    maps_src: np.ndarray,
    maps_tgt: np.ndarray,
    *,
    k_percent: float,
    q0: int,
    show_progress: bool = False,
) -> Tuple[np.ndarray, int]:
    """
    [N_src, N_tgt] directional similarity: keys from src head i, coverage on tgt head j.

    Aligns both samples to min(L_src, L_tgt) token positions.
    """
    N_src, L_src, _ = maps_src.shape
    N_tgt, L_tgt, _ = maps_tgt.shape
    L_eff = min(L_src, L_tgt)
    q0 = max(0, min(int(q0), L_eff - 1))
    maps_src = maps_src[:, :L_eff, :L_eff]
    maps_tgt = maps_tgt[:, :L_eff, :L_eff]

    n_q = L_eff - q0
    if n_q <= 0:
        return np.zeros((N_src, N_tgt), dtype=np.float32), L_eff

    keys_by_i_q: List[List[np.ndarray]] = []
    iterator = range(N_src)
    if show_progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="precompute keys (src)", unit="head")
        except ImportError:
            pass

    for i in iterator:
        row_keys: List[np.ndarray] = []
        for q in range(q0, L_eff):
            row_keys.append(keys_for_k_percent_mass(maps_src[i, q], q, k_percent))
        keys_by_i_q.append(row_keys)

    directional = np.zeros((N_src, N_tgt), dtype=np.float32)
    iterator_i = range(N_src)
    if show_progress:
        try:
            from tqdm import tqdm

            iterator_i = tqdm(iterator_i, desc="sim(src→tgt)", unit="src")
        except ImportError:
            pass

    inv_nq = 1.0 / n_q
    for i in iterator_i:
        accum = np.zeros(N_tgt, dtype=np.float64)
        for q_idx, q in enumerate(range(q0, L_eff)):
            keys = keys_by_i_q[i][q_idx]
            causal = maps_tgt[:, q, : q + 1]
            totals = causal.sum(axis=1)
            if keys.size == 0:
                continue
            covered = causal[:, keys].sum(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                cov = np.divide(
                    covered,
                    totals,
                    out=np.zeros(N_tgt, dtype=np.float64),
                    where=totals > 0,
                )
            accum += cov
        directional[i, :] = (accum * inv_nq).astype(np.float32)
    return directional, L_eff


def cross_sample_directional_matrix_streaming(
    sample_dir_src: Path,
    maps_tgt: np.ndarray,
    *,
    num_layers: int,
    num_heads: int,
    k_percent: float,
    q0: int,
    L_eff: int,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Low-memory cross-sample [N_src, N_tgt]: keep tgt in RAM, stream src heads from disk.

    Peak RAM ~ one full sample stack (maps_tgt) instead of two.
    """
    sample_dir_src = Path(sample_dir_src)
    N_src = num_layers * num_heads
    N_tgt = maps_tgt.shape[0]
    maps_tgt = maps_tgt[:, :L_eff, :L_eff]
    q0 = max(0, min(int(q0), L_eff - 1))

    n_q = L_eff - q0
    if n_q <= 0:
        return np.zeros((N_src, N_tgt), dtype=np.float32)

    directional = np.zeros((N_src, N_tgt), dtype=np.float32)
    iterator = range(N_src)
    if show_progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="sim(src→tgt)", unit="src")
        except ImportError:
            pass

    inv_nq = 1.0 / n_q
    for flat_i in iterator:
        layer = flat_i // num_heads
        head = flat_i % num_heads
        A_i = load_head(sample_dir_src, layer, head)[:L_eff, :L_eff]

        accum = np.zeros(N_tgt, dtype=np.float64)
        for q in range(q0, L_eff):
            keys = keys_for_k_percent_mass(A_i[q], q, k_percent)
            causal = maps_tgt[:, q, : q + 1]
            totals = causal.sum(axis=1)
            if keys.size == 0:
                continue
            covered = causal[:, keys].sum(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                cov = np.divide(
                    covered,
                    totals,
                    out=np.zeros(N_tgt, dtype=np.float64),
                    where=totals > 0,
                )
            accum += cov
        directional[flat_i, :] = (accum * inv_nq).astype(np.float32)

    return directional


def pairwise_head_matrix(
    maps: List[np.ndarray],
    *,
    k_percent: float,
    q0: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    maps[h] = attention matrix for head h in the same layer.

    Returns:
        directional [H, H] where M[i,j] = sim(i->j)
        symmetric [H, H] where S[i,j] = (M[i,j]+M[j,i])/2
    """
    H = len(maps)
    directional = np.zeros((H, H), dtype=np.float32)
    for i in range(H):
        for j in range(H):
            directional[i, j] = directional_similarity(
                maps[i], maps[j], k_percent=k_percent, q0=q0
            )
    symmetric = 0.5 * (directional + directional.T)
    return directional, symmetric


def infer_sample_dims(sample_dir: Path) -> Tuple[int, int, int]:
    layers = sorted(sample_dir.glob("layer_*"))
    if not layers:
        raise FileNotFoundError(f"No layer_* under {sample_dir}")
    num_layers = len(layers)
    num_heads = len(list(layers[0].glob("head_*.npy")))
    a00 = load_head(sample_dir, 0, 0)
    return num_layers, num_heads, int(a00.shape[0])


def plot_similarity_heatmap(
    matrix: np.ndarray,
    out_png: Path,
    *,
    title: str,
    cmap: str = "viridis",
    num_heads: Optional[int] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 150,
    xlabel: str = "head j (layer-major)",
    ylabel: str = "head i (layer-major)",
) -> None:
    n = matrix.shape[0]
    if figsize is None:
        side = max(8.0, min(24.0, n / 32.0))
        figsize = (side, side)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, aspect="equal", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)

    if num_heads is not None and n > num_heads:
        ticks = np.arange(0, n + 1, num_heads)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([str(t // num_heads) for t in ticks], fontsize=7)
        ax.set_yticklabels([str(t // num_heads) for t in ticks], fontsize=7)
        for t in ticks[1:-1]:
            ax.axhline(t - 0.5, color="white", linewidth=0.25, alpha=0.5)
            ax.axvline(t - 0.5, color="white", linewidth=0.25, alpha=0.5)

    plt.colorbar(im, ax=ax, label="similarity", fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def compute_global_similarity(
    sample_dir: Path,
    out_dir: Path,
    *,
    k_percent: float = 95.0,
    q_start: Optional[int] = None,
    save_heatmaps: bool = True,
    cmap: str = "viridis",
    matrix_kind: str = "directional",
    show_progress: bool = True,
    dpi: int = 150,
) -> dict:
    """All layers × all heads as one [N, N] matrix (N = num_layers * num_heads)."""
    sample_dir = Path(sample_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    maps, num_layers, num_heads, L = load_all_heads(sample_dir)
    N = num_layers * num_heads
    q0 = q_start_default(L, q_start)

    directional = global_directional_matrix(
        maps, k_percent=k_percent, q0=q0, show_progress=show_progress
    )
    symmetric = 0.5 * (directional + directional.T)

    np.save(out_dir / "global_directional.npy", directional)
    np.save(out_dir / "global_symmetric.npy", symmetric)

    kind = matrix_kind.lower()
    if kind not in ("symmetric", "directional"):
        raise ValueError(f"matrix_kind must be symmetric or directional, got {matrix_kind!r}")
    matrix = symmetric if kind == "symmetric" else directional
    matrix_path = out_dir / f"global_{kind}.npy"
    heatmap_path = out_dir / f"global_{kind}_heatmap.png"

    meta = {
        "sample_dir": str(sample_dir),
        "seq_len": L,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "global_heads": N,
        "k_percent": k_percent,
        "q_start": q0,
        "matrix_kind": kind,
        "directional_mean": float(directional.mean()),
        "symmetric_mean": float(symmetric.mean()),
        "directional_path": str(out_dir / "global_directional.npy"),
        "symmetric_path": str(out_dir / "global_symmetric.npy"),
        "heatmap_path": str(heatmap_path),
        "indexing": "flat_idx = layer * num_heads + head",
    }

    if save_heatmaps:
        title_kind = "sim(i→j)" if kind == "directional" else "sym(i,j)"
        plot_similarity_heatmap(
            matrix,
            heatmap_path,
            title=f"Global {title_kind}, N={N} ({num_layers}L×{num_heads}H), K={k_percent}%, q>={q0}",
            cmap=cmap,
            num_heads=num_heads,
            dpi=dpi,
        )

    meta_path = out_dir / "similarity_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def compute_cross_sample_similarity(
    sample_dir_src: Path,
    sample_dir_tgt: Path,
    out_dir: Path,
    *,
    k_percent: float = 95.0,
    q_start: Optional[int] = None,
    save_heatmaps: bool = True,
    cmap: str = "viridis",
    show_progress: bool = True,
    dpi: int = 150,
) -> dict:
    """Directional sim from all heads in sample A to all heads in sample B."""
    sample_dir_src = Path(sample_dir_src)
    sample_dir_tgt = Path(sample_dir_tgt)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    num_layers_s, num_heads_s, L_src = infer_sample_dims(sample_dir_src)
    num_layers_t, num_heads_t, L_tgt = infer_sample_dims(sample_dir_tgt)
    if (num_layers_s, num_heads_s) != (num_layers_t, num_heads_t):
        raise ValueError(
            f"Layer/head layout mismatch: src=({num_layers_s},{num_heads_s}) "
            f"tgt=({num_layers_t},{num_heads_t})"
        )

    L_eff = min(L_src, L_tgt)
    q0 = q_start if q_start is not None else min(q_start_default(L_src, None), q_start_default(L_tgt, None))
    q0 = max(0, min(int(q0), L_eff - 1))

    # Load target once (~8GB); stream source heads to avoid 2× memory OOM.
    maps_tgt, _, _, _ = load_all_heads(sample_dir_tgt)
    directional = cross_sample_directional_matrix_streaming(
        sample_dir_src,
        maps_tgt,
        num_layers=num_layers_s,
        num_heads=num_heads_s,
        k_percent=k_percent,
        q0=q0,
        L_eff=L_eff,
        show_progress=show_progress,
    )
    del maps_tgt
    N_src, N_tgt = directional.shape
    L_used = L_eff

    np.save(out_dir / "cross_directional.npy", directional)
    heatmap_path = out_dir / "cross_directional_heatmap.png"
    src_name = sample_dir_src.name
    tgt_name = sample_dir_tgt.name

    meta = {
        "sample_dir_src": str(sample_dir_src),
        "sample_dir_tgt": str(sample_dir_tgt),
        "seq_len_src": L_src,
        "seq_len_tgt": L_tgt,
        "seq_len_aligned": L_used,
        "num_layers": num_layers_s,
        "num_heads": num_heads_s,
        "heads_src": N_src,
        "heads_tgt": N_tgt,
        "k_percent": k_percent,
        "q_start": q0,
        "matrix_kind": "directional",
        "definition": "M[i,j]=sim(src_head_i -> tgt_head_j) via K%% mass coverage on aligned tokens",
        "directional_mean": float(directional.mean()),
        "directional_path": str(out_dir / "cross_directional.npy"),
        "heatmap_path": str(heatmap_path),
        "indexing": "flat_idx = layer * num_heads + head",
    }

    if save_heatmaps:
        plot_similarity_heatmap(
            directional,
            heatmap_path,
            title=(
                f"Cross-sample sim(src→tgt), K={k_percent}%, q>={q0}, "
                f"N={N_src}×{N_tgt}, L={L_used}"
            ),
            cmap=cmap,
            num_heads=num_heads_s,
            dpi=dpi,
            xlabel=f"{tgt_name} head j (layer-major)",
            ylabel=f"{src_name} head i (layer-major)",
        )

    meta_path = out_dir / "similarity_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def compute_sample_similarity(
    sample_dir: Path,
    out_dir: Path,
    *,
    k_percent: float = 90.0,
    q_start: Optional[int] = None,
    layers: Optional[List[int]] = None,
    save_heatmaps: bool = True,
    cmap: str = "viridis",
    matrix_kind: str = "directional",
    dpi: int = 150,
) -> dict:
    sample_dir = Path(sample_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    num_layers, num_heads, L = infer_sample_dims(sample_dir)
    q0 = q_start_default(L, q_start)
    layer_list = layers if layers is not None else list(range(num_layers))

    meta = {
        "sample_dir": str(sample_dir),
        "seq_len": L,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "k_percent": k_percent,
        "q_start": q0,
        "definition": (
            "sim(i->j)=mean_q coverage on head j row q using keys from head i "
            "that cover K% of head i row mass (causal k<=q)"
        ),
    }

    kind = matrix_kind.lower()
    if kind not in ("symmetric", "directional"):
        raise ValueError(f"matrix_kind must be symmetric or directional, got {matrix_kind!r}")

    layer_results = {}
    for layer in layer_list:
        maps = [load_head(sample_dir, layer, h) for h in range(num_heads)]
        directional, symmetric = pairwise_head_matrix(
            maps, k_percent=k_percent, q0=q0
        )

        np.save(out_dir / f"layer_{layer:02d}_directional.npy", directional)
        np.save(out_dir / f"layer_{layer:02d}_symmetric.npy", symmetric)

        layer_results[str(layer)] = {
            "directional_mean": float(directional.mean()),
            "symmetric_mean": float(symmetric.mean()),
            "directional_path": str(out_dir / f"layer_{layer:02d}_directional.npy"),
            "symmetric_path": str(out_dir / f"layer_{layer:02d}_symmetric.npy"),
        }

        if save_heatmaps:
            if kind == "directional":
                plot_similarity_heatmap(
                    directional,
                    out_dir / f"layer_{layer:02d}_directional.png",
                    title=f"Layer {layer} sim(i→j), K={k_percent}%, q>={q0}",
                    cmap=cmap,
                    dpi=dpi,
                )
            else:
                plot_similarity_heatmap(
                    symmetric,
                    out_dir / f"layer_{layer:02d}_symmetric.png",
                    title=f"Layer {layer} sym(i,j), K={k_percent}%, q>={q0}",
                    cmap=cmap,
                    dpi=dpi,
                )

    meta["layers"] = layer_results
    meta_path = out_dir / "similarity_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="K%% mass-coverage similarity between attention heads"
    )
    p.add_argument(
        "--sample_dir",
        type=str,
        default=None,
        help="sample_* dir with layer_XX/head_YY.npy (required unless --dataset_dir)",
    )
    p.add_argument(
        "--sample_dir_tgt",
        type=str,
        default=None,
        help="Second sample for cross-sample sim(src->tgt); requires --sample_dir as source",
    )
    p.add_argument(
        "--k_percent",
        type=float,
        default=90.0,
        help="Cumulative attention mass %% to select keys on head i (default 90)",
    )
    p.add_argument("--q_start", type=int, default=None, help="First query row (default L//2)")
    p.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated layer indices (default: all)",
    )
    p.add_argument("--no_heatmaps", action="store_true")
    p.add_argument("--cmap", type=str, default="viridis")
    p.add_argument(
        "--global_matrix",
        action="store_true",
        help="One N×N matrix over all layers×heads (default: per-layer 16×16)",
    )
    p.add_argument(
        "--matrix_kind",
        type=str,
        default="directional",
        choices=("symmetric", "directional"),
        help="Heatmap matrix for global mode (default directional, sim i→j)",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="PNG resolution for heatmaps (default 150)",
    )
    p.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="Parent dir with sample_* subdirs; run global N×N on each sample",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Default: per-mode under sample dir or cross dir",
    )
    p.add_argument("--no_progress", action="store_true")
    return p.parse_args()


def _cross_out_dir(
    base: Path,
    sample_dir_src: Path,
    sample_dir_tgt: Path,
    k: Union[float, int],
) -> Path:
    return base / f"cross_{sample_dir_src.name}_to_{sample_dir_tgt.name}_k{k}"


def _parse_layers(s: Optional[str]) -> Optional[List[int]]:
    if not s:
        return None
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _discover_samples(parent: Path) -> List[Path]:
    samples = sorted(parent.glob("sample_*"))
    if not samples:
        raise FileNotFoundError(f"No sample_* under {parent}")
    return [p for p in samples if p.is_dir()]


def main() -> None:
    args = parse_args()
    k = int(args.k_percent) if float(args.k_percent).is_integer() else args.k_percent
    show_progress = not args.no_progress

    if args.dataset_dir:
        parent = Path(args.dataset_dir)
        samples = _discover_samples(parent)
        all_meta = {}
        for sample_dir in samples:
            out_dir = sample_dir / f"similarity_k{k}_global"
            meta = compute_global_similarity(
                sample_dir,
                out_dir,
                k_percent=float(args.k_percent),
                q_start=args.q_start,
                save_heatmaps=not args.no_heatmaps,
                cmap=args.cmap,
                matrix_kind=args.matrix_kind,
                show_progress=show_progress,
                dpi=int(args.dpi),
            )
            all_meta[sample_dir.name] = meta
            print(f"Wrote global similarity under {out_dir}")
        summary_path = parent / f"similarity_k{k}_global_summary.json"
        summary_path.write_text(
            json.dumps(all_meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(all_meta, indent=2, ensure_ascii=False))
        print(f"Summary: {summary_path}")
        return

    if not args.sample_dir:
        raise SystemExit("Provide --sample_dir or --dataset_dir")
    sample_dir = Path(args.sample_dir)

    if args.sample_dir_tgt:
        sample_dir_tgt = Path(args.sample_dir_tgt)
        parent = sample_dir.parent
        out_dir = (
            Path(args.out_dir)
            if args.out_dir
            else _cross_out_dir(parent, sample_dir, sample_dir_tgt, k)
        )
        meta = compute_cross_sample_similarity(
            sample_dir,
            sample_dir_tgt,
            out_dir,
            k_percent=float(args.k_percent),
            q_start=args.q_start,
            save_heatmaps=not args.no_heatmaps,
            cmap=args.cmap,
            show_progress=show_progress,
            dpi=int(args.dpi),
        )
        print(json.dumps(meta, indent=2, ensure_ascii=False))
        print(f"Wrote cross-sample similarity under {out_dir}")
        return

    out_suffix = f"similarity_k{k}_global" if args.global_matrix else f"similarity_k{k}"
    out_dir = Path(args.out_dir) if args.out_dir else sample_dir / out_suffix

    if args.global_matrix:
        meta = compute_global_similarity(
            sample_dir,
            out_dir,
            k_percent=float(args.k_percent),
            q_start=args.q_start,
            save_heatmaps=not args.no_heatmaps,
            cmap=args.cmap,
            matrix_kind=args.matrix_kind,
            show_progress=show_progress,
            dpi=int(args.dpi),
        )
    else:
        meta = compute_sample_similarity(
            sample_dir,
            out_dir,
            k_percent=float(args.k_percent),
            q_start=args.q_start,
            layers=_parse_layers(args.layers),
            save_heatmaps=not args.no_heatmaps,
            cmap=args.cmap,
            matrix_kind=args.matrix_kind,
            dpi=int(args.dpi),
        )
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"Wrote similarity under {out_dir}")


if __name__ == "__main__":
    main()
