#!/usr/bin/env python3
"""
Compute per-column argmax heads and an "integrated" mapping (chain-compressed),
with a layer constraint.

Given a similarity matrix M (shape [N, N], N = num_layers * num_heads):
1) For each column j, pick i* = argmax_i M[i, j] (ties -> smallest i).
   This defines f(j) = i* and also records max_value(j).
2) Integration: repeatedly apply f until reaching a terminal representative.
   - If a fixed point exists, it becomes the representative.
   - If a cycle exists, choose the smallest index in the cycle as representative.
   This defines g(j).
3) Layer constraint: if layer(g(j)) > layer(j), set final(j) = j, else final(j) = g(j).

Outputs CSV + JSON per matrix.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--analysis_dir",
        type=str,
        default="/home/ubuntu/work/attention_map/analysis/ab_similarity_experiment_k95",
        help="Directory containing summary.json and matrix .npy files",
    )
    p.add_argument(
        "--matrices",
        nargs="+",
        default=[
            "M_AA_directional.npy",
            "M_AB_directional.npy",
            "M_BB_directional.npy",
        ],
        help="One or more .npy filenames under analysis_dir",
    )
    p.add_argument(
        "--also_binary",
        action="store_true",
        help="If set and M_AB_binary.npy exists, compute mappings for it too",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory (default: analysis_dir/mappings)",
    )
    p.add_argument(
        "--exclude_diagonal",
        action="store_true",
        help="When picking per-column argmax, disallow i=j (ignore diagonal self-match)",
    )
    p.add_argument(
        "--similarity_threshold",
        type=float,
        default=None,
        help=(
            "If set, keep final mapping j->i only when M[i,j] >= threshold; "
            "otherwise fall back to j->j."
        ),
    )
    p.add_argument(
        "--no_plots",
        action="store_true",
        help="Skip PNG visualizations",
    )
    p.add_argument(
        "--plots_dir",
        type=str,
        default=None,
        help="Directory for PNGs (default: out_dir/plots)",
    )
    p.add_argument(
        "--only_mapping_matrix",
        action="store_true",
        help="Only output the final mapping matrix PNG (skip other plot types)",
    )
    return p.parse_args()


def load_summary(analysis_dir: Path) -> Tuple[int, int]:
    summary_path = analysis_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing summary.json: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return int(summary["num_layers"]), int(summary["num_heads"])


def layer_of(flat: int, num_heads: int) -> int:
    return int(flat // num_heads)


def head_of(flat: int, num_heads: int) -> int:
    return int(flat % num_heads)


def per_column_argmax(M: np.ndarray, *, exclude_diagonal: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (f, maxv) where:
      f[j] = argmax_i M[i, j] with tie-breaker smallest i
      maxv[j] = M[f[j], j]
    """
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"Expected square matrix, got {M.shape}")
    N = M.shape[0]
    f = np.zeros(N, dtype=np.int32)
    maxv = np.zeros(N, dtype=np.float32)
    # Tie-breaking: find max, then choose smallest index among equals.
    for j in range(N):
        col = M[:, j]
        if exclude_diagonal:
            # Disallow i=j by treating it as -inf for argmax.
            # (copy is necessary to avoid modifying mmap-backed arrays)
            col = np.array(col, copy=True)
            col[j] = -np.inf
        m = float(np.max(col))
        maxv[j] = m
        # np.where returns increasing indices; pick first -> smallest i
        f[j] = int(np.where(col == m)[0][0])
    return f, maxv


def integrate_mapping(f: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Integrate (chain-compress) mapping f into representative g.

    Returns:
      g[j] = representative of j
      steps[j] = number of f-applications taken until termination/cycle resolved
    """
    N = int(f.shape[0])
    g = np.full(N, -1, dtype=np.int32)
    steps = np.zeros(N, dtype=np.int32)

    for start in range(N):
        if g[start] != -1:
            continue
        seen: Dict[int, int] = {}
        path: List[int] = []
        cur = start
        while True:
            if g[cur] != -1:
                rep = int(g[cur])
                break
            if cur in seen:
                # Found a cycle: nodes from seen[cur] onward.
                cycle = path[seen[cur] :]
                rep = int(min(cycle))
                break
            seen[cur] = len(path)
            path.append(cur)
            cur = int(f[cur])

        # Assign representative for all nodes in the path.
        for idx, node in enumerate(path):
            g[node] = rep
            steps[node] = idx + 1

    return g, steps


def apply_layer_constraint(
    j_to_rep: np.ndarray, *, num_heads: int
) -> np.ndarray:
    N = int(j_to_rep.shape[0])
    out = j_to_rep.copy()
    for j in range(N):
        rep = int(out[j])
        if layer_of(rep, num_heads) > layer_of(j, num_heads):
            out[j] = j
    return out


def apply_similarity_threshold(
    M: np.ndarray,
    mapping: np.ndarray,
    *,
    threshold: float | None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    If threshold is provided, for each column j:
      i = mapping[j]
      if M[i, j] < threshold: mapping[j] = j
    Returns (new_mapping, sim_values) where sim_values[j] = M[mapping_before_threshold[j], j].
    """
    N = int(mapping.shape[0])
    sim = np.zeros(N, dtype=np.float32)
    out = mapping.copy()
    for j in range(N):
        i = int(mapping[j])
        sim[j] = float(M[i, j])
        if threshold is not None and sim[j] < threshold:
            out[j] = j
    return out, sim


def write_outputs(
    *,
    out_dir: Path,
    matrix_name: str,
    num_heads: int,
    f: np.ndarray,
    maxv: np.ndarray,
    g: np.ndarray,
    steps: np.ndarray,
    final: np.ndarray,
    final_thr: np.ndarray,
    sim_final: np.ndarray,
    similarity_threshold: float | None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = matrix_name.replace(".npy", "")

    # CSV
    csv_path = out_dir / f"{stem}__per_column_mapping.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(
            [
                "j_flat",
                "j_layer",
                "j_head",
                "argmax_i_flat",
                "argmax_i_layer",
                "argmax_i_head",
                "max_value",
                "integrated_rep_flat",
                "integrated_rep_layer",
                "integrated_rep_head",
                "integration_steps",
                "final_rep_flat",
                "final_rep_layer",
                "final_rep_head",
                "final_rep_sim",
                "final_thr_rep_flat",
                "final_thr_rep_layer",
                "final_thr_rep_head",
            ]
        )
        for j in range(len(f)):
            i = int(f[j])
            rep = int(g[j])
            fin = int(final[j])
            fin_thr = int(final_thr[j])
            w.writerow(
                [
                    j,
                    layer_of(j, num_heads),
                    head_of(j, num_heads),
                    i,
                    layer_of(i, num_heads),
                    head_of(i, num_heads),
                    float(maxv[j]),
                    rep,
                    layer_of(rep, num_heads),
                    head_of(rep, num_heads),
                    int(steps[j]),
                    fin,
                    layer_of(fin, num_heads),
                    head_of(fin, num_heads),
                    float(sim_final[j]),
                    fin_thr,
                    layer_of(fin_thr, num_heads),
                    head_of(fin_thr, num_heads),
                ]
            )

    # JSON (compact mapping arrays + meta)
    json_path = out_dir / f"{stem}__per_column_mapping.json"
    payload = {
        "matrix": matrix_name,
        "N": int(len(f)),
        "num_heads": int(num_heads),
        "f_argmax": [int(x) for x in f.tolist()],
        "max_value": [float(x) for x in maxv.tolist()],
        "g_integrated": [int(x) for x in g.tolist()],
        "integration_steps": [int(x) for x in steps.tolist()],
        "final_with_layer_constraint": [int(x) for x in final.tolist()],
        "final_similarity": [float(x) for x in sim_final.tolist()],
        "final_with_threshold": [int(x) for x in final_thr.tolist()],
        "similarity_threshold": similarity_threshold,
        "note": {
            "tie_break": "if multiple i share max value in column j, choose smallest i",
            "integration": "follow f until fixed point; if cycle, use smallest index in cycle",
            "layer_constraint": "if layer(final) > layer(j), set final=j",
            "threshold": "if similarity_threshold is set and M[final,j] < threshold, set final=j",
        },
    }
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


def _imshow_grid(
    grid: np.ndarray,
    out_png: Path,
    *,
    title: str,
    cmap: str,
    cbar_label: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 10.0))
    im = ax.imshow(
        grid,
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.set_xticks(np.arange(grid.shape[1]))
    # Reduce y tick density for readability
    step = max(1, grid.shape[0] // 18)
    ax.set_yticks(np.arange(0, grid.shape[0], step))
    plt.colorbar(im, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _plot_layer_transition(
    counts: np.ndarray, out_png: Path, *, title: str
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(
        counts,
        aspect="equal",
        cmap="magma",
        interpolation="nearest",
    )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("final layer")
    ax.set_ylabel("source layer (j)")
    plt.colorbar(im, ax=ax, label="count", fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _plot_hist(
    values: np.ndarray, out_png: Path, *, title: str, xlabel: str
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(values, bins=50, color="#4c72b0", alpha=0.9)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _plot_mapping_matrix(
    mapping: np.ndarray,
    out_png: Path,
    *,
    title: str,
) -> None:
    """
    Plot a sparse 0/1 matrix where entry (i, j)=1 means column-head j maps to row-head i.
    This matches the original similarity matrix indexing convention (rows=i, cols=j).
    """
    N = int(mapping.shape[0])
    mat = np.zeros((N, N), dtype=np.uint8)
    for j in range(N):
        i = int(mapping[j])
        if 0 <= i < N:
            mat[i, j] = 1

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(mat, aspect="equal", cmap="gray_r", interpolation="nearest")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("head j (column)")
    ax.set_ylabel("head i (row)")
    plt.colorbar(im, ax=ax, label="mapping (1=selected)", fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def write_plots(
    *,
    plots_dir: Path,
    matrix_name: str,
    num_layers: int,
    num_heads: int,
    f: np.ndarray,
    maxv: np.ndarray,
    g: np.ndarray,
    final: np.ndarray,
    final_thr: np.ndarray,
    only_mapping_matrix: bool,
    similarity_threshold: float | None,
) -> None:
    stem = matrix_name.replace(".npy", "")
    plots_dir.mkdir(parents=True, exist_ok=True)

    # (layer, head) grids for quick inspection
    def to_grid(flat_arr: np.ndarray) -> np.ndarray:
        return flat_arr.reshape(num_layers, num_heads)

    f_layer = np.array([layer_of(int(x), num_heads) for x in f], dtype=np.int32)
    g_layer = np.array([layer_of(int(x), num_heads) for x in g], dtype=np.int32)
    final_layer = np.array(
        [layer_of(int(x), num_heads) for x in final], dtype=np.int32
    )

    _imshow_grid(
        to_grid(f_layer),
        plots_dir / f"{stem}__argmax_layer.png",
        title=f"{stem}: per-column argmax layer(i*)",
        cmap="viridis",
        cbar_label="layer(i*)",
        vmin=0,
        vmax=max(1, num_layers - 1),
    ) if not only_mapping_matrix else None
    _imshow_grid(
        to_grid(g_layer),
        plots_dir / f"{stem}__integrated_rep_layer.png",
        title=f"{stem}: integrated representative layer(g(j))",
        cmap="viridis",
        cbar_label="layer(g(j))",
        vmin=0,
        vmax=max(1, num_layers - 1),
    ) if not only_mapping_matrix else None
    _imshow_grid(
        to_grid(final_layer),
        plots_dir / f"{stem}__final_rep_layer.png",
        title=f"{stem}: final representative layer (after constraint)",
        cmap="viridis",
        cbar_label="final layer",
        vmin=0,
        vmax=max(1, num_layers - 1),
    ) if not only_mapping_matrix else None

    # Layer transition counts: j_layer -> final_layer
    if not only_mapping_matrix:
        trans = np.zeros((num_layers, num_layers), dtype=np.int32)
        for j in range(len(final)):
            jl = layer_of(j, num_heads)
            fl = layer_of(int(final[j]), num_heads)
            trans[jl, fl] += 1
        _plot_layer_transition(
            trans,
            plots_dir / f"{stem}__layer_transition_counts.png",
            title=f"{stem}: layer(j) -> layer(final(j)) counts",
        )

    # Histogram of per-column maxima (for directional matrices this is informative).
    if not only_mapping_matrix:
        _plot_hist(
            maxv,
            plots_dir / f"{stem}__max_value_hist.png",
            title=f"{stem}: per-column max similarity histogram",
            xlabel="max_i M[i, j]",
        )

    # Mapping matrix (like original similarity heatmaps): (i, j) marked if j -> i.
    _plot_mapping_matrix(
        final_thr,
        plots_dir / f"{stem}__final_mapping_matrix.png",
        title=(
            f"{stem}: final mapping matrix (i, j)=1 if j→i"
            + ("" if similarity_threshold is None else f", thr>={similarity_threshold}")
        ),
    )

    print(f"Wrote plots under {plots_dir}")


def main() -> None:
    args = parse_args()
    analysis_dir = Path(args.analysis_dir)
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = analysis_dir / ("mappings_excldiag" if args.exclude_diagonal else "mappings")

    num_layers, num_heads = load_summary(analysis_dir)
    N_expected = num_layers * num_heads
    plots_dir = (
        Path(args.plots_dir)
        if args.plots_dir
        else out_dir / "plots"
    )

    matrices = list(args.matrices)
    if args.also_binary and (analysis_dir / "M_AB_binary.npy").is_file():
        if "M_AB_binary.npy" not in matrices:
            matrices.append("M_AB_binary.npy")

    for name in matrices:
        p = analysis_dir / name
        if not p.is_file():
            raise FileNotFoundError(f"Missing matrix: {p}")
        M = np.load(p)
        if M.shape != (N_expected, N_expected):
            raise ValueError(
                f"{name}: expected {(N_expected, N_expected)} but got {M.shape}"
            )

        f, maxv = per_column_argmax(M, exclude_diagonal=args.exclude_diagonal)
        g, steps = integrate_mapping(f)
        final = apply_layer_constraint(g, num_heads=num_heads)
        final_thr, sim_final = apply_similarity_threshold(
            M, final, threshold=args.similarity_threshold
        )

        write_outputs(
            out_dir=out_dir,
            matrix_name=name,
            num_heads=num_heads,
            f=f,
            maxv=maxv,
            g=g,
            steps=steps,
            final=final,
            final_thr=final_thr,
            sim_final=sim_final,
            similarity_threshold=args.similarity_threshold,
        )

        if not args.no_plots:
            write_plots(
                plots_dir=plots_dir,
                matrix_name=name,
                num_layers=num_layers,
                num_heads=num_heads,
                f=f,
                maxv=maxv,
                g=g,
                final=final,
                final_thr=final_thr,
                only_mapping_matrix=args.only_mapping_matrix,
                similarity_threshold=args.similarity_threshold,
            )


if __name__ == "__main__":
    main()

