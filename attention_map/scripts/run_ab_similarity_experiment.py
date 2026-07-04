#!/usr/bin/env python3
"""
Run the A-A / B-B / A-B attention similarity experiment.

Experiment definition:
1. Use the same K%-mass coverage rule as `attention_map.similarity`.
2. Use only the last query row (last token), q = seq_len - 1.
3. Compute three directional similarity matrices:
   - A-A
   - B-B
   - A-B
4. For each attention head, compute sparsity under K% selection:
   - average selected tokens per query row
   - average selected ratio per query row
   - min/max selected tokens across query rows
5. Threshold each similarity matrix into binary masks:
   1 if entry >= threshold, else 0
6. Compute pairwise XNOR similarity between those binary matrices.

This script is designed for large full attention maps and avoids loading all
heads into RAM at once.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.sparse import csr_matrix

from attention_map.similarity import (
    keys_for_k_percent_mass,
    plot_similarity_heatmap,
)


def infer_sample_dims_flex(sample_dir: Path) -> Tuple[int, int, int, bool]:
    """Infer (num_layers, num_heads, seq_len, is_last_row) for a sample dir.

    Supports two on-disk formats per head file:
      * full square map  [seq_len, seq_len]   (is_last_row=False)
      * last-token row    [1, seq_len] or [seq_len]  (is_last_row=True)
    Since this experiment only uses the last query row anyway, both work.
    """
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


def _query_row(attn: np.ndarray, q: int, is_last_row: bool) -> np.ndarray:
    """Return attention row for query position q (keys 0..seq_len-1)."""
    if is_last_row:
        return np.asarray(attn).reshape(-1)
    return attn[q]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="A-A / B-B / A-B similarity experiment on two full-attention samples"
    )
    p.add_argument(
        "--sample_a_dir",
        type=str,
        default=(
            "/home/ubuntu/work/attention_map/outputs_longbench_v2/"
            "code_repository_understanding/code_repo_qa/"
            "sample_66fa208bbb02136c067c5fc1_line0007"
        ),
        help="Sample A directory with layer_XX/head_YY.npy",
    )
    p.add_argument(
        "--sample_b_dir",
        type=str,
        default=(
            "/home/ubuntu/work/attention_map/outputs_longbench_v2/"
            "single_document_qa/financial/"
            "sample_66f36490821e116aacb2cc22_line0001"
        ),
        help="Sample B directory with layer_XX/head_YY.npy",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="/home/ubuntu/work/attention_map/analysis/ab_similarity_experiment_k95",
        help="Output directory for matrices, heatmaps, and summaries",
    )
    p.add_argument(
        "--k_percent",
        type=float,
        default=95.0,
        help="K%% mass coverage (default 95)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Threshold for binary masking (default 0.9)",
    )
    p.add_argument(
        "--block_size",
        type=int,
        default=16,
        help="Number of target heads loaded per block (default 16, safer on 16GB RAM)",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Heatmap DPI (default 200)",
    )
    p.add_argument(
        "--no_heatmaps",
        action="store_true",
        help="Skip PNG heatmap export",
    )
    p.add_argument(
        "--sparsity-only",
        action="store_true",
        help="Only rebuild per_head_sparsity.json (skip similarity matrices)",
    )
    return p.parse_args()


def head_path(sample_dir: Path, flat_idx: int, num_heads: int) -> Path:
    layer = flat_idx // num_heads
    head = flat_idx % num_heads
    return sample_dir / f"layer_{layer:02d}" / f"head_{head:02d}.npy"


def build_sparse_masks(
    sample_dir: Path,
    *,
    num_layers: int,
    num_heads: int,
    seq_len: int,
    q0: int,
    k_percent: float,
    is_last_row: bool = False,
) -> Tuple[csr_matrix, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a CSR matrix S of shape [N_heads, n_q * L], where each row encodes the
    selected (query,row key) positions under the K% mass rule.

    S[i] @ target_flattened_head gives the sum of raw target attention on all
    selected positions across all query rows. Dividing by n_q yields the mean.
    """
    n_q = seq_len - q0
    n_heads_total = num_layers * num_heads
    flattened_width = n_q * seq_len

    data_parts: List[np.ndarray] = []
    index_parts: List[np.ndarray] = []
    indptr = [0]

    avg_counts = np.zeros(n_heads_total, dtype=np.float32)
    avg_ratios = np.zeros(n_heads_total, dtype=np.float32)
    min_counts = np.zeros(n_heads_total, dtype=np.int32)
    max_counts = np.zeros(n_heads_total, dtype=np.int32)
    total_counts = np.zeros(n_heads_total, dtype=np.int64)

    for flat_idx in range(n_heads_total):
        attn = np.load(head_path(sample_dir, flat_idx, num_heads), mmap_mode="r")
        per_row_indices: List[np.ndarray] = []
        counts = np.zeros(n_q, dtype=np.int32)
        ratios = np.zeros(n_q, dtype=np.float32)

        for q_idx, q in enumerate(range(q0, seq_len)):
            keys = keys_for_k_percent_mass(
                _query_row(attn, q, is_last_row), q, k_percent
            ).astype(np.int32, copy=False)
            per_row_indices.append(q_idx * seq_len + keys)
            counts[q_idx] = keys.size
            ratios[q_idx] = keys.size / float(q + 1)

        flat_indices = (
            np.concatenate(per_row_indices)
            if per_row_indices
            else np.empty((0,), dtype=np.int32)
        )
        data_parts.append(np.ones(flat_indices.shape[0], dtype=np.float32))
        index_parts.append(flat_indices)
        indptr.append(indptr[-1] + flat_indices.shape[0])

        avg_counts[flat_idx] = float(counts.mean())
        avg_ratios[flat_idx] = float(ratios.mean())
        min_counts[flat_idx] = int(counts.min())
        max_counts[flat_idx] = int(counts.max())
        total_counts[flat_idx] = int(counts.sum())

        if flat_idx % 64 == 0:
            print(f"[build] {sample_dir.name}: head {flat_idx}/{n_heads_total}", flush=True)

    data = np.concatenate(data_parts)
    indices = np.concatenate(index_parts)
    indptr_arr = np.asarray(indptr, dtype=np.int64)
    sparse_masks = csr_matrix(
        (data, indices, indptr_arr),
        shape=(n_heads_total, flattened_width),
        dtype=np.float32,
    )
    return sparse_masks, avg_counts, avg_ratios, min_counts, max_counts, total_counts


def load_target_block(
    sample_dir: Path,
    *,
    start: int,
    end: int,
    num_heads: int,
    seq_len: int,
    q0: int,
    is_last_row: bool = False,
) -> np.ndarray:
    """Load a block of target heads as flattened [block, n_q * L] raw attention values."""
    rows = []
    for flat_idx in range(start, end):
        attn = np.load(head_path(sample_dir, flat_idx, num_heads), mmap_mode="r").astype(
            np.float32, copy=False
        )
        if is_last_row:
            rows.append(np.asarray(attn).reshape(-1))
        else:
            rows.append(attn[q0:, :].reshape(-1))
    return np.stack(rows, axis=0)


def compute_directional_matrix(
    sparse_masks: csr_matrix,
    *,
    target_dir: Path,
    num_layers: int,
    num_heads: int,
    seq_len: int,
    q0: int,
    block_size: int,
    name: str,
    is_last_row: bool = False,
) -> np.ndarray:
    n_heads_total = num_layers * num_heads
    n_q = seq_len - q0
    matrix = np.zeros((n_heads_total, n_heads_total), dtype=np.float32)

    for start in range(0, n_heads_total, block_size):
        end = min(n_heads_total, start + block_size)
        block = load_target_block(
            target_dir,
            start=start,
            end=end,
            num_heads=num_heads,
            seq_len=seq_len,
            q0=q0,
            is_last_row=is_last_row,
        )
        print(f"[compute {name}] target heads {start}:{end}", flush=True)
        matrix[:, start:end] = (sparse_masks @ block.T) / float(n_q)
        del block
    return matrix


def xnor_similarity(mask_x: np.ndarray, mask_y: np.ndarray) -> float:
    return float(np.mean(mask_x == mask_y))


def top_heads(per_head: List[Dict], reverse: bool) -> List[Dict]:
    return sorted(
        per_head,
        key=lambda item: item["avg_selected_tokens"],
        reverse=reverse,
    )[:5]


def build_per_head_sparsity_payload(
    *,
    args: argparse.Namespace,
    seq_len: int,
    q0: int,
    num_heads: int,
    n_heads_total: int,
    a_avg_counts: np.ndarray,
    a_avg_ratios: np.ndarray,
    a_min_counts: np.ndarray,
    a_max_counts: np.ndarray,
    a_total_counts: np.ndarray,
    b_avg_counts: np.ndarray,
    b_avg_ratios: np.ndarray,
    b_min_counts: np.ndarray,
    b_max_counts: np.ndarray,
    b_total_counts: np.ndarray,
) -> Dict:
    per_head_a = [
        {
            "flat_head": int(i),
            "layer": int(i // num_heads),
            "head": int(i % num_heads),
            "avg_selected_tokens": float(a_avg_counts[i]),
            "avg_selected_ratio": float(a_avg_ratios[i]),
            "min_selected_tokens": int(a_min_counts[i]),
            "max_selected_tokens": int(a_max_counts[i]),
            "total_selected_tokens": int(a_total_counts[i]),
        }
        for i in range(n_heads_total)
    ]
    per_head_b = [
        {
            "flat_head": int(i),
            "layer": int(i // num_heads),
            "head": int(i % num_heads),
            "avg_selected_tokens": float(b_avg_counts[i]),
            "avg_selected_ratio": float(b_avg_ratios[i]),
            "min_selected_tokens": int(b_min_counts[i]),
            "max_selected_tokens": int(b_max_counts[i]),
            "total_selected_tokens": int(b_total_counts[i]),
        }
        for i in range(n_heads_total)
    ]
    n_q = seq_len - q0
    return {
        "meta": {
            "k_percent": float(args.k_percent),
            "q_start": int(q0),
            "seq_len": int(seq_len),
            "num_query_rows": int(n_q),
            "num_heads_total": int(n_heads_total),
            "note": (
                "total_selected_tokens per head = sum over query rows of "
                "keys selected by k% mass rule (each row counts separately)."
            ),
        },
        "summary": {
            "A": {
                "total_selected_tokens_all_heads": int(a_total_counts.sum()),
                "total_selected_tokens_mean_per_head": float(a_total_counts.mean()),
            },
            "B": {
                "total_selected_tokens_all_heads": int(b_total_counts.sum()),
                "total_selected_tokens_mean_per_head": float(b_total_counts.mean()),
            },
        },
        "A": per_head_a,
        "B": per_head_b,
    }


def write_per_head_sparsity(out_dir: Path, payload: Dict) -> None:
    (out_dir / "per_head_sparsity.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def save_binary_heatmap(matrix: np.ndarray, out_png: Path, *, title: str, num_heads: int, dpi: int) -> None:
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
    sample_a_dir = Path(args.sample_a_dir)
    sample_b_dir = Path(args.sample_b_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    a_layers, a_heads, a_len, a_last_row = infer_sample_dims_flex(sample_a_dir)
    b_layers, b_heads, b_len, b_last_row = infer_sample_dims_flex(sample_b_dir)

    if (a_layers, a_heads) != (b_layers, b_heads):
        raise ValueError(
            f"Layer/head mismatch: A=({a_layers},{a_heads}), B=({b_layers},{b_heads})"
        )
    if a_len != b_len:
        raise ValueError(
            f"Sequence length mismatch: A={a_len}, B={b_len}. "
            "Please use aligned attention maps (same seq_len)."
        )

    seq_len = a_len
    q0 = seq_len - 1  # last query row only (last token)
    num_layers = a_layers
    num_heads = a_heads
    n_heads_total = num_layers * num_heads
    print(
        f"A format: {'last-row' if a_last_row else 'full-map'}, "
        f"B format: {'last-row' if b_last_row else 'full-map'}, seq_len={seq_len}",
        flush=True,
    )

    print("Building sparse masks for A...", flush=True)
    sparse_a, a_avg_counts, a_avg_ratios, a_min_counts, a_max_counts, a_total_counts = (
        build_sparse_masks(
            sample_a_dir,
            num_layers=num_layers,
            num_heads=num_heads,
            seq_len=seq_len,
            q0=q0,
            k_percent=args.k_percent,
            is_last_row=a_last_row,
        )
    )

    if args.sparsity_only:
        del sparse_a
        print("Building sparse masks for B...", flush=True)
        _, b_avg_counts, b_avg_ratios, b_min_counts, b_max_counts, b_total_counts = (
            build_sparse_masks(
                sample_b_dir,
                num_layers=num_layers,
                num_heads=num_heads,
                seq_len=seq_len,
                q0=q0,
                k_percent=args.k_percent,
                is_last_row=b_last_row,
            )
        )
        payload = build_per_head_sparsity_payload(
            args=args,
            seq_len=seq_len,
            q0=q0,
            num_heads=num_heads,
            n_heads_total=n_heads_total,
            a_avg_counts=a_avg_counts,
            a_avg_ratios=a_avg_ratios,
            a_min_counts=a_min_counts,
            a_max_counts=a_max_counts,
            a_total_counts=a_total_counts,
            b_avg_counts=b_avg_counts,
            b_avg_ratios=b_avg_ratios,
            b_min_counts=b_min_counts,
            b_max_counts=b_max_counts,
            b_total_counts=b_total_counts,
        )
        write_per_head_sparsity(out_dir, payload)
        summary_path = out_dir / "summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["sparsity"] = {
                "num_query_rows": int(seq_len - q0),
                "A": {
                    **payload["summary"]["A"],
                    "avg_selected_tokens_mean": float(a_avg_counts.mean()),
                    "total_selected_tokens_std_per_head": float(a_total_counts.std()),
                },
                "B": {
                    **payload["summary"]["B"],
                    "avg_selected_tokens_mean": float(b_avg_counts.mean()),
                    "total_selected_tokens_std_per_head": float(b_total_counts.std()),
                },
            }
            summary_path.write_text(
                json.dumps(summary, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
        print(f"Wrote per_head_sparsity.json under {out_dir}")
        return

    print("Computing A-A...", flush=True)
    m_aa = compute_directional_matrix(
        sparse_a,
        target_dir=sample_a_dir,
        num_layers=num_layers,
        num_heads=num_heads,
        seq_len=seq_len,
        q0=q0,
        block_size=args.block_size,
        name="A-A",
        is_last_row=a_last_row,
    )
    print("Computing A-B...", flush=True)
    m_ab = compute_directional_matrix(
        sparse_a,
        target_dir=sample_b_dir,
        num_layers=num_layers,
        num_heads=num_heads,
        seq_len=seq_len,
        q0=q0,
        block_size=args.block_size,
        name="A-B",
        is_last_row=b_last_row,
    )
    del sparse_a

    print("Building sparse masks for B...", flush=True)
    sparse_b, b_avg_counts, b_avg_ratios, b_min_counts, b_max_counts, b_total_counts = (
        build_sparse_masks(
            sample_b_dir,
            num_layers=num_layers,
            num_heads=num_heads,
            seq_len=seq_len,
            q0=q0,
            k_percent=args.k_percent,
            is_last_row=b_last_row,
        )
    )
    print("Computing B-B...", flush=True)
    m_bb = compute_directional_matrix(
        sparse_b,
        target_dir=sample_b_dir,
        num_layers=num_layers,
        num_heads=num_heads,
        seq_len=seq_len,
        q0=q0,
        block_size=args.block_size,
        name="B-B",
        is_last_row=b_last_row,
    )
    del sparse_b

    np.save(out_dir / "M_AA_directional.npy", m_aa)
    np.save(out_dir / "M_AB_directional.npy", m_ab)
    np.save(out_dir / "M_BB_directional.npy", m_bb)

    if not args.no_heatmaps:
        plot_similarity_heatmap(
            m_aa,
            out_dir / "M_AA_directional.png",
            title=f"A-A directional similarity, K={args.k_percent}%, q>={q0}",
            num_heads=num_heads,
            dpi=args.dpi,
        )
        plot_similarity_heatmap(
            m_ab,
            out_dir / "M_AB_directional.png",
            title=f"A-B directional similarity, K={args.k_percent}%, q>={q0}",
            num_heads=num_heads,
            dpi=args.dpi,
        )
        plot_similarity_heatmap(
            m_bb,
            out_dir / "M_BB_directional.png",
            title=f"B-B directional similarity, K={args.k_percent}%, q>={q0}",
            num_heads=num_heads,
            dpi=args.dpi,
        )

    b_aa = (m_aa >= args.threshold).astype(np.uint8)#二值化
    b_ab = (m_ab >= args.threshold).astype(np.uint8)
    b_bb = (m_bb >= args.threshold).astype(np.uint8)

    np.save(out_dir / "M_AA_binary.npy", b_aa)
    np.save(out_dir / "M_AB_binary.npy", b_ab)
    np.save(out_dir / "M_BB_binary.npy", b_bb)

    if not args.no_heatmaps:
        save_binary_heatmap(
            b_aa,
            out_dir / "M_AA_binary.png",
            title=f"A-A binary mask >= {args.threshold}",
            num_heads=num_heads,
            dpi=args.dpi,
        )
        save_binary_heatmap(
            b_ab,
            out_dir / "M_AB_binary.png",
            title=f"A-B binary mask >= {args.threshold}",
            num_heads=num_heads,
            dpi=args.dpi,
        )
        save_binary_heatmap(
            b_bb,
            out_dir / "M_BB_binary.png",
            title=f"B-B binary mask >= {args.threshold}",
            num_heads=num_heads,
            dpi=args.dpi,
        )

    per_head_payload = build_per_head_sparsity_payload(
        args=args,
        seq_len=seq_len,
        q0=q0,
        num_heads=num_heads,
        n_heads_total=n_heads_total,
        a_avg_counts=a_avg_counts,
        a_avg_ratios=a_avg_ratios,
        a_min_counts=a_min_counts,
        a_max_counts=a_max_counts,
        a_total_counts=a_total_counts,
        b_avg_counts=b_avg_counts,
        b_avg_ratios=b_avg_ratios,
        b_min_counts=b_min_counts,
        b_max_counts=b_max_counts,
        b_total_counts=b_total_counts,
    )
    per_head_a = per_head_payload["A"]
    per_head_b = per_head_payload["B"]

    xnor_aa_bb = xnor_similarity(b_aa, b_bb)
    xnor_aa_ab = xnor_similarity(b_aa, b_ab)
    xnor_bb_ab = xnor_similarity(b_bb, b_ab)

    summary = {
        "sample_a_dir": str(sample_a_dir),
        "sample_b_dir": str(sample_b_dir),
        "out_dir": str(out_dir),
        "k_percent": float(args.k_percent),
        "threshold": float(args.threshold),
        "seq_len": int(seq_len),
        "q_start": int(q0),
        "num_layers": int(num_layers),
        "num_heads": int(num_heads),
        "global_heads": int(n_heads_total),
        "matrix_stats": {
            "AA": {
                "mean": float(m_aa.mean()),
                "std": float(m_aa.std()),
                "min": float(m_aa.min()),
                "max": float(m_aa.max()),
                "binary_density": float(b_aa.mean()),
            },
            "BB": {
                "mean": float(m_bb.mean()),
                "std": float(m_bb.std()),
                "min": float(m_bb.min()),
                "max": float(m_bb.max()),
                "binary_density": float(b_bb.mean()),
            },
            "AB": {
                "mean": float(m_ab.mean()),
                "std": float(m_ab.std()),
                "min": float(m_ab.min()),
                "max": float(m_ab.max()),
                "binary_density": float(b_ab.mean()),
            },
        },
        "xnor_similarity": {
            "AA_vs_BB": xnor_aa_bb,
            "AA_vs_AB": xnor_aa_ab,
            "BB_vs_AB": xnor_bb_ab,
        },
        "expectation_check": {
            "AA_vs_BB_gt_AA_vs_AB": bool(xnor_aa_bb > xnor_aa_ab),
            "AA_vs_BB_gt_BB_vs_AB": bool(xnor_aa_bb > xnor_bb_ab),
        },
        "sparsity": {
            "num_query_rows": int(seq_len - q0),
            "A": {
                "avg_selected_tokens_mean": float(a_avg_counts.mean()),
                "avg_selected_tokens_std": float(a_avg_counts.std()),
                "avg_selected_ratio_mean": float(a_avg_ratios.mean()),
                "avg_selected_ratio_std": float(a_avg_ratios.std()),
                "total_selected_tokens_all_heads": int(a_total_counts.sum()),
                "total_selected_tokens_mean_per_head": float(a_total_counts.mean()),
                "total_selected_tokens_std_per_head": float(a_total_counts.std()),
                "top5_dense_heads_by_count": top_heads(per_head_a, reverse=True),
                "top5_sparse_heads_by_count": top_heads(per_head_a, reverse=False),
            },
            "B": {
                "avg_selected_tokens_mean": float(b_avg_counts.mean()),
                "avg_selected_tokens_std": float(b_avg_counts.std()),
                "avg_selected_ratio_mean": float(b_avg_ratios.mean()),
                "avg_selected_ratio_std": float(b_avg_ratios.std()),
                "total_selected_tokens_all_heads": int(b_total_counts.sum()),
                "total_selected_tokens_mean_per_head": float(b_total_counts.mean()),
                "total_selected_tokens_std_per_head": float(b_total_counts.std()),
                "top5_dense_heads_by_count": top_heads(per_head_b, reverse=True),
                "top5_sparse_heads_by_count": top_heads(per_head_b, reverse=False),
            },
        },
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_per_head_sparsity(out_dir, per_head_payload)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote experiment outputs under {out_dir}")


if __name__ == "__main__":
    main()
