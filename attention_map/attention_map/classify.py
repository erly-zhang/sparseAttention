#!/usr/bin/env python3
"""
Classify attention patterns for saved full attention maps.

Implements a MInference-v1-like scoring:
  score(M) = mean_{q >= q_start} sum_k A[q,k] * M[q,k]

We compare three candidate mask families:
  - stream_llm (A-shape): left vertical + local band
  - vertical_and_slash: vertical columns (from last_q stats) + diagonal offsets (slash)
  - block_sparse: blockwise top-k (on pooled attention)

Input format: attention_map outputs with query_slice=all
  sample_dir/layer_{i:02d}/head_{j:02d}.npy  -> [L, L] float16/float32
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _load_square(sample_dir: Path, layer: int, head: int) -> np.ndarray:
    p = sample_dir / f"layer_{layer:02d}" / f"head_{head:02d}.npy"
    a = np.load(p)
    a = np.squeeze(a)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"Expected square [L,L] attention at {p}, got {a.shape}")
    return a.astype(np.float32, copy=False)


def _infer_dims(sample_dir: Path) -> Tuple[int, int, int]:
    layers = sorted(sample_dir.glob("layer_*"))
    if not layers:
        raise FileNotFoundError(f"No layer_* under {sample_dir}")
    num_layers = len(layers)
    heads = sorted(layers[0].glob("head_*.npy"))
    num_heads = len(heads)
    a00 = _load_square(sample_dir, 0, 0)
    L = int(a00.shape[0])
    return num_layers, num_heads, L


def _q_start(L: int, q_start: Optional[int]) -> int:
    if q_start is not None:
        return max(0, min(int(q_start), L - 1))
    # MInference hardcodes 2500; scale down for short sequences.
    return min(2500, L // 2)


def _dedupe_grid(grid: List[Tuple[int, int]], L: int) -> List[Tuple[int, int]]:
    seen: set[Tuple[int, int]] = set()
    out: List[Tuple[int, int]] = []
    for v, s in grid:
        v = max(1, min(int(v), L))
        s = max(1, min(int(s), L))
        key = (v, s)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _scaled_keep_limits(L: int) -> Tuple[int, int]:
    """Force-include first columns / local diagonals when picking VS mask."""
    k = max(4, min(30, L // 64))
    return k, k


def _vs_caps(L: int) -> Tuple[int, int]:
    """Upper bounds for vertical_and_slash grid and scoring."""
    L = int(L)
    return max(4, L // 16), max(8, L // 8)


def _stream_grid_for_length(L: int) -> List[Tuple[int, int]]:
    """
    A-shape (stream_llm) mask sizes scaled with L.

    Reference at 4k: v≈L/20 (~100), s≈L/5 (~800). Three tiers from tight to loose.
    """
    L = int(L)
    v_max, s_max = _vs_caps(L)
    candidates: List[Tuple[int, int]] = [
        (max(4, L // 64), max(8, min(L // 16, s_max))),
        (max(4, min(L // 32, v_max)), max(8, min(L // 8, s_max))),
        (max(4, min(L // 20, v_max)), max(8, min(L // 4, L))),
    ]
    if L >= 4096:
        candidates.append((min(100, v_max), min(800, L)))
    return _dedupe_grid(candidates, L)


def _vs_grid_for_length(L: int) -> List[Tuple[int, int]]:
    """vertical_and_slash mask sizes scaled with L (same base tiers as stream)."""
    L = int(L)
    v_max, s_max = _vs_caps(L)
    candidates: List[Tuple[int, int]] = [
        (max(4, L // 64), max(8, min(L // 16, s_max))),
        (max(4, min(L // 32, v_max)), max(8, min(L // 8, s_max))),
    ]
    if L >= 8192:
        candidates.extend(
            [
                (min(100, v_max), min(750, s_max)),
                (min(500, v_max), min(700, s_max)),
            ]
        )
    if L >= 32768:
        candidates.append((min(1000, v_max), min(4096, s_max)))
    return _dedupe_grid(candidates, L)


def default_grids_for_length(L: int) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]], int]:
    """Default classification grids; stream_llm and VS (v,s) pairs scale with seq_len L."""
    L = int(L)

    stream_candidates = _stream_grid_for_length(L)
    vs_candidates = _vs_grid_for_length(L)

    block_size = 32 if L >= 256 else max(8, L // 8)
    block_grid = [(block_size, 8)]
    last_q = min(64, max(16, L // 32))

    return stream_candidates, vs_candidates, block_grid, last_q


def legacy_grids_for_length(L: int) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]], int]:
    """Original fixed grids (MInference-style); loose on L < 8k."""
    L = int(L)
    return (
        [(100, 800)],
        _dedupe_grid([(30, 800), (100, 750), (500, 700), (1000, 4096)], L),
        [(32, 8)],
        64,
    )


def _causal_tril_inplace(A: np.ndarray) -> np.ndarray:
    """Ensure non-causal region contributes 0 to scores."""
    # A is already causal for typical models, but saved tensors may include 0s/garbage above diagonal.
    # Force upper-tri to 0 to keep scoring meaningful.
    iu = np.triu_indices(A.shape[0], k=1)
    A = A.copy()
    A[iu] = 0.0
    return A


def score_stream_llm(A: np.ndarray, q0: int, vertical_size: int, slash_size: int) -> float:
    """
    A-shape mask coverage: union of
      - left vertical_size columns
      - local band: (i - k) <= slash_size-1, k <= i
    """
    L = A.shape[0]
    v = max(0, min(int(vertical_size), L))
    s = max(1, min(int(slash_size), L))

    # Prefix sums per row for fast band sums.
    # row_cumsum[i, t] = sum_{k < t} A[i,k]
    row_cumsum = np.cumsum(A, axis=1)

    def row_sum(i: int, lo: int, hi: int) -> float:
        # sum A[i, lo:hi]
        if hi <= 0 or lo >= hi:
            return 0.0
        lo = max(0, lo)
        hi = min(L, hi)
        if lo == 0:
            return float(row_cumsum[i, hi - 1])
        return float(row_cumsum[i, hi - 1] - row_cumsum[i, lo - 1])

    total = 0.0
    count = 0
    for i in range(q0, L):
        # vertical part: columns [0, v)
        sv = row_sum(i, 0, v)
        # band part: columns [max(0, i-s+1), i+1]
        lo = max(0, i - s + 1)
        sb = row_sum(i, lo, i + 1)
        # overlap: band columns that are also < v
        ov = row_sum(i, lo, min(v, i + 1))
        total += (sv + sb - ov)
        count += 1
    return total / max(count, 1)


def _vertical_slash_stats_from_last_q(A: np.ndarray, last_q: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute:
      vertical_scores[k] = sum_{q in last_q rows} A[q,k]
      slash_scores[d] = sum_{q in last_q rows} A[q, q-d] for each diagonal offset d>=0
    """
    L = A.shape[0]
    last_q = max(1, min(int(last_q), L))
    q_start = L - last_q
    sub = A[q_start:, :]  # [last_q, L]
    vertical_scores = sub.sum(axis=0)  # [L]

    # diagonal offsets d = q-k (>=0). For last_q rows, offsets in [0, L-1].
    slash_scores = np.zeros((L,), dtype=np.float32)
    # Compute by iterating q rows in the submatrix; last_q <= 64 so this is cheap.
    for qi in range(q_start, L):
        row = A[qi]
        # keys k in [0, qi] (causal); offset d = qi-k
        # accumulate row[k] into slash_scores[qi-k]
        k = np.arange(0, qi + 1, dtype=np.int32)
        d = (qi - k).astype(np.int32)
        slash_scores[d] += row[k]
    return vertical_scores, slash_scores


def score_vertical_and_slash(
    A: np.ndarray,
    q0: int,
    vertical_size: int,
    slash_size: int,
    last_q: int = 64,
    keep_global: int = 30,
    keep_local: int = 30,
) -> Tuple[float, Dict[str, List[int]]]:
    """
    Mimic MInference mask construction conceptually, but score without building full [L,L] mask.

    - Select vertical columns: top vertical_size columns by vertical_scores,
      forcing first keep_global columns to be included.
    - Select slash diagonals: top slash_size diagonal offsets by slash_scores,
      forcing small offsets near 0..keep_local to be included (local diagonals).
    """
    L = A.shape[0]
    v_max, s_max = _vs_caps(L)
    v = max(0, min(int(vertical_size), L, v_max))
    s = max(1, min(int(slash_size), L, s_max))

    vertical_scores, slash_scores = _vertical_slash_stats_from_last_q(A, last_q=last_q)

    # force global columns
    keep_global = max(0, min(int(keep_global), L, v_max))
    forced_cols = set(range(keep_global))

    # pick remaining columns by score
    if v <= keep_global:
        cols = sorted(list(range(v)))
    else:
        # argsort descending
        order = np.argsort(-vertical_scores)
        picked = []
        for k in order:
            if k in forced_cols:
                continue
            picked.append(int(k))
            if len(picked) >= (v - keep_global):
                break
        cols = sorted(list(forced_cols) + picked)

    # force local diagonals (offsets 0..keep_local-1)
    keep_local = max(0, min(int(keep_local), L, s_max))
    forced_diags = set(range(keep_local))

    if s <= keep_local:
        diags = sorted(list(range(s)))
    else:
        order = np.argsort(-slash_scores)
        picked = []
        for d in order:
            if int(d) in forced_diags:
                continue
            picked.append(int(d))
            if len(picked) >= (s - keep_local):
                break
        diags = sorted(list(forced_diags) + picked)

    cols_set = set(cols)
    diags_set = set(diags)

    # Score union(cols, diags) per query row >= q0.
    # For each row i:
    #   vertical contribution: sum_{k in cols, k<=i} A[i,k]
    #   slash contribution: sum_{d in diags, i-d>=0} A[i, i-d]
    #   overlap: positions that are both (k in cols) and (d in diags where k=i-d)
    total = 0.0
    count = 0
    for i in range(q0, L):
        # vertical sum
        if cols:
            valid_cols = [k for k in cols if k <= i]
            sv = float(A[i, valid_cols].sum()) if valid_cols else 0.0
        else:
            sv = 0.0

        # slash sum
        ks = []
        for d in diags:
            k = i - d
            if k >= 0:
                ks.append(k)
        ss = float(A[i, ks].sum()) if ks else 0.0

        # overlap where k in cols AND (i-k) in diags
        ov = 0.0
        if cols and diags:
            for k in valid_cols:
                if (i - k) in diags_set:
                    ov += float(A[i, k])

        total += (sv + ss - ov)
        count += 1

    score = total / max(count, 1)
    return score, {"vertical_cols": cols, "slash_diags": diags}


def score_block_sparse(
    A: np.ndarray,
    q0: int,
    block_size: int = 32,
    topk_ratio: int = 8,
) -> Tuple[float, Dict[str, int]]:
    """
    Block-wise: pool A into blocks by sum, then for each block-row keep ~1/topk_ratio blocks.
    Score = sum of A in kept blocks (union), averaged over q>=q0.
    """
    L = A.shape[0]
    bs = max(1, int(block_size))
    nb = int(math.ceil(L / bs))
    # pad to multiple of bs
    pad = nb * bs - L
    if pad:
        Ap = np.pad(A, ((0, pad), (0, pad)), mode="constant", constant_values=0.0)
    else:
        Ap = A
    # pool into [nb, nb] by summing each bs×bs block
    # reshape to [nb, bs, nb, bs] then sum over bs dims
    pooled = Ap.reshape(nb, bs, nb, bs).sum(axis=(1, 3))  # [nb, nb]
    # causal: only consider blocks on/under diagonal
    pooled = np.tril(pooled)

    keep_per_row = max(1, nb // max(1, int(topk_ratio)))
    q0b = q0 // bs

    total = 0.0
    count = 0
    for br in range(q0b, nb):
        row = pooled[br]
        # indices of best blocks (largest pooled attention)
        # only among <= br (causal)
        cand = np.arange(0, br + 1)
        scores = row[cand]
        if scores.size == 0:
            continue
        top_idx = cand[np.argsort(-scores)[:keep_per_row]]
        # sum original A for rows in this block-row and cols in selected blocks
        r0, r1 = br * bs, min((br + 1) * bs, L)
        s = 0.0
        for bc in top_idx:
            c0, c1 = bc * bs, min((bc + 1) * bs, L)
            s += float(A[r0:r1, c0:c1].sum())
        # normalize by number of query rows in this block-row (to be comparable to per-row mean)
        nrows = max(1, r1 - r0)
        total += s / nrows
        count += 1

    score = total / max(count, 1)
    return score, {"block_size": bs, "topk_ratio": int(topk_ratio), "keep_blocks_per_row": keep_per_row}


@dataclass
class PatternResult:
    pattern: str
    score: float
    params: Dict


def classify_head(
    A: np.ndarray,
    q_start: Optional[int] = None,
    last_q: int = 64,
    stream_grid: List[Tuple[int, int]] = None,
    vs_grid: List[Tuple[int, int]] = None,
    block_grid: List[Tuple[int, int]] = None,
    keep_global: Optional[int] = None,
    keep_local: Optional[int] = None,
) -> List[PatternResult]:
    A = _causal_tril_inplace(A)
    L = A.shape[0]
    q0 = _q_start(L, q_start)

    kg, kl = _scaled_keep_limits(L)
    if keep_global is None:
        keep_global = kg
    if keep_local is None:
        keep_local = kl

    if stream_grid is None or vs_grid is None or block_grid is None:
        auto_stream, auto_vs, auto_block, _ = default_grids_for_length(L)
        if stream_grid is None:
            stream_grid = auto_stream
        if vs_grid is None:
            vs_grid = auto_vs
        if block_grid is None:
            block_grid = auto_block

    results: List[PatternResult] = []

    # stream_llm (A-shape)
    for v, s in stream_grid:
        score = score_stream_llm(A, q0=q0, vertical_size=v, slash_size=s)
        results.append(PatternResult("stream_llm", score, {"vertical_size": v, "slash_size": s, "q_start": q0}))

    # vertical_and_slash
    for v, s in vs_grid:
        score, detail = score_vertical_and_slash(
            A,
            q0=q0,
            vertical_size=v,
            slash_size=s,
            last_q=last_q,
            keep_global=keep_global,
            keep_local=keep_local,
        )
        params = {
            "vertical_size": v,
            "slash_size": s,
            "last_q": last_q,
            "q_start": q0,
            "keep_global": keep_global,
            "keep_local": keep_local,
            **detail,
        }
        results.append(PatternResult("vertical_and_slash", score, params))

    # block_sparse
    for bs, ratio in block_grid:
        score, detail = score_block_sparse(A, q0=q0, block_size=bs, topk_ratio=ratio)
        params = {"q_start": q0, **detail}
        results.append(PatternResult("block_sparse", score, params))

    results.sort(key=lambda r: r.score, reverse=True)
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Classify attention patterns for a saved sample_* directory")
    p.add_argument(
        "--sample_dir",
        type=str,
        default="/home/ubuntu/work/attention_map/outputs/4k/niah_single_1/sample_001179_line0000",
    )
    p.add_argument("--out", type=str, default=None, help="Output json path. Default: <sample_dir>/pattern_classification.json")
    p.add_argument("--layers", type=str, default=None, help="Comma-separated layer indices. Default: all")
    p.add_argument("--heads", type=str, default=None, help="Comma-separated head indices. Default: all")
    p.add_argument("--q_start", type=int, default=None, help="Override q_start (default: min(2500, L//2))")
    p.add_argument(
        "--last_q",
        type=int,
        default=None,
        help="Rows from the end used for VS stats (default: min(64, L//32))",
    )
    p.add_argument(
        "--legacy_grids",
        action="store_true",
        help="Use fixed (100,800) / (1000,4096) grids instead of length-scaled defaults",
    )
    return p.parse_args()


def _parse_csv_ints(s: Optional[str]) -> Optional[List[int]]:
    if not s:
        return None
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    args = parse_args()
    sample_dir = Path(args.sample_dir)
    num_layers, num_heads, L = _infer_dims(sample_dir)

    layers = _parse_csv_ints(args.layers) or list(range(num_layers))
    heads = _parse_csv_ints(args.heads) or list(range(num_heads))

    out_path = Path(args.out) if args.out else (sample_dir / "pattern_classification.json")

    if args.legacy_grids:
        stream_grid, vs_grid, block_grid, default_last_q = legacy_grids_for_length(L)
    else:
        stream_grid, vs_grid, block_grid, default_last_q = default_grids_for_length(L)
    last_q = int(args.last_q) if args.last_q is not None else default_last_q
    keep_global, keep_local = _scaled_keep_limits(L)
    vs_v_max, vs_s_max = _vs_caps(L)
    if args.legacy_grids:
        keep_global, keep_local = 30, 30

    summary = {
        "sample_dir": str(sample_dir),
        "seq_len": L,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "q_start": _q_start(L, args.q_start),
        "last_q": last_q,
        "legacy_grids": bool(args.legacy_grids),
        "keep_global": keep_global,
        "keep_local": keep_local,
        "vs_v_max": vs_v_max,
        "vs_s_max": vs_s_max,
        "stream_grid": stream_grid,
        "vs_grid": vs_grid,
        "block_grid": block_grid,
        "results": {},
    }

    for layer in layers:
        layer_key = f"{layer:02d}"
        summary["results"][layer_key] = {}
        for head in heads:
            A = _load_square(sample_dir, layer, head)
            ranked = classify_head(
                A,
                q_start=args.q_start,
                last_q=last_q,
                stream_grid=stream_grid,
                vs_grid=vs_grid,
                block_grid=block_grid,
                keep_global=keep_global,
                keep_local=keep_local,
            )
            summary["results"][layer_key][str(head)] = {
                "best": asdict(ranked[0]),
                "ranked": [asdict(r) for r in ranked],
            }

    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

