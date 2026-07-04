#!/usr/bin/env python3
"""
Classify saved full attention maps into three MInference-like patterns.

Input layout:
  sample_dir/layer_00/head_00.npy  -> [L, L]
  sample_dir/layer_00/head_01.npy  -> [L, L]
  ...

Patterns:
  1. stream_llm / A-shape:
     fixed initial columns + local causal band.
  2. vertical_and_slash:
     dynamic important columns + dynamic diagonal offsets estimated from last_q rows.
  3. block_sparse:
     block-level top-k blocks per block row.

Important changes from the earlier version:
  - Default grids are scaled for short sequences such as L=1892.
  - Very large VS candidates such as (1000, 4096) are removed by default.
  - VS candidates are capped so they cannot cover almost the whole causal map.
  - The script records score_gap = best_score - second_best_score.
  - Optional --min_gap can mark low-confidence heads.

Example:
  python classify_modified.py \
    --sample_dir /path/to/sample_000 \
    --out /path/to/pattern_classification.json

Legacy behavior:
  python classify_modified.py --sample_dir /path/to/sample --legacy_grids
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# -----------------------------
# IO and basic utilities
# -----------------------------

def load_square(sample_dir: Path, layer: int, head: int) -> np.ndarray:
    path = sample_dir / f"layer_{layer:02d}" / f"head_{head:02d}.npy"
    arr = np.squeeze(np.load(path))
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Expected square [L,L] attention at {path}, got {arr.shape}")
    return arr.astype(np.float32, copy=False)


def infer_dims(sample_dir: Path) -> Tuple[int, int, int]:
    layer_dirs = sorted(sample_dir.glob("layer_*"))
    if not layer_dirs:
        raise FileNotFoundError(f"No layer_* directories found under {sample_dir}")

    # Use layer_00 if available, otherwise first sorted layer directory.
    first_layer = sample_dir / "layer_00"
    if not first_layer.exists():
        first_layer = layer_dirs[0]

    head_files = sorted(first_layer.glob("head_*.npy"))
    if not head_files:
        raise FileNotFoundError(f"No head_*.npy files found under {first_layer}")

    num_layers = len(layer_dirs)
    num_heads = len(head_files)
    arr0 = np.squeeze(np.load(head_files[0]))
    if arr0.ndim != 2 or arr0.shape[0] != arr0.shape[1]:
        raise ValueError(f"Expected square [L,L] attention at {head_files[0]}, got {arr0.shape}")
    return num_layers, num_heads, int(arr0.shape[0])


def parse_csv_ints(value: Optional[str]) -> Optional[List[int]]:
    if not value:
        return None
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_grid(value: Optional[str]) -> Optional[List[Tuple[int, int]]]:
    """Parse grids like '32:128,50:256,64:384'."""
    if not value:
        return None
    out: List[Tuple[int, int]] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Grid item must be v:s or a:b, got {item!r}")
        left, right = item.split(":", 1)
        out.append((int(left), int(right)))
    return out


def q_start_for_length(L: int, q_start: Optional[int]) -> int:
    if q_start is not None:
        return max(0, min(int(q_start), L - 1))
    return min(2500, L // 2)


def causal_tril(A: np.ndarray) -> np.ndarray:
    """Return a copy with the non-causal upper triangle set to zero."""
    B = A.astype(np.float32, copy=True)
    B[np.triu_indices(B.shape[0], k=1)] = 0.0
    return B


def dedupe_and_clip_grid(grid: Sequence[Tuple[int, int]], L: int) -> List[Tuple[int, int]]:
    seen: set[Tuple[int, int]] = set()
    out: List[Tuple[int, int]] = []
    for a, b in grid:
        a = max(1, min(int(a), L))
        b = max(1, min(int(b), L))
        key = (a, b)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


# -----------------------------
# Length-aware default parameters
# -----------------------------

def scaled_keep_limits(L: int) -> Tuple[int, int]:
    """
    Forced global columns and forced local diagonals for VS.

    Earlier keep_global=30 and keep_local=30 were acceptable for long contexts,
    but for short L they should be scaled down.
    """
    keep = max(4, min(30, L // 64))
    return keep, keep


def vs_caps(L: int) -> Tuple[int, int]:
    """
    Upper bounds for VS candidates.

    This prevents a candidate like slash_size=L from covering almost the full causal map.
    For L=1892: v_cap=118, s_cap=236.
    """
    return max(8, L // 16), max(16, L // 8)


def default_grids_for_length(L: int) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]], int]:
    """Return stream_grid, vs_grid, block_grid, last_q for the sequence length."""
    # For L≈1892 this gives (32,128), (50,256), (64,384), which is much safer than (100,800).
    stream_grid = [
        (max(8, min(32, L // 32)), max(32, min(128, L // 16))),
        (max(16, min(50, L // 24)), max(64, min(256, L // 8))),
        (max(24, min(64, L // 20)), max(96, min(384, L // 5))),
    ]

    v_cap, s_cap = vs_caps(L)
    vs_grid = [
        (max(4, min(16, v_cap)), max(32, min(128, s_cap))),
        (max(8, min(32, v_cap)), max(64, min(256, s_cap))),
        (max(16, min(64, v_cap)), max(96, min(384, s_cap))),
        (max(32, min(128, v_cap)), max(128, min(512, s_cap))),
    ]

    # Use 32 for normal maps; for very short maps use smaller blocks.
    block_size = 32 if L >= 512 else max(8, L // 16)
    # topk_ratio=8 keeps around 12.5% block columns per block row.
    block_grid = [(block_size, 8)]

    # MInference uses 64; for short sequences, scale modestly but keep enough rows for stable stats.
    last_q = min(64, max(16, L // 32))

    return (
        dedupe_and_clip_grid(stream_grid, L),
        dedupe_and_clip_grid(vs_grid, L),
        dedupe_and_clip_grid(block_grid, L),
        last_q,
    )


def legacy_grids_for_length(L: int) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]], int]:
    """The original loose grids, kept only for comparison."""
    return (
        dedupe_and_clip_grid([(100, 800)], L),
        dedupe_and_clip_grid([(30, 800), (100, 750), (500, 700), (1000, 4096)], L),
        dedupe_and_clip_grid([(32, 8)], L),
        64,
    )


# -----------------------------
# Scoring functions
# -----------------------------

def score_stream_llm(A: np.ndarray, q0: int, vertical_size: int, slash_size: int) -> float:
    """
    A-shape / StreamLLM coverage:
      M(q,k)=1 if k < vertical_size or q-slash_size+1 <= k <= q.
    """
    L = A.shape[0]
    v = max(0, min(int(vertical_size), L))
    s = max(1, min(int(slash_size), L))

    row_cumsum = np.cumsum(A, axis=1)

    def row_sum(i: int, lo: int, hi: int) -> float:
        lo = max(0, min(lo, L))
        hi = max(0, min(hi, L))
        if hi <= lo:
            return 0.0
        if lo == 0:
            return float(row_cumsum[i, hi - 1])
        return float(row_cumsum[i, hi - 1] - row_cumsum[i, lo - 1])

    total = 0.0
    nrows = 0
    for q in range(q0, L):
        local_lo = max(0, q - s + 1)
        local_hi = q + 1
        vertical_mass = row_sum(q, 0, v)
        local_mass = row_sum(q, local_lo, local_hi)
        overlap_mass = row_sum(q, local_lo, min(v, local_hi))
        total += vertical_mass + local_mass - overlap_mass
        nrows += 1

    return total / max(nrows, 1)


def vertical_slash_stats_from_last_q(A: np.ndarray, last_q: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate important vertical columns and slash diagonal offsets from the last_q rows.

    vertical_scores[k] = sum_q A[q,k]
    slash_scores[d] = sum_q A[q,q-d], d>=0
    """
    L = A.shape[0]
    last_q = max(1, min(int(last_q), L))
    q_begin = L - last_q

    sub = A[q_begin:, :]
    vertical_scores = sub.sum(axis=0).astype(np.float32)

    slash_scores = np.zeros(L, dtype=np.float32)
    for q in range(q_begin, L):
        row = A[q, : q + 1]
        # row[k] contributes to offset d=q-k, so reversed row maps k=q,q-1,...,0 to d=0,1,...,q.
        slash_scores[: q + 1] += row[::-1]

    return vertical_scores, slash_scores


def pick_top_with_forced(scores: np.ndarray, total_keep: int, forced: Iterable[int]) -> List[int]:
    L = scores.shape[0]
    total_keep = max(1, min(int(total_keep), L))
    forced_set = {int(x) for x in forced if 0 <= int(x) < L}

    if total_keep <= len(forced_set):
        return sorted(list(forced_set))[:total_keep]

    picked: List[int] = []
    order = np.argsort(-scores)
    for idx in order:
        idx = int(idx)
        if idx in forced_set:
            continue
        picked.append(idx)
        if len(picked) >= total_keep - len(forced_set):
            break

    return sorted(list(forced_set) + picked)


def score_vertical_and_slash(
    A: np.ndarray,
    q0: int,
    vertical_size: int,
    slash_size: int,
    *,
    last_q: int,
    keep_global: int,
    keep_local: int,
    cap_candidates: bool = True,
) -> Tuple[float, Dict[str, object]]:
    """
    Vertical-Slash coverage.

    Columns and diagonal offsets are selected dynamically from last_q rows, then scored on q>=q0.
    """
    L = A.shape[0]
    v = max(1, min(int(vertical_size), L))
    s = max(1, min(int(slash_size), L))

    if cap_candidates:
        v_cap, s_cap = vs_caps(L)
        v = min(v, v_cap)
        s = min(s, s_cap)

    keep_global = max(0, min(int(keep_global), v, L))
    keep_local = max(0, min(int(keep_local), s, L))

    vertical_scores, slash_scores = vertical_slash_stats_from_last_q(A, last_q)
    cols = pick_top_with_forced(vertical_scores, v, range(keep_global))
    diags = pick_top_with_forced(slash_scores, s, range(keep_local))

    cols_arr = np.asarray(cols, dtype=np.int32)
    diags_arr = np.asarray(diags, dtype=np.int32)
    diag_set = set(int(x) for x in diags)

    total = 0.0
    nrows = 0
    for q in range(q0, L):
        valid_cols = cols_arr[cols_arr <= q]
        vertical_mass = float(A[q, valid_cols].sum()) if valid_cols.size else 0.0

        keys_from_diags = q - diags_arr
        keys_from_diags = keys_from_diags[keys_from_diags >= 0]
        slash_mass = float(A[q, keys_from_diags].sum()) if keys_from_diags.size else 0.0

        overlap_mass = 0.0
        for k in valid_cols:
            if (q - int(k)) in diag_set:
                overlap_mass += float(A[q, int(k)])

        total += vertical_mass + slash_mass - overlap_mass
        nrows += 1

    return total / max(nrows, 1), {
        "vertical_size_used": int(v),
        "slash_size_used": int(s),
        "keep_global": int(keep_global),
        "keep_local": int(keep_local),
        "vertical_cols": [int(x) for x in cols],
        "slash_diags": [int(x) for x in diags],
    }


def score_block_sparse(A: np.ndarray, q0: int, block_size: int, topk_ratio: int) -> Tuple[float, Dict[str, int]]:
    """
    Block-wise coverage.

    Pool A into block sums. For each block row, keep the top blocks among causal block columns.
    """
    L = A.shape[0]
    bs = max(1, min(int(block_size), L))
    ratio = max(1, int(topk_ratio))
    nb = int(math.ceil(L / bs))
    pad = nb * bs - L

    if pad:
        Ap = np.pad(A, ((0, pad), (0, pad)), mode="constant", constant_values=0.0)
    else:
        Ap = A

    pooled = Ap.reshape(nb, bs, nb, bs).sum(axis=(1, 3))
    pooled = np.tril(pooled)

    keep_per_row = max(1, nb // ratio)
    q0_block = q0 // bs

    total = 0.0
    n_block_rows = 0
    for br in range(q0_block, nb):
        cand = np.arange(0, br + 1, dtype=np.int32)
        row_scores = pooled[br, cand]
        if row_scores.size == 0:
            continue
        top_blocks = cand[np.argsort(-row_scores)[:keep_per_row]]

        r0, r1 = br * bs, min((br + 1) * bs, L)
        block_mass = 0.0
        for bc in top_blocks:
            c0, c1 = int(bc) * bs, min((int(bc) + 1) * bs, L)
            block_mass += float(A[r0:r1, c0:c1].sum())

        total += block_mass / max(1, r1 - r0)
        n_block_rows += 1

    return total / max(n_block_rows, 1), {
        "block_size": int(bs),
        "topk_ratio": int(ratio),
        "num_blocks": int(nb),
        "keep_blocks_per_row": int(keep_per_row),
    }


@dataclass
class PatternResult:
    pattern: str
    score: float
    params: Dict[str, object]


def classify_head(
    A: np.ndarray,
    *,
    q_start: Optional[int],
    last_q: int,
    stream_grid: Sequence[Tuple[int, int]],
    vs_grid: Sequence[Tuple[int, int]],
    block_grid: Sequence[Tuple[int, int]],
    keep_global: int,
    keep_local: int,
    cap_vs: bool = True,
) -> List[PatternResult]:
    A = causal_tril(A)
    L = A.shape[0]
    q0 = q_start_for_length(L, q_start)

    results: List[PatternResult] = []

    for v, s in stream_grid:
        score = score_stream_llm(A, q0, v, s)
        results.append(PatternResult(
            pattern="stream_llm",
            score=float(score),
            params={"vertical_size": int(v), "slash_size": int(s), "q_start": int(q0)},
        ))

    for v, s in vs_grid:
        score, detail = score_vertical_and_slash(
            A,
            q0,
            v,
            s,
            last_q=last_q,
            keep_global=keep_global,
            keep_local=keep_local,
            cap_candidates=cap_vs,
        )
        params: Dict[str, object] = {
            "vertical_size_requested": int(v),
            "slash_size_requested": int(s),
            "last_q": int(last_q),
            "q_start": int(q0),
            **detail,
        }
        results.append(PatternResult("vertical_and_slash", float(score), params))

    for bs, ratio in block_grid:
        score, detail = score_block_sparse(A, q0, bs, ratio)
        results.append(PatternResult(
            pattern="block_sparse",
            score=float(score),
            params={"q_start": int(q0), **detail},
        ))

    results.sort(key=lambda x: x.score, reverse=True)
    return results


# -----------------------------
# CLI
# -----------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Classify attention maps into StreamLLM / Vertical-Slash / Block-Sparse patterns")
    p.add_argument("--sample_dir", type=str, required=True, help="Directory containing layer_XX/head_YY.npy files")
    p.add_argument("--out", type=str, default=None, help="Output JSON. Default: <sample_dir>/pattern_classification.json")
    p.add_argument("--layers", type=str, default=None, help="Comma-separated layer indices. Default: all")
    p.add_argument("--heads", type=str, default=None, help="Comma-separated head indices. Default: all")
    p.add_argument("--q_start", type=int, default=None, help="Rows q>=q_start are used for scoring. Default: min(2500,L//2)")
    p.add_argument("--last_q", type=int, default=None, help="Rows from the end used for VS estimation. Default scales with L")
    p.add_argument("--stream_grid", type=str, default=None, help="Override stream grid, e.g. '32:128,50:256,64:384'")
    p.add_argument("--vs_grid", type=str, default=None, help="Override VS grid, e.g. '16:128,32:256,64:384'")
    p.add_argument("--block_grid", type=str, default=None, help="Override block grid, e.g. '32:8,64:8'")
    p.add_argument("--keep_global", type=int, default=None, help="Forced initial columns for VS. Default scales with L")
    p.add_argument("--keep_local", type=int, default=None, help="Forced local diagonal offsets for VS. Default scales with L")
    p.add_argument("--min_gap", type=float, default=0.0, help="Mark heads with score_gap < min_gap as low_confidence")
    p.add_argument("--legacy_grids", action="store_true", help="Use original loose grids for comparison")
    p.add_argument("--no_cap_vs", action="store_true", help="Do not cap VS candidate sizes. Not recommended for short L")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    sample_dir = Path(args.sample_dir)
    num_layers, num_heads, L = infer_dims(sample_dir)

    if args.legacy_grids:
        stream_grid, vs_grid, block_grid, default_last_q = legacy_grids_for_length(L)
    else:
        stream_grid, vs_grid, block_grid, default_last_q = default_grids_for_length(L)

    user_stream = parse_grid(args.stream_grid)
    user_vs = parse_grid(args.vs_grid)
    user_block = parse_grid(args.block_grid)
    if user_stream is not None:
        stream_grid = dedupe_and_clip_grid(user_stream, L)
    if user_vs is not None:
        vs_grid = dedupe_and_clip_grid(user_vs, L)
    if user_block is not None:
        block_grid = dedupe_and_clip_grid(user_block, L)

    last_q = int(args.last_q) if args.last_q is not None else int(default_last_q)
    last_q = max(1, min(last_q, L))

    default_keep_global, default_keep_local = scaled_keep_limits(L)
    keep_global = int(args.keep_global) if args.keep_global is not None else default_keep_global
    keep_local = int(args.keep_local) if args.keep_local is not None else default_keep_local

    layers = parse_csv_ints(args.layers) or list(range(num_layers))
    heads = parse_csv_ints(args.heads) or list(range(num_heads))

    out_path = Path(args.out) if args.out else sample_dir / "pattern_classification.json"

    counts: Dict[str, int] = {"stream_llm": 0, "vertical_and_slash": 0, "block_sparse": 0, "low_confidence": 0}
    results: Dict[str, Dict[str, object]] = {}

    for layer in layers:
        layer_key = f"{layer:02d}"
        results[layer_key] = {}
        for head in heads:
            A = load_square(sample_dir, layer, head)
            ranked = classify_head(
                A,
                q_start=args.q_start,
                last_q=last_q,
                stream_grid=stream_grid,
                vs_grid=vs_grid,
                block_grid=block_grid,
                keep_global=keep_global,
                keep_local=keep_local,
                cap_vs=not args.no_cap_vs,
            )

            best = ranked[0]
            second = ranked[1] if len(ranked) > 1 else ranked[0]
            score_gap = float(best.score - second.score)
            low_confidence = bool(score_gap < float(args.min_gap)) if args.min_gap > 0 else False

            counts[best.pattern] = counts.get(best.pattern, 0) + 1
            if low_confidence:
                counts["low_confidence"] += 1

            results[layer_key][str(head)] = {
                "best": asdict(best),
                "second": asdict(second),
                "score_gap": score_gap,
                "low_confidence": low_confidence,
                "ranked": [asdict(x) for x in ranked],
            }

    summary = {
        "sample_dir": str(sample_dir),
        "seq_len": int(L),
        "num_layers": int(num_layers),
        "num_heads": int(num_heads),
        "layers_scored": [int(x) for x in layers],
        "heads_scored": [int(x) for x in heads],
        "q_start": int(q_start_for_length(L, args.q_start)),
        "last_q": int(last_q),
        "stream_grid": [[int(a), int(b)] for a, b in stream_grid],
        "vs_grid": [[int(a), int(b)] for a, b in vs_grid],
        "block_grid": [[int(a), int(b)] for a, b in block_grid],
        "keep_global": int(keep_global),
        "keep_local": int(keep_local),
        "vs_caps": list(vs_caps(L)),
        "cap_vs": bool(not args.no_cap_vs),
        "legacy_grids": bool(args.legacy_grids),
        "min_gap": float(args.min_gap),
        "counts": counts,
        "results": results,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"results"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
