#!/usr/bin/env python3
"""
Shared per-layer sparse mask experiment on LongBench-v2.

Pipeline:
  1. Full-attention prefill baseline -> extract last_q query rows per layer/head.
  2. Per-layer directional coverage similarity -> select one representative head.
  3. Build shared sparse masks from representative heads (decoupled mask builders).
  4. Optionally apply shared masks during prefill and/or decode (decoupled stages).

Analysis-only mode (--apply_prefill false --apply_decode false) is fully supported.
Sparse attention intervention hooks are provided for Qwen2 eager attention.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Allow importing attention_map utilities when running from repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def str_to_bool(value: str) -> bool:
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Shared per-layer sparse mask experiment (single-cluster)"
    )
    p.add_argument(
        "--model_name_or_path",
        type=str,
        default="/home/ubuntu/work/model/Qwen2.5-3B",
    )
    p.add_argument(
        "--data_path",
        type=str,
        default="/home/ubuntu/work/datasets/longbench_v2/longbench_v2_train.jsonl",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="/home/ubuntu/work/experiments/outputs/shared_layer_mask_top_p95",
    )
    p.add_argument("--num_samples", type=int, default=3)
    p.add_argument(
        "--samples_per_domain",
        type=int,
        default=None,
        help="If set, load up to N samples per --domains (overrides num_samples cap logic).",
    )
    p.add_argument(
        "--domains",
        type=str,
        nargs="*",
        default=None,
        help="Optional domain filter, e.g. --domains 'Long In-context Learning' 'Single-Document QA'",
    )
    p.add_argument("--sub_domain", type=str, default=None)
    p.add_argument("--difficulty", type=str, default=None)
    p.add_argument("--length", type=str, default=None)
    p.add_argument("--start_line", type=int, default=0)
    p.add_argument("--max_input_length", type=int, default=8192)
    p.add_argument("--last_q", type=int, default=32)
    p.add_argument("--dtype", type=str, choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument(
        "--mask_method",
        type=str,
        choices=["top_p", "top_k", "top_p_local", "adaptive_top_p", "top_ratio"],
        default="top_p",
    )
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=128)
    p.add_argument(
        "--top_ratio",
        type=float,
        default=0.0642,
        help="For mask_method=top_ratio, keep the top fraction of valid tokens by attention score.",
    )
    p.add_argument("--local_window", type=int, default=256)
    p.add_argument(
        "--adaptive_top_p_min",
        type=float,
        default=0.65,
        help="Minimum top_p for adaptive_top_p (FlexPrefill proxy); scales up toward --top_p near query",
    )
    p.add_argument(
        "--representative_selection",
        type=str,
        choices=["coverage", "random", "survey_fixed"],
        default="coverage",
        help="How to pick per-layer representative head during phase-1",
    )
    p.add_argument(
        "--sparse_head_indices",
        type=str,
        default=None,
        help='Heads receiving sparse mask: comma-separated ints, "even", or "odd" (DuoAttention proxy)',
    )
    p.add_argument(
        "--baseline_id",
        type=str,
        default=None,
        help="Load settings from experiments/baseline_registry.json and write experiment_manifest.json",
    )
    p.add_argument(
        "--debug_layers",
        type=str,
        default=None,
        help='Smoke/debug: only sparsify these layer indices, e.g. "0,1"',
    )

    p.add_argument("--apply_prefill", type=str_to_bool, default=False)
    p.add_argument("--apply_decode", type=str_to_bool, default=False)

    p.add_argument("--generate", type=str_to_bool, default=False)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--do_sample", type=str_to_bool, default=False)
    p.add_argument("--temperature", type=float, default=1.0)

    p.add_argument("--save_attention_maps", type=str_to_bool, default=False)
    p.add_argument("--save_masks", type=str_to_bool, default=True)
    p.add_argument("--save_similarity", type=str_to_bool, default=True)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help="Chunked prefill size for memory-efficient long context.",
    )

    # Post-run filtering on experiment metrics.
    p.add_argument(
        "--filter_after_run",
        type=str_to_bool,
        default=True,
        help="After all samples finish, filter by coverage/sparsity and write filtered_summary.json",
    )
    p.add_argument(
        "--min_layer_mean_coverage",
        type=float,
        default=0.85,
        help="Keep sample if every layer's mean_coverage >= this value",
    )
    p.add_argument(
        "--min_avg_representative_score",
        type=float,
        default=0.88,
        help="Keep sample if average representative_score across layers >= this value",
    )
    p.add_argument(
        "--min_avg_sparsity",
        type=float,
        default=0.15,
        help="Keep sample if average layer sparsity >= this value",
    )
    p.add_argument(
        "--max_avg_sparsity",
        type=float,
        default=0.95,
        help="Keep sample if average layer sparsity <= this value",
    )

    # Downstream task evaluation: accuracy / perplexity / output consistency.
    p.add_argument(
        "--run_task_eval",
        type=str_to_bool,
        default=False,
        help=(
            "Two-phase eval: (1) head_selection_num_samples for per-layer head "
            "confirmation; (2) eval_num_samples for sparse-accuracy testing"
        ),
    )
    p.add_argument(
        "--head_selection_num_samples",
        type=int,
        default=3,
        help="Phase-1 samples used to confirm per-layer representative heads",
    )
    p.add_argument(
        "--eval_num_samples",
        type=int,
        default=10,
        help="Phase-2 samples for sparse-attention accuracy / perplexity / consistency",
    )
    p.add_argument(
        "--eval_apply_prefill",
        type=str_to_bool,
        default=True,
        help="Apply shared sparse mask during prefill in task eval",
    )
    p.add_argument(
        "--eval_apply_decode",
        type=str_to_bool,
        default=True,
        help="Apply shared sparse mask during decode in task eval",
    )
    p.add_argument(
        "--eval_max_new_tokens",
        type=int,
        default=8,
        help="Max new tokens for MCQ answer generation in task eval",
    )
    p.add_argument(
        "--eval_ppl_chunk_size",
        type=int,
        default=None,
        help="Chunk size for memory-efficient PPL (default: --chunk_size or 2048)",
    )
    p.add_argument(
        "--eval_compute_ppl",
        type=str_to_bool,
        default=True,
        help="Compute full-sequence perplexity in task eval (chunked to save VRAM)",
    )
    p.add_argument(
        "--skip_head_selection",
        type=str_to_bool,
        default=False,
        help="Skip phase-1 and load global_representative_heads.json from output_dir",
    )
    p.add_argument(
        "--eval_mode_combos",
        type=str,
        default="ff,ss,sf,fs",
        help=(
            "Comma-separated prefill/decode combos for task eval: "
            "ff=full+full, ss=sparse+sparse, sf=sparse+full, fs=full+sparse"
        ),
    )
    p.add_argument(
        "--ff_reference_roots",
        type=str,
        default=None,
        help=(
            "Optional comma-separated output roots containing task_eval.json files. "
            "When ff is requested, prefill_full_decode_full is loaded by sample_id "
            "from these roots instead of recomputing dense full-attention generation."
        ),
    )
    return p.parse_args()


def parse_debug_layers(spec: Optional[str]) -> Optional[set]:
    if spec is None or str(spec).strip() == "":
        return None
    return {int(x.strip()) for x in str(spec).split(",") if x.strip()}


EVAL_MODE_COMBO_SPECS: Dict[str, Dict[str, Any]] = {
    "ff": {
        "key": "prefill_full_decode_full",
        "apply_prefill": False,
        "apply_decode": False,
    },
    "ss": {
        "key": "prefill_sparse_decode_sparse",
        "apply_prefill": True,
        "apply_decode": True,
    },
    "sf": {
        "key": "prefill_sparse_decode_full",
        "apply_prefill": True,
        "apply_decode": False,
    },
    "fs": {
        "key": "prefill_full_decode_sparse",
        "apply_prefill": False,
        "apply_decode": True,
    },
}


# ---------------------------------------------------------------------------
# Model / data loading
# ---------------------------------------------------------------------------


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    return mapping[dtype_name]


def load_model_and_tokenizer(args: argparse.Namespace):
    dtype = _resolve_dtype(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    device_map = None if args.device == "cpu" else {"": args.device}
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    if args.device == "cpu":
        model = model.to(args.device)
    model.eval()
    return model, tokenizer


def build_prompt(sample: Dict[str, Any]) -> str:
    """Assemble LongBench-v2 multiple-choice prompt."""
    context = str(sample.get("context", "")).strip()
    question = str(sample.get("question", "")).strip()
    choice_a = str(sample.get("choice_A", "")).strip()
    choice_b = str(sample.get("choice_B", "")).strip()
    choice_c = str(sample.get("choice_C", "")).strip()
    choice_d = str(sample.get("choice_D", "")).strip()

    return f"""You are given a long context and a multiple-choice question.
Read the context carefully and choose the correct answer from A, B, C, or D.

Context:
{context}

Question:
{question}

Choices:
A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}

Please answer with only one letter: A, B, C, or D.

Answer:
"""


def _row_matches_filters(row: Dict[str, Any], args: argparse.Namespace) -> bool:
    if args.domains:
        row_domain = str(row.get("domain", row.get("_selected_domain", "unknown")))
        if row_domain not in args.domains:
            return False
    if args.sub_domain is not None:
        if str(row.get("sub_domain", "unknown")) != args.sub_domain:
            return False
    if args.difficulty is not None:
        if str(row.get("difficulty", "unknown")) != args.difficulty:
            return False
    if args.length is not None:
        if str(row.get("length", "unknown")) != args.length:
            return False
    return True


def load_longbench_v2_samples(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """
    Load LongBench-v2 samples from local json/jsonl.

    Supports:
      - sequential first-N loading (--num_samples)
      - metadata filters (--domains / --sub_domain / --difficulty / --length)
      - per-domain cap (--samples_per_domain)
    """
    data_path = Path(args.data_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"data_path not found: {data_path}")

    rows: List[Dict[str, Any]] = []
    per_domain_count: Dict[str, int] = {}

    def _accept_row(row: Dict[str, Any], line_idx: int) -> bool:
        if not _row_matches_filters(row, args):
            return False

        if args.samples_per_domain is not None:
            domain = str(row.get("domain", row.get("_selected_domain", "unknown")))
            if per_domain_count.get(domain, 0) >= args.samples_per_domain:
                return False
        elif len(rows) >= args.num_samples:
            return False

        return True

    def _append_row(row: Dict[str, Any], line_idx: int) -> None:
        row = dict(row)
        row["_line_index"] = line_idx
        domain = str(row.get("domain", row.get("_selected_domain", "unknown")))
        per_domain_count[domain] = per_domain_count.get(domain, 0) + 1
        rows.append(row)

    if data_path.suffix.lower() == ".jsonl":
        with data_path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if line_idx < args.start_line:
                    continue
                if args.samples_per_domain is None and len(rows) >= args.num_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if _accept_row(row, line_idx):
                    _append_row(row, line_idx)
    elif data_path.suffix.lower() == ".json":
        with data_path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            for line_idx, row in enumerate(obj):
                if line_idx < args.start_line:
                    continue
                if not isinstance(row, dict):
                    continue
                if _accept_row(row, line_idx):
                    _append_row(row, line_idx)
                    if args.samples_per_domain is None and len(rows) >= args.num_samples:
                        break
        elif isinstance(obj, dict):
            if _accept_row(obj, 0):
                _append_row(obj, 0)
    else:
        raise ValueError(f"Unsupported data format: {data_path}")

    if not rows:
        raise ValueError(
            "No samples matched filters: "
            f"data_path={data_path}, domains={args.domains}, "
            f"samples_per_domain={args.samples_per_domain}, num_samples={args.num_samples}"
        )

    logger.info(
        "Loaded %d samples from %s | domains=%s",
        len(rows),
        data_path,
        sorted(per_domain_count.keys()),
    )
    return rows


# ---------------------------------------------------------------------------
# Attention utilities
# ---------------------------------------------------------------------------


def normalize_attention(attn: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Row-wise safe normalization over key dimension."""
    denom = attn.sum(dim=-1, keepdim=True).clamp_min(eps)
    return attn / denom


def query_abs_positions_from_last_q(seq_len: int, last_q: int) -> torch.Tensor:
    """Absolute query positions for the last_q rows."""
    last_q = min(last_q, seq_len)
    start = seq_len - last_q
    return torch.arange(start, seq_len, dtype=torch.long)


def build_causal_valid_mask(
    num_queries: int,
    seq_len: int,
    query_abs_positions: Optional[torch.Tensor] = None,
    *,
    device: torch.device,
) -> torch.Tensor:
    """
    Boolean mask of valid (query, key) pairs under causal attention.
    True = position is legal before applying sparsity.
    """
    if query_abs_positions is None:
        query_abs_positions = torch.arange(num_queries, device=device)
    else:
        query_abs_positions = query_abs_positions.to(device)

    key_idx = torch.arange(seq_len, device=device).view(1, -1)
    query_idx = query_abs_positions.view(-1, 1)
    return key_idx <= query_idx


def build_top_p_mask(
    attn: torch.Tensor,
    top_p: float,
    query_abs_positions: Optional[torch.Tensor] = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Top-p mass mask for attn [last_q, seq_len].
    Returns bool mask, True = keep.
    """
    attn = normalize_attention(attn, eps=eps)
    num_q, seq_len = attn.shape
    device = attn.device

    if query_abs_positions is None:
        query_abs_positions = query_abs_positions_from_last_q(seq_len, num_q).to(device)
    else:
        query_abs_positions = query_abs_positions.to(device)

    valid = build_causal_valid_mask(num_q, seq_len, query_abs_positions, device=device)
    out = torch.zeros_like(valid, dtype=torch.bool)

    for row in range(num_q):
        abs_q = int(query_abs_positions[row].item())
        causal_len = abs_q + 1
        row_attn = attn[row, :causal_len]
        total = float(row_attn.sum().item())
        if total <= 0:
            out[row, :causal_len] = True
            continue

        target = top_p * total
        sorted_idx = torch.argsort(row_attn, descending=True)
        chosen: List[int] = []
        cum = 0.0
        for idx in sorted_idx.tolist():
            cum += float(row_attn[idx].item())
            chosen.append(idx)
            if cum >= target:
                break
        out[row, chosen] = True

    return out & valid


# ---------------------------------------------------------------------------
# Sparse mask builders (decoupled)
# ---------------------------------------------------------------------------


class SparseMaskBuilder(ABC):
    @abstractmethod
    def build(
        self,
        attn: torch.Tensor,
        *,
        query_abs_positions: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Return bool mask [last_q, seq_len], True = keep."""


class TopPMaskBuilder(SparseMaskBuilder):
    def __init__(self, top_p: float = 0.95) -> None:
        self.top_p = top_p

    def build(
        self,
        attn: torch.Tensor,
        *,
        query_abs_positions: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        return build_top_p_mask(attn, self.top_p, query_abs_positions=query_abs_positions)


class TopKMaskBuilder(SparseMaskBuilder):
    def __init__(self, top_k: int = 128) -> None:
        self.top_k = top_k

    def build(
        self,
        attn: torch.Tensor,
        *,
        query_abs_positions: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        attn = normalize_attention(attn)
        num_q, seq_len = attn.shape
        device = attn.device

        if query_abs_positions is None:
            query_abs_positions = query_abs_positions_from_last_q(seq_len, num_q).to(device)
        else:
            query_abs_positions = query_abs_positions.to(device)

        valid = build_causal_valid_mask(num_q, seq_len, query_abs_positions, device=device)
        out = torch.zeros_like(valid, dtype=torch.bool)

        for row in range(num_q):
            abs_q = int(query_abs_positions[row].item())
            causal_len = abs_q + 1
            k = min(self.top_k, causal_len)
            row_attn = attn[row, :causal_len]
            if k <= 0:
                continue
            top_idx = torch.topk(row_attn, k=k, largest=True).indices
            out[row, top_idx] = True

        return out & valid


class TopRatioMaskBuilder(SparseMaskBuilder):
    """Keep a fixed fraction of causal-valid tokens with highest attention score."""

    def __init__(self, top_ratio: float = 0.0642) -> None:
        if not 0.0 < top_ratio <= 1.0:
            raise ValueError(f"top_ratio must be in (0, 1], got {top_ratio}")
        self.top_ratio = float(top_ratio)

    def build(
        self,
        attn: torch.Tensor,
        *,
        query_abs_positions: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        attn = normalize_attention(attn)
        num_q, seq_len = attn.shape
        device = attn.device

        if query_abs_positions is None:
            query_abs_positions = query_abs_positions_from_last_q(seq_len, num_q).to(device)
        else:
            query_abs_positions = query_abs_positions.to(device)

        valid = build_causal_valid_mask(num_q, seq_len, query_abs_positions, device=device)
        out = torch.zeros_like(valid, dtype=torch.bool)

        for row in range(num_q):
            abs_q = int(query_abs_positions[row].item())
            causal_len = abs_q + 1
            keep_k = max(1, int(math.ceil(causal_len * self.top_ratio)))
            row_attn = attn[row, :causal_len]
            top_idx = torch.topk(row_attn, k=min(keep_k, causal_len), largest=True).indices
            out[row, top_idx] = True

        return out & valid


class AdaptiveTopPMaskBuilder(SparseMaskBuilder):
    """FlexPrefill proxy: higher top_p for query rows closer to sequence end."""

    def __init__(self, top_p: float = 0.85, min_top_p: float = 0.65) -> None:
        self.top_p = top_p
        self.min_top_p = min(min_top_p, top_p)

    def build(
        self,
        attn: torch.Tensor,
        *,
        query_abs_positions: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        attn = normalize_attention(attn)
        num_q, seq_len = attn.shape
        device = attn.device

        if query_abs_positions is None:
            query_abs_positions = query_abs_positions_from_last_q(seq_len, num_q).to(device)
        else:
            query_abs_positions = query_abs_positions.to(device)

        valid = build_causal_valid_mask(num_q, seq_len, query_abs_positions, device=device)
        out = torch.zeros_like(valid, dtype=torch.bool)
        span = max(self.top_p - self.min_top_p, 1e-6)

        for row in range(num_q):
            # Rows later in last_q window (closer to query end) get higher top_p.
            t = row / max(num_q - 1, 1)
            row_top_p = self.min_top_p + span * t
            row_attn = attn[row : row + 1]
            row_pos = query_abs_positions[row : row + 1]
            row_mask = build_top_p_mask(row_attn, row_top_p, query_abs_positions=row_pos)
            out[row] = row_mask[0]

        return out & valid


class TopPWithLocalWindowMaskBuilder(SparseMaskBuilder):
    def __init__(self, top_p: float = 0.95, local_window: int = 256) -> None:
        self.top_p = top_p
        self.local_window = local_window

    def build(
        self,
        attn: torch.Tensor,
        *,
        query_abs_positions: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        top_p_mask = build_top_p_mask(
            attn, self.top_p, query_abs_positions=query_abs_positions
        )
        num_q, seq_len = attn.shape
        device = attn.device

        if query_abs_positions is None:
            query_abs_positions = query_abs_positions_from_last_q(seq_len, num_q).to(device)
        else:
            query_abs_positions = query_abs_positions.to(device)

        local_mask = torch.zeros((num_q, seq_len), dtype=torch.bool, device=device)
        for row in range(num_q):
            abs_q = int(query_abs_positions[row].item())
            start = max(0, abs_q - self.local_window + 1)
            local_mask[row, start : abs_q + 1] = True

        valid = build_causal_valid_mask(num_q, seq_len, query_abs_positions, device=device)
        return (top_p_mask | local_mask) & valid


def get_mask_builder(args: argparse.Namespace) -> SparseMaskBuilder:
    if args.mask_method == "top_p":
        return TopPMaskBuilder(top_p=args.top_p)
    if args.mask_method == "top_k":
        return TopKMaskBuilder(top_k=args.top_k)
    if args.mask_method == "top_ratio":
        return TopRatioMaskBuilder(top_ratio=args.top_ratio)
    if args.mask_method == "top_p_local":
        return TopPWithLocalWindowMaskBuilder(
            top_p=args.top_p, local_window=args.local_window
        )
    if args.mask_method == "adaptive_top_p":
        return AdaptiveTopPMaskBuilder(
            top_p=args.top_p, min_top_p=args.adaptive_top_p_min
        )
    raise ValueError(f"Unknown mask_method: {args.mask_method}")


def parse_sparse_head_indices(
    spec: Optional[str], num_heads: int
) -> Optional[List[int]]:
    if spec is None or str(spec).strip() == "":
        return None
    spec = str(spec).strip().lower()
    if spec == "even":
        return [h for h in range(num_heads) if h % 2 == 0]
    if spec == "odd":
        return [h for h in range(num_heads) if h % 2 == 1]
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


_BASELINE_REGISTRY_PATH = Path(__file__).resolve().parent / "baseline_registry.json"


def load_baseline_registry() -> Dict[str, Any]:
    if not _BASELINE_REGISTRY_PATH.is_file():
        raise FileNotFoundError(f"Missing baseline registry: {_BASELINE_REGISTRY_PATH}")
    return json.loads(_BASELINE_REGISTRY_PATH.read_text(encoding="utf-8"))


def apply_baseline_config(args: argparse.Namespace) -> Dict[str, Any]:
    registry = load_baseline_registry()
    baselines = registry.get("baselines", {})
    if args.baseline_id not in baselines:
        raise KeyError(
            f"Unknown baseline_id={args.baseline_id!r}; "
            f"available: {sorted(baselines.keys())}"
        )
    cfg = baselines[args.baseline_id]
    args.representative_selection = cfg.get(
        "representative_selection", args.representative_selection
    )
    args.mask_method = cfg.get("mask_method", args.mask_method)
    args.top_p = float(cfg.get("top_p", args.top_p))
    args.top_k = int(cfg.get("top_k", args.top_k))
    args.top_ratio = float(cfg.get("top_ratio", args.top_ratio))
    args.local_window = int(cfg.get("local_window", args.local_window))
    if "adaptive_top_p_min" in cfg:
        args.adaptive_top_p_min = float(cfg["adaptive_top_p_min"])
    sparse = cfg.get("sparse_head_indices")
    if sparse is not None:
        args.sparse_head_indices = str(sparse)
    return cfg


def write_experiment_manifest(
    output_dir: Path,
    args: argparse.Namespace,
    baseline_cfg: Optional[Dict[str, Any]],
) -> Path:
    manifest: Dict[str, Any] = {
        "baseline_id": args.baseline_id,
        "method_name": (baseline_cfg or {}).get("method_name", "Single-Cluster"),
        "paper_source": (baseline_cfg or {}).get("paper_source", ""),
        "approximation_notes": (baseline_cfg or {}).get("approximation_notes", ""),
        "representative_selection": args.representative_selection,
        "mask_method": args.mask_method,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "top_ratio": args.top_ratio,
        "local_window": args.local_window,
        "sparse_head_indices": args.sparse_head_indices,
        "seed": args.seed,
        "eval_num_samples": args.eval_num_samples,
        "head_selection_num_samples": args.head_selection_num_samples,
        "eval_mode_combos": args.eval_mode_combos,
        "model_name_or_path": args.model_name_or_path,
        "data_path": args.data_path,
        "output_dir": str(output_dir),
    }
    path = output_dir / "experiment_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Similarity / representative head selection
# ---------------------------------------------------------------------------


def _row_coverage(
    target_row: torch.Tensor,
    key_positions: torch.Tensor,
    abs_q: int,
    eps: float = 1e-12,
) -> float:
    causal = target_row[: abs_q + 1]
    total = float(causal.sum().item())
    if total <= 0 or key_positions.numel() == 0:
        return 0.0
    covered = float(causal[key_positions].sum().item())
    return covered / total


def compute_directional_coverage_similarity(
    layer_attn: torch.Tensor,
    mask_builder_for_similarity: SparseMaskBuilder,
    query_abs_positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Directional coverage similarity for one layer.

    Args:
        layer_attn: [num_heads, last_q, seq_len]
    Returns:
        similarity: [num_heads, num_heads], S[i,j] = coverage of head j by masks from head i.
    """
    layer_attn = normalize_attention(layer_attn)
    num_heads, last_q, seq_len = layer_attn.shape
    device = layer_attn.device

    if query_abs_positions is None:
        query_abs_positions = query_abs_positions_from_last_q(seq_len, last_q).to(device)
    else:
        query_abs_positions = query_abs_positions.to(device)

    # Precompute sparse key positions per (head_i, query_row) from head i's attention.
    keys_by_head_query: List[List[torch.Tensor]] = []
    for head_i in range(num_heads):
        mask_i = mask_builder_for_similarity.build(
            layer_attn[head_i],
            query_abs_positions=query_abs_positions,
        )
        row_keys: List[torch.Tensor] = []
        for row in range(last_q):
            row_keys.append(mask_i[row].nonzero(as_tuple=False).view(-1))
        keys_by_head_query.append(row_keys)

    similarity = torch.zeros((num_heads, num_heads), dtype=torch.float32, device="cpu")
    for head_i in range(num_heads):
        for head_j in range(num_heads):
            scores: List[float] = []
            for row in range(last_q):
                abs_q = int(query_abs_positions[row].item())
                scores.append(
                    _row_coverage(
                        layer_attn[head_j, row],
                        keys_by_head_query[head_i][row],
                        abs_q,
                    )
                )
            similarity[head_i, head_j] = float(sum(scores) / max(len(scores), 1))

    return similarity


def select_representative_head(
    similarity: torch.Tensor,
) -> Tuple[int, torch.Tensor]:
    """
    Pick mask provider head with highest mean outgoing coverage.

    Returns:
        representative_head, coverage_score_per_head (mean_j S[i,j])
    """
    coverage_score_per_head = similarity.mean(dim=1)
    representative_head = int(coverage_score_per_head.argmax().item())
    return representative_head, coverage_score_per_head


def select_representative_head_by_mode(
    similarity: torch.Tensor,
    *,
    mode: str,
    layer_idx: int,
    num_heads: int,
    num_layers: int,
    seed: int,
) -> Tuple[int, torch.Tensor]:
    coverage_score_per_head = similarity.mean(dim=1)
    if mode == "coverage":
        rep = int(coverage_score_per_head.argmax().item())
    elif mode == "random":
        rng = random.Random(seed + layer_idx * 10007)
        rep = rng.randrange(num_heads)
    elif mode == "survey_fixed":
        third = max(num_layers // 3, 1)
        if layer_idx < third:
            rep = 0
        elif layer_idx < 2 * third:
            rep = num_heads // 2
        else:
            rep = num_heads - 1
    else:
        raise ValueError(f"Unknown representative_selection: {mode}")
    return rep, coverage_score_per_head


def build_layer_shared_masks(
    attentions: torch.Tensor,
    representative_heads: Dict[int, int],
    mask_builder: SparseMaskBuilder,
    query_abs_positions: Optional[torch.Tensor] = None,
) -> Dict[int, torch.Tensor]:
    """
    Build per-layer shared masks from representative head attention maps.

    Args:
        attentions: [num_layers, num_heads, last_q, seq_len]
    Returns:
        layer_to_mask: layer_idx -> bool mask [last_q, seq_len]
    """
    num_layers = attentions.shape[0]
    layer_to_mask: Dict[int, torch.Tensor] = {}

    for layer_idx in range(num_layers):
        rep_head = representative_heads[layer_idx]
        rep_attn = attentions[layer_idx, rep_head].to(torch.float32)
        layer_to_mask[layer_idx] = mask_builder.build(
            rep_attn, query_abs_positions=query_abs_positions
        )

    return layer_to_mask


def compute_layer_coverage_stats(
    attentions: torch.Tensor,
    layer_to_mask: Dict[int, torch.Tensor],
    query_abs_positions: Optional[torch.Tensor] = None,
) -> Dict[int, Dict[str, float]]:
    """Coverage of shared mask on every head in each layer."""
    attentions = normalize_attention(attentions)
    num_layers, num_heads, last_q, seq_len = attentions.shape
    device = attentions.device

    if query_abs_positions is None:
        query_abs_positions = query_abs_positions_from_last_q(seq_len, last_q).to(device)
    else:
        query_abs_positions = query_abs_positions.to(device)

    stats: Dict[int, Dict[str, float]] = {}
    for layer_idx in range(num_layers):
        mask = layer_to_mask[layer_idx].to(device)
        valid = build_causal_valid_mask(last_q, seq_len, query_abs_positions, device=device)

        coverages: List[float] = []
        for head in range(num_heads):
            attn_h = attentions[layer_idx, head]
            num = float((attn_h * mask * valid).sum().item())
            den = float((attn_h * valid).sum().item())
            coverages.append(num / den if den > 0 else 0.0)

        cov_t = torch.tensor(coverages, dtype=torch.float32)
        stats[layer_idx] = {
            "mean_coverage": float(cov_t.mean().item()),
            "min_coverage": float(cov_t.min().item()),
            "max_coverage": float(cov_t.max().item()),
            "std_coverage": float(cov_t.std(unbiased=False).item()),
            "per_head_coverage": coverages,
        }

    return stats


def compute_sparsity_stats(
    layer_to_mask: Dict[int, torch.Tensor],
    query_abs_positions: Optional[torch.Tensor] = None,
    representative_heads: Optional[Dict[int, int]] = None,
    coverage_scores: Optional[Dict[int, Dict[str, float]]] = None,
    layer_coverage_stats: Optional[Dict[int, Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """Per-layer sparsity under causal valid positions."""
    if not layer_to_mask:
        return {}

    any_mask = next(iter(layer_to_mask.values()))
    last_q, seq_len = any_mask.shape
    device = any_mask.device

    if query_abs_positions is None:
        query_abs_positions = query_abs_positions_from_last_q(seq_len, last_q).to(device)
    else:
        query_abs_positions = query_abs_positions.to(device)

    valid = build_causal_valid_mask(last_q, seq_len, query_abs_positions, device=device)
    out: Dict[str, Any] = {}

    for layer_idx, mask in layer_to_mask.items():
        kept = int((mask & valid).sum().item())
        total = int(valid.sum().item())
        keep_ratio = kept / total if total > 0 else 0.0
        entry: Dict[str, Any] = {
            "kept_positions": kept,
            "total_positions": total,
            "keep_ratio": keep_ratio,
            "sparsity": 1.0 - keep_ratio,
        }
        if representative_heads is not None:
            rep = representative_heads[layer_idx]
            entry["representative_head"] = rep
            if coverage_scores is not None:
                entry["representative_head_coverage_score"] = float(
                    coverage_scores[layer_idx]["coverage_score_per_head"][rep]
                )
        if layer_coverage_stats is not None and layer_idx in layer_coverage_stats:
            entry["mean_layer_coverage"] = layer_coverage_stats[layer_idx]["mean_coverage"]
            entry["min_layer_coverage"] = layer_coverage_stats[layer_idx]["min_coverage"]
            entry["std_layer_coverage"] = layer_coverage_stats[layer_idx]["std_coverage"]

        out[str(layer_idx)] = entry

    return out


# ---------------------------------------------------------------------------
# Attention collection (memory-efficient last_q rows)
# ---------------------------------------------------------------------------


@torch.inference_mode()
def _max_q_for_kv(
    num_heads: int,
    kv_len: int,
    max_attn_score_bytes: int,
) -> int:
    """Max query tokens in one forward given kv_len and eager attn score budget."""
    denom = max(num_heads * kv_len * 4, 1)
    return max(1, max_attn_score_bytes // denom)


def resolve_max_attn_score_bytes(seq_len: int) -> int:
    """Eager attention score memory budget; tighter above 32k for long KV cache."""
    return 1_500_000_000 if seq_len > 32768 else 6_000_000_000


def _next_prefill_chunk_end(
    start: int,
    seq_len: int,
    requested_chunk: int,
    num_heads: int,
    max_attn_score_bytes: int,
) -> int:
    """Exclusive end index for the next prefill chunk (dynamic q given kv_len)."""
    remaining = seq_len - start
    if remaining <= 0:
        return start
    q_req = min(max(1, int(requested_chunk)), remaining)
    kv_len = start + q_req
    q_safe = min(q_req, _max_q_for_kv(num_heads, kv_len, max_attn_score_bytes))
    return start + max(1, q_safe)


@torch.inference_mode()
def _attention_safe_prefill_chunk_size(
    chunk_size: Optional[int],
    seq_len: int,
    *,
    num_heads: int,
    max_attn_score_bytes: int = 6_000_000_000,
) -> int:
    """
    Requested prefill q-chunk upper bound (actual q per step is further capped by kv_len).
    """
    user_chunk = int(chunk_size) if chunk_size is not None else seq_len
    user_chunk = max(1, min(user_chunk, seq_len))
    if seq_len <= 32768:
        return user_chunk
    # At seq_len=131k even q=1 is safe; keep user chunk as the requested upper bound.
    max_at_full = _max_q_for_kv(num_heads, seq_len, max_attn_score_bytes)
    safe = max(1, min(user_chunk, max_at_full))
    if safe < user_chunk:
        logger.info(
            "Attention prefill chunk capped: %d -> %d (seq_len=%d, heads=%d)",
            user_chunk,
            safe,
            seq_len,
            num_heads,
        )
    return safe


def _prefill_prefix_past_key_values(
    model,
    prefix_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    prefix_len: int,
    num_heads: int,
    requested_chunk: int,
    max_attn_score_bytes: int,
    device: torch.device,
) -> Any:
    """
    Build KV cache for prefix only (no attentions returned).

    Uses per-step dynamic q sizing: later chunks have larger kv_len, so q shrinks
    automatically to stay within eager attention score memory.
    """
    if prefix_len <= 0:
        return None

    past_key_values = None
    user_chunk = max(1, int(requested_chunk))

    # Never one-shot prefill long prefixes: q=prefix_len, kv=prefix_len OOMs at 128k.
    if prefix_len > 32768 or user_chunk < prefix_len:
        start = 0
        while start < prefix_len:
            end = _next_prefill_chunk_end(
                start,
                prefix_len,
                user_chunk,
                num_heads,
                max_attn_score_bytes,
            )
            out = model(
                input_ids=prefix_ids[:, start:end],
                attention_mask=attention_mask[:, :end],
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=False,
            )
            past_key_values = out.past_key_values
            del out
            start = end
            if device.type == "cuda":
                torch.cuda.empty_cache()
        return past_key_values

    out = model(
        input_ids=prefix_ids,
        attention_mask=attention_mask[:, :prefix_len],
        use_cache=True,
        output_attentions=False,
    )
    past_key_values = out.past_key_values
    del out
    return past_key_values


@torch.inference_mode()
def collect_last_q_attentions(
    model,
    tokenizer,
    prompt: str,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
    """
    Run full-attention prefill and extract last_q query rows.

    Returns:
        attentions: [num_layers, num_heads, last_q, seq_len] (CPU float32)
        seq_len: int
        input_ids: [1, seq_len]
        attention_mask: [1, seq_len]
    """
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=args.max_input_length is not None,
        max_length=args.max_input_length,
    )
    device = next(model.parameters()).device
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)

    batch, seq_len = input_ids.shape
    if batch != 1:
        raise ValueError("Only batch size 1 is supported.")

    last_q = min(args.last_q, seq_len)
    if last_q <= 0:
        raise ValueError("last_q must be positive.")

    prefix_ids = input_ids[:, :-last_q] if seq_len > last_q else input_ids[:, :0]
    tail_ids = input_ids[:, -last_q:]
    prefix_len = prefix_ids.shape[1]

    past_key_values = None
    num_heads = int(getattr(model.config, "num_attention_heads", 28))
    # Leave headroom for 128k KV cache + activations (often ~35–40 GiB before attn scores).
    max_attn_bytes = resolve_max_attn_score_bytes(seq_len)
    requested_chunk = _attention_safe_prefill_chunk_size(
        args.chunk_size, prefix_len if prefix_len > 0 else seq_len,
        num_heads=num_heads,
        max_attn_score_bytes=max_attn_bytes,
    )
    if seq_len > 32768:
        logger.info(
            "Long-context attention collect: seq_len=%d prefix_len=%d "
            "requested_chunk=%d max_attn_score_mb=%.0f",
            seq_len,
            prefix_len,
            requested_chunk,
            max_attn_bytes / 1e6,
        )

    # Stage 1: prefill prefix without attentions (dynamic q per kv_len).
    if prefix_len > 0:
        past_key_values = _prefill_prefix_past_key_values(
            model,
            prefix_ids,
            attention_mask,
            prefix_len=prefix_len,
            num_heads=num_heads,
            requested_chunk=requested_chunk,
            max_attn_score_bytes=max_attn_bytes,
            device=device,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Stage 2: collect last_q attention rows one token at a time (limits q=1).
    num_layers = model.config.num_hidden_layers
    per_layer_rows: List[List[torch.Tensor]] = [[] for _ in range(num_layers)]
    current_past = past_key_values
    abs_pos = prefix_len

    for local_q in range(last_q):
        token = tail_ids[:, local_q : local_q + 1]
        out = model(
            input_ids=token,
            attention_mask=attention_mask[:, : abs_pos + 1],
            past_key_values=current_past,
            use_cache=True,
            output_attentions=True,
        )
        if out.attentions is None:
            raise RuntimeError(
                "attentions is None; load model with attn_implementation='eager'."
            )
        current_past = out.past_key_values
        for layer_idx, layer_attn in enumerate(out.attentions):
            row = layer_attn[0, :, 0, :].detach().to("cpu", dtype=torch.float32)
            row_full = torch.zeros(num_heads, seq_len, dtype=torch.float32)
            kv_len = min(seq_len, row.shape[-1])
            row_full[:, :kv_len] = row[:, :kv_len]
            per_layer_rows[layer_idx].append(row_full)
        del out
        abs_pos += 1
        if device.type == "cuda":
            torch.cuda.empty_cache()

    attentions = torch.stack(
        [
            torch.stack(per_layer_rows[layer_idx], dim=0).permute(1, 0, 2)
            for layer_idx in range(num_layers)
        ],
        dim=0,
    )  # [layers, heads, last_q, seq_len]
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return attentions, seq_len, input_ids, attention_mask


# ---------------------------------------------------------------------------
# Shared mask controller + attention patch (prefill/decode decoupled)
# ---------------------------------------------------------------------------


@dataclass
class AttentionForwardContext:
    stage: str  # "prefill" | "decode"
    q_len: int
    kv_len: int
    query_abs_start: int
    analysis_seq_len: int


@dataclass
class SharedMaskAttentionController:
    layer_to_mask: Dict[int, torch.Tensor]
    apply_prefill: bool
    apply_decode: bool
    last_q: int
    analysis_seq_len: int
    sparse_head_indices: Optional[List[int]] = None
    debug_layers: Optional[set] = None

    def should_apply(self, stage: str, layer_idx: int) -> bool:
        if self.debug_layers is not None and layer_idx not in self.debug_layers:
            return False
        if layer_idx not in self.layer_to_mask:
            return False
        if stage == "prefill":
            return self.apply_prefill
        if stage == "decode":
            return self.apply_decode
        return False

    def get_runtime_mask(
        self,
        layer_idx: int,
        ctx: AttentionForwardContext,
        *,
        actual_q_len: Optional[int] = None,
        actual_kv_len: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        """
        Build runtime bool mask [q_len, kv_len] for the current forward pass.
        """
        if not self.should_apply(ctx.stage, layer_idx):
            return None

        base_mask = self.layer_to_mask[layer_idx]
        device = base_mask.device
        q_len = actual_q_len if actual_q_len is not None else ctx.q_len
        kv_len = actual_kv_len if actual_kv_len is not None else ctx.kv_len
        runtime = torch.ones((q_len, kv_len), dtype=torch.bool, device=device)

        if ctx.stage == "prefill":
            # Only sparsify the last last_q query rows when doing full/long prefill.
            for local_q in range(q_len):
                abs_q = ctx.query_abs_start + local_q
                if abs_q < self.analysis_seq_len - self.last_q:
                    continue
                rel = abs_q - (self.analysis_seq_len - self.last_q)
                if 0 <= rel < base_mask.shape[0]:
                    row = base_mask[rel]
                    copy_len = min(row.shape[0], kv_len)
                    runtime[local_q, :copy_len] = row[:copy_len]

        elif ctx.stage == "decode":
            # Single-step decode: reuse last prefill query's key pattern.
            last_row = base_mask[-1]
            hist_len = min(last_row.shape[0], kv_len)
            runtime[0, :hist_len] = last_row[:hist_len]
            # Newly generated tokens (beyond analysis_seq_len) remain visible.
            if kv_len > hist_len:
                runtime[0, hist_len:kv_len] = True

        # Enforce causal validity.
        for local_q in range(q_len):
            abs_q = ctx.query_abs_start + local_q
            if abs_q + 1 < kv_len:
                runtime[local_q, abs_q + 1 :] = False

        return runtime


_PATCH_STATE: Dict[str, Any] = {
    "patched": False,
    "original_fn": None,
    "controller": None,
    "context": None,
    "repeat_kv": None,
}


def apply_shared_sparse_mask_to_attention_scores(
    attn_scores: torch.Tensor,
    shared_mask: torch.Tensor,
    sparse_head_indices: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    Apply shared sparse positions to attention scores before softmax.

    Args:
        attn_scores: [batch, num_heads, q_len, kv_len]
        shared_mask: [q_len, kv_len] bool, True = keep
        sparse_head_indices: if set, only these heads are sparsified (DuoAttention proxy)
    """
    if shared_mask is None:
        return attn_scores

    q_len = attn_scores.shape[-2]
    kv_len = attn_scores.shape[-1]
    mask = shared_mask[:q_len, :]
    if mask.shape[-1] < kv_len:
        pad = torch.ones(
            (mask.shape[0], kv_len - mask.shape[-1]),
            dtype=torch.bool,
            device=mask.device,
        )
        mask = torch.cat([mask, pad], dim=-1)
    mask = mask[:, :kv_len]
    mask = mask.unsqueeze(0).unsqueeze(0).expand(
        attn_scores.shape[0], attn_scores.shape[1], q_len, kv_len
    )
    finfo = torch.finfo(attn_scores.dtype)
    masked = attn_scores.masked_fill(~mask, finfo.min)
    if sparse_head_indices is None:
        return masked
    out = attn_scores.clone()
    for head_idx in sparse_head_indices:
        if 0 <= head_idx < out.shape[1]:
            out[:, head_idx, :, :] = masked[:, head_idx, :, :]
    return out


def _patched_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    original_fn = _PATCH_STATE["original_fn"]
    repeat_kv = _PATCH_STATE["repeat_kv"]
    controller: Optional[SharedMaskAttentionController] = _PATCH_STATE["controller"]
    ctx: Optional[AttentionForwardContext] = _PATCH_STATE["context"]

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    layer_idx = getattr(module, "layer_idx", None)
    if (
        controller is not None
        and ctx is not None
        and layer_idx is not None
        and controller.should_apply(ctx.stage, layer_idx)
    ):
        shared_mask = controller.get_runtime_mask(
            layer_idx,
            ctx,
            actual_q_len=attn_weights.shape[-2],
            actual_kv_len=attn_weights.shape[-1],
        )
        if shared_mask is not None:
            attn_weights = apply_shared_sparse_mask_to_attention_scores(
                attn_weights,
                shared_mask.to(attn_weights.device),
                sparse_head_indices=controller.sparse_head_indices,
            )

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def patch_model_attention(
    model,
    controller: SharedMaskAttentionController,
) -> None:
    """Monkey-patch Qwen2 eager attention forward."""
    if _PATCH_STATE["patched"]:
        _PATCH_STATE["controller"] = controller
        return

    from transformers.models.qwen2.modeling_qwen2 import (
        eager_attention_forward,
        repeat_kv,
    )

    import transformers.models.qwen2.modeling_qwen2 as modeling_qwen2

    _PATCH_STATE["original_fn"] = eager_attention_forward
    _PATCH_STATE["repeat_kv"] = repeat_kv
    _PATCH_STATE["controller"] = controller
    modeling_qwen2.eager_attention_forward = _patched_eager_attention_forward
    _PATCH_STATE["patched"] = True
    logger.info(
        "Patched Qwen2 eager_attention_forward (prefill=%s, decode=%s)",
        controller.apply_prefill,
        controller.apply_decode,
    )


def unpatch_model_attention(model) -> None:
    if not _PATCH_STATE["patched"]:
        return

    import transformers.models.qwen2.modeling_qwen2 as modeling_qwen2

    if _PATCH_STATE["original_fn"] is not None:
        modeling_qwen2.eager_attention_forward = _PATCH_STATE["original_fn"]

    _PATCH_STATE["patched"] = False
    _PATCH_STATE["original_fn"] = None
    _PATCH_STATE["controller"] = None
    _PATCH_STATE["context"] = None
    logger.info("Restored original Qwen2 eager_attention_forward")


def set_attention_forward_context(ctx: Optional[AttentionForwardContext]) -> None:
    _PATCH_STATE["context"] = ctx


@torch.inference_mode()
def run_generation_with_optional_sparse(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    controller: Optional[SharedMaskAttentionController],
    args: argparse.Namespace,
) -> str:
    """
    Optional generation with decoupled prefill/decode sparse mask application.
    """
    if controller is None or (not controller.apply_prefill and not controller.apply_decode):
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature if args.do_sample else None,
            use_cache=True,
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    device = input_ids.device
    seq_len = input_ids.shape[1]
    generated = input_ids
    attn_mask = attention_mask

    # --- Prefill ---
    if controller.apply_prefill:
        set_attention_forward_context(
            AttentionForwardContext(
                stage="prefill",
                q_len=seq_len,
                kv_len=seq_len,
                query_abs_start=0,
                analysis_seq_len=controller.analysis_seq_len,
            )
        )
    else:
        set_attention_forward_context(None)

    out = model(
        input_ids=generated,
        attention_mask=attn_mask,
        use_cache=True,
        output_attentions=False,
    )
    past_key_values = out.past_key_values
    del out

    # --- Decode loop ---
    for step in range(args.max_new_tokens):
        if controller.apply_decode:
            set_attention_forward_context(
                AttentionForwardContext(
                    stage="decode",
                    q_len=1,
                    kv_len=generated.shape[1],
                    query_abs_start=generated.shape[1] - 1,
                    analysis_seq_len=controller.analysis_seq_len,
                )
            )
        else:
            set_attention_forward_context(None)

        out = model(
            input_ids=generated[:, -1:],
            attention_mask=attn_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=False,
        )
        logits = out.logits[:, -1, :]
        past_key_values = out.past_key_values
        del out

        if args.do_sample:
            probs = F.softmax(logits / args.temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = logits.argmax(dim=-1, keepdim=True)

        generated = torch.cat([generated, next_token], dim=1)
        attn_mask = torch.cat(
            [attn_mask, torch.ones((1, 1), device=device, dtype=attn_mask.dtype)],
            dim=1,
        )

        if next_token.item() == tokenizer.eos_token_id:
            break

    set_attention_forward_context(None)
    return tokenizer.decode(generated[0], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_results(
    sample_dir: Path,
    *,
    sample_id: str,
    prompt: str,
    seq_len: int,
    sample_meta: Dict[str, Any],
    representative_heads: Dict[int, int],
    coverage_scores: Dict[int, Dict[str, Any]],
    similarity_matrices: Dict[int, torch.Tensor],
    layer_to_mask: Dict[int, torch.Tensor],
    layer_stats: Dict[str, Any],
    sparsity_stats: Dict[str, Any],
    args: argparse.Namespace,
    full_output: Optional[str] = None,
    sparse_output: Optional[str] = None,
    representative_head_attentions: Optional[torch.Tensor] = None,
) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / "input.txt").write_text(prompt, encoding="utf-8")

    rep_json = {
        str(layer): {
            "representative_head": representative_heads[layer],
            "coverage_score": coverage_scores[layer]["coverage_score_per_head"][
                representative_heads[layer]
            ],
            "mean_coverage": coverage_scores[layer]["mean_coverage"],
            "min_coverage": coverage_scores[layer]["min_coverage"],
            "std_coverage": coverage_scores[layer]["std_coverage"],
        }
        for layer in representative_heads
    }
    (sample_dir / "representative_heads.json").write_text(
        json.dumps(rep_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (sample_dir / "layer_stats.json").write_text(
        json.dumps(layer_stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (sample_dir / "sparsity_stats.json").write_text(
        json.dumps(sparsity_stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    meta = {
        "sample_id": sample_id,
        "seq_len": seq_len,
        "domain": sample_meta.get("domain"),
        "sub_domain": sample_meta.get("sub_domain"),
        "difficulty": sample_meta.get("difficulty"),
        "length": sample_meta.get("length"),
        "answer": sample_meta.get("answer"),
        "prompt_token_len": sample_meta.get("prompt_token_len"),
        "last_q": args.last_q,
        "mask_method": args.mask_method,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "top_ratio": args.top_ratio,
        "local_window": args.local_window,
        "apply_prefill": args.apply_prefill,
        "apply_decode": args.apply_decode,
    }
    (sample_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if args.save_similarity:
        torch.save(
            {str(k): v.cpu() for k, v in similarity_matrices.items()},
            sample_dir / "similarity_matrices.pt",
        )

    if args.save_masks:
        torch.save(
            {str(k): v.cpu() for k, v in layer_to_mask.items()},
            sample_dir / "shared_masks.pt",
        )

    if args.save_attention_maps and representative_head_attentions is not None:
        torch.save(representative_head_attentions.cpu(), sample_dir / "representative_head_attentions.pt")

    if full_output is not None:
        (sample_dir / "full_output.txt").write_text(full_output, encoding="utf-8")
    if sparse_output is not None:
        (sample_dir / "sparse_output.txt").write_text(sparse_output, encoding="utf-8")


# ---------------------------------------------------------------------------
# Downstream task evaluation (accuracy / perplexity / consistency)
# ---------------------------------------------------------------------------


def tokenize_prompt(
    tokenizer,
    prompt: str,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=args.max_input_length is not None,
        max_length=args.max_input_length,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)
    return input_ids, attention_mask


def extract_mcq_answer(text: str) -> Optional[str]:
    """Extract the first standalone A/B/C/D letter from generated text."""
    if not text:
        return None
    upper = text.upper().strip()
    patterns = [
        r"(?:ANSWER|CHOICE|OPTION)\s*[:：]?\s*([ABCD])\b",
        r"(?:THE\s+)?(?:CORRECT\s+)?(?:ANSWER|CHOICE|OPTION)\s+IS\s+([ABCD])\b",
        r"^\s*([ABCD])\b",
        r"\b([ABCD])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, upper)
        if match:
            return match.group(1)
    return None


def free_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def resolve_eval_chunk_size(args: argparse.Namespace, seq_len: int) -> int:
    chunk_size = args.eval_ppl_chunk_size or args.chunk_size or 2048
    return max(1, min(int(chunk_size), seq_len))


@torch.inference_mode()
def _chunked_prefill_past_key_values(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    controller: Optional[SharedMaskAttentionController],
    *,
    analysis_seq_len: int,
    chunk_size: int,
) -> Tuple[Any, torch.Tensor]:
    """Memory-efficient prefill; returns KV cache and final-position logits."""
    seq_len = input_ids.shape[1]
    device = input_ids.device
    past_key_values = None
    last_logits = None
    use_sparse = controller is not None and controller.apply_prefill
    num_heads = int(getattr(model.config, "num_attention_heads", 28))
    max_attn_bytes = resolve_max_attn_score_bytes(seq_len)
    requested_chunk = max(1, int(chunk_size))

    if seq_len > 32768:
        logger.info(
            "Long-context eval prefill: seq_len=%d requested_chunk=%d max_attn_score_mb=%.0f",
            seq_len,
            requested_chunk,
            max_attn_bytes / 1e6,
        )

    start = 0
    while start < seq_len:
        end = _next_prefill_chunk_end(
            start, seq_len, requested_chunk, num_heads, max_attn_bytes
        )
        chunk_len = end - start
        if use_sparse:
            set_attention_forward_context(
                AttentionForwardContext(
                    stage="prefill",
                    q_len=chunk_len,
                    kv_len=end,
                    query_abs_start=start,
                    analysis_seq_len=analysis_seq_len,
                )
            )
        else:
            set_attention_forward_context(None)

        out = model(
            input_ids=input_ids[:, start:end],
            attention_mask=attention_mask[:, :end],
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=False,
        )
        past_key_values = out.past_key_values
        last_logits = out.logits[:, -1, :].detach()
        del out
        if device.type == "cuda":
            free_cuda_cache()
        start = end

    set_attention_forward_context(None)
    if last_logits is None:
        raise RuntimeError("chunked prefill did not produce logits")
    return past_key_values, last_logits


@torch.inference_mode()
def compute_sequence_nll_and_ppl(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    controller: Optional[SharedMaskAttentionController],
    *,
    seq_len: int,
    chunk_size: int,
) -> Dict[str, float]:
    """
    Teacher-forcing NLL / perplexity on the full input sequence.

    Uses chunked prefill + KV cache so eager attention never materializes
    a full [seq_len, seq_len] map (critical for 32k contexts).
    """
    device = input_ids.device
    use_sparse = controller is not None and (
        controller.apply_prefill or controller.apply_decode
    )
    patched = use_sparse
    if patched:
        patch_model_attention(model, controller)

    num_heads = int(getattr(model.config, "num_attention_heads", 28))
    max_attn_bytes = resolve_max_attn_score_bytes(seq_len)
    requested_chunk = max(1, int(chunk_size))

    if seq_len > 32768:
        logger.info(
            "Long-context PPL prefill: seq_len=%d requested_chunk=%d max_attn_score_mb=%.0f",
            seq_len,
            requested_chunk,
            max_attn_bytes / 1e6,
        )

    total_nll = 0.0
    total_tokens = 0
    past_key_values = None

    try:
        start = 0
        while start < seq_len:
            end = _next_prefill_chunk_end(
                start, seq_len, requested_chunk, num_heads, max_attn_bytes
            )
            chunk_len = end - start
            if use_sparse:
                set_attention_forward_context(
                    AttentionForwardContext(
                        stage="prefill",
                        q_len=chunk_len,
                        kv_len=end,
                        query_abs_start=start,
                        analysis_seq_len=controller.analysis_seq_len,
                    )
                )
            else:
                set_attention_forward_context(None)

            out = model(
                input_ids=input_ids[:, start:end],
                attention_mask=attention_mask[:, :end],
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=False,
            )
            past_key_values = out.past_key_values
            logits = out.logits

            if chunk_len > 1:
                pred_logits = logits[:, : chunk_len - 1, :].contiguous()
                target_ids = input_ids[:, start + 1 : end].contiguous()
                nll = F.cross_entropy(
                    pred_logits.view(-1, pred_logits.size(-1)),
                    target_ids.view(-1),
                    reduction="sum",
                )
                total_nll += float(nll.item())
                total_tokens += int(target_ids.numel())

            del out, logits
            if device.type == "cuda":
                free_cuda_cache()
            start = end
    finally:
        set_attention_forward_context(None)
        if patched:
            unpatch_model_attention(model)
        del past_key_values
        free_cuda_cache()

    if total_tokens <= 0:
        return {"loss": float("nan"), "perplexity": float("nan"), "num_tokens": 0}

    loss = total_nll / total_tokens
    ppl = float(math.exp(loss)) if math.isfinite(loss) else float("nan")
    return {"loss": loss, "perplexity": ppl, "num_tokens": total_tokens}


def _decode_loop(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    past_key_values: Any,
    *,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    controller: Optional[SharedMaskAttentionController],
    analysis_seq_len: int,
    prefill_logits: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Token-by-token decode with optional sparse mask on decode stage."""
    device = input_ids.device
    input_len = input_ids.shape[1]
    generated = input_ids
    attn_mask = attention_mask
    new_token_ids: List[int] = []
    eos_id = tokenizer.eos_token_id

    if prefill_logits is not None and max_new_tokens > 0:
        if do_sample:
            probs = F.softmax(prefill_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = prefill_logits.argmax(dim=-1, keepdim=True)

        new_token_ids.append(int(next_token.item()))
        generated = torch.cat([generated, next_token], dim=1)
        attn_mask = torch.cat(
            [attn_mask, torch.ones((1, 1), device=device, dtype=attn_mask.dtype)],
            dim=1,
        )
        if eos_id is not None and next_token.item() == eos_id:
            set_attention_forward_context(None)
            new_ids_tensor = torch.tensor(new_token_ids, dtype=torch.long)
            text = tokenizer.decode(new_ids_tensor, skip_special_tokens=True)
            return {
                "new_token_ids": new_token_ids,
                "generated_text": text,
                "full_decoded": tokenizer.decode(generated[0], skip_special_tokens=True),
                "past_key_values": past_key_values,
            }

    for _ in range(max_new_tokens - len(new_token_ids)):
        if controller is not None and controller.apply_decode:
            set_attention_forward_context(
                AttentionForwardContext(
                    stage="decode",
                    q_len=1,
                    kv_len=generated.shape[1],
                    query_abs_start=generated.shape[1] - 1,
                    analysis_seq_len=analysis_seq_len,
                )
            )
        else:
            set_attention_forward_context(None)

        out = model(
            input_ids=generated[:, -1:],
            attention_mask=attn_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=False,
        )
        logits = out.logits[:, -1, :]
        past_key_values = out.past_key_values
        del out

        if do_sample:
            probs = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = logits.argmax(dim=-1, keepdim=True)

        new_token_ids.append(int(next_token.item()))
        generated = torch.cat([generated, next_token], dim=1)
        attn_mask = torch.cat(
            [attn_mask, torch.ones((1, 1), device=device, dtype=attn_mask.dtype)],
            dim=1,
        )

        if eos_id is not None and next_token.item() == eos_id:
            break

    set_attention_forward_context(None)
    new_ids_tensor = torch.tensor(new_token_ids, dtype=torch.long)
    text = tokenizer.decode(new_ids_tensor, skip_special_tokens=True)
    return {
        "new_token_ids": new_token_ids,
        "generated_text": text,
        "full_decoded": tokenizer.decode(generated[0], skip_special_tokens=True),
        "past_key_values": past_key_values,
    }


@torch.inference_mode()
def generate_new_tokens(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    controller: Optional[SharedMaskAttentionController],
    *,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    analysis_seq_len: int,
    prefill_chunk_size: int = 4096,
) -> Dict[str, Any]:
    """
    Greedy/sample generation; returns only newly generated suffix.

    Always uses chunked prefill for long inputs to avoid materializing
    full [seq_len, seq_len] eager attention (required for 32k contexts).
    """
    input_len = input_ids.shape[1]
    use_sparse = controller is not None and (
        controller.apply_prefill or controller.apply_decode
    )
    patched = use_sparse

    if patched:
        patch_model_attention(model, controller)

    try:
        if input_len > prefill_chunk_size or input_len > 32768:
            past_key_values, prefill_logits = _chunked_prefill_past_key_values(
                model,
                input_ids,
                attention_mask,
                controller if controller is not None and controller.apply_prefill else None,
                analysis_seq_len=analysis_seq_len,
                chunk_size=prefill_chunk_size,
            )
        else:
            if controller is not None and controller.apply_prefill:
                set_attention_forward_context(
                    AttentionForwardContext(
                        stage="prefill",
                        q_len=input_len,
                        kv_len=input_len,
                        query_abs_start=0,
                        analysis_seq_len=analysis_seq_len,
                    )
                )
            else:
                set_attention_forward_context(None)
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                output_attentions=False,
            )
            past_key_values = out.past_key_values
            prefill_logits = out.logits[:, -1, :].detach()
            del out

        free_cuda_cache()
        result = _decode_loop(
            model,
            tokenizer,
            input_ids,
            attention_mask,
            past_key_values,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            controller=controller,
            analysis_seq_len=analysis_seq_len,
            prefill_logits=prefill_logits,
        )
        del result["past_key_values"]
        free_cuda_cache()
        return result
    finally:
        set_attention_forward_context(None)
        if patched:
            unpatch_model_attention(model)
        free_cuda_cache()


def parse_eval_mode_combos(spec: str) -> List[str]:
    valid = set(EVAL_MODE_COMBO_SPECS)
    modes = [m.strip().lower() for m in spec.split(",") if m.strip()]
    if not modes:
        raise ValueError("eval_mode_combos must contain at least one of: ff, ss, sf, fs")
    unknown = [m for m in modes if m not in valid]
    if unknown:
        raise ValueError(f"Unknown eval_mode_combos: {unknown}; valid={sorted(valid)}")
    return modes


def make_mode_controller(
    layer_to_mask: Dict[int, torch.Tensor],
    *,
    apply_prefill: bool,
    apply_decode: bool,
    last_q: int,
    analysis_seq_len: int,
    sparse_head_indices: Optional[List[int]] = None,
    debug_layers: Optional[set] = None,
) -> Optional[SharedMaskAttentionController]:
    if not apply_prefill and not apply_decode:
        return None
    return SharedMaskAttentionController(
        layer_to_mask={k: v.clone() for k, v in layer_to_mask.items()},
        apply_prefill=apply_prefill,
        apply_decode=apply_decode,
        last_q=last_q,
        analysis_seq_len=analysis_seq_len,
        sparse_head_indices=sparse_head_indices,
        debug_layers=debug_layers,
    )


def build_mode_generation_metrics(
    gen_result: Dict[str, Any],
    gold_answer: Optional[str],
) -> Dict[str, Any]:
    text = gen_result["generated_text"]
    answer = extract_mcq_answer(text)
    gold = gold_answer.strip().upper() if gold_answer else None
    return {
        "generated_text": text,
        "answer": answer,
        "correct": answer == gold if gold else None,
        "num_new_tokens": len(gen_result["new_token_ids"]),
    }


@torch.inference_mode()
def generate_new_tokens_from_past(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    past_key_values: Any,
    controller: Optional[SharedMaskAttentionController],
    *,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    analysis_seq_len: int,
    prefill_logits: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """
    Decode-only generation starting from an existing KV cache.

    This lets us reuse prefill KV across multiple decode modes (e.g. ff + fs)
    without changing any evaluation semantics.
    """
    use_sparse = controller is not None and controller.apply_decode
    patched = use_sparse
    if patched:
        patch_model_attention(model, controller)
    try:
        result = _decode_loop(
            model,
            tokenizer,
            input_ids,
            attention_mask,
            past_key_values,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            controller=controller,
            analysis_seq_len=analysis_seq_len,
            prefill_logits=prefill_logits,
        )
        # Drop KV cache from return payload (keeps parity with generate_new_tokens()).
        del result["past_key_values"]
        free_cuda_cache()
        return result
    finally:
        set_attention_forward_context(None)
        if patched:
            unpatch_model_attention(model)
        free_cuda_cache()


def evaluate_mode_combo(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_to_mask: Dict[int, torch.Tensor],
    *,
    combo: str,
    seq_len: int,
    last_q: int,
    args: argparse.Namespace,
    gold_answer: Optional[str],
    ppl_chunk: int,
) -> Dict[str, Any]:
    spec = EVAL_MODE_COMBO_SPECS[combo]
    controller = make_mode_controller(
        layer_to_mask,
        apply_prefill=spec["apply_prefill"],
        apply_decode=spec["apply_decode"],
        last_q=last_q,
        analysis_seq_len=seq_len,
        sparse_head_indices=getattr(args, "resolved_sparse_head_indices", None),
        debug_layers=getattr(args, "debug_layers", None),
    )
    ppl = {"loss": float("nan"), "perplexity": float("nan"), "num_tokens": 0}
    if args.eval_compute_ppl:
        ppl = compute_sequence_nll_and_ppl(
            model,
            input_ids,
            attention_mask,
            controller=controller,
            seq_len=seq_len,
            chunk_size=ppl_chunk,
        )
        free_cuda_cache()

    gen = generate_new_tokens(
        model,
        tokenizer,
        input_ids,
        attention_mask,
        controller=controller,
        max_new_tokens=args.eval_max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        analysis_seq_len=seq_len,
        prefill_chunk_size=ppl_chunk,
    )
    free_cuda_cache()
    return {
        "combo": combo,
        "mode_key": spec["key"],
        "apply_prefill": spec["apply_prefill"],
        "apply_decode": spec["apply_decode"],
        "perplexity": ppl,
        "generation": build_mode_generation_metrics(gen, gold_answer),
    }


def legacy_generation_from_modes(
    modes: Dict[str, Dict[str, Any]],
    gold_answer: Optional[str],
) -> Dict[str, Any]:
    """Backward-compatible generation block from ff/ss mode results."""
    gold = gold_answer.strip().upper() if gold_answer else None
    ff = modes.get("prefill_full_decode_full", {}).get("generation", {})
    ss = modes.get("prefill_sparse_decode_sparse", {}).get("generation", {})
    full_answer = ff.get("answer")
    sparse_answer = ss.get("answer")
    full_text = ff.get("generated_text", "")
    sparse_text = ss.get("generated_text", "")
    return {
        "full_generated_text": full_text,
        "sparse_generated_text": sparse_text,
        "full_answer": full_answer,
        "sparse_answer": sparse_answer,
        "gold_answer": gold,
        "full_correct": ff.get("correct"),
        "sparse_correct": ss.get("correct"),
        "answer_letters_match": full_answer == sparse_answer,
        "exact_text_match": full_text.strip() == sparse_text.strip(),
        "token_match_ratio": None,
        "token_prefix_match_length": None,
        "full_num_new_tokens": ff.get("num_new_tokens"),
        "sparse_num_new_tokens": ss.get("num_new_tokens"),
    }


def ensure_modes_from_legacy(result: Dict[str, Any]) -> Dict[str, Any]:
    """Populate modes from legacy full/sparse fields when re-running partial combos."""
    if result.get("modes"):
        return result
    modes: Dict[str, Dict[str, Any]] = {}
    generation = result.get("generation", {})
    if result.get("full_perplexity") is not None and generation:
        modes["prefill_full_decode_full"] = {
            "combo": "ff",
            "apply_prefill": False,
            "apply_decode": False,
            "perplexity": result["full_perplexity"],
            "generation": {
                "generated_text": generation.get("full_generated_text", ""),
                "answer": generation.get("full_answer"),
                "correct": generation.get("full_correct"),
                "num_new_tokens": generation.get("full_num_new_tokens"),
            },
        }
    if result.get("sparse_perplexity") is not None and generation:
        modes["prefill_sparse_decode_sparse"] = {
            "combo": "ss",
            "apply_prefill": True,
            "apply_decode": True,
            "perplexity": result["sparse_perplexity"],
            "generation": {
                "generated_text": generation.get("sparse_generated_text", ""),
                "answer": generation.get("sparse_answer"),
                "correct": generation.get("sparse_correct"),
                "num_new_tokens": generation.get("sparse_num_new_tokens"),
            },
        }
    if modes:
        result["modes"] = modes
    return result


def merge_task_eval_results(
    existing: Dict[str, Any],
    new: Dict[str, Any],
) -> Dict[str, Any]:
    existing = ensure_modes_from_legacy(existing)
    merged = dict(existing)
    merged_modes = dict(existing.get("modes", {}))
    merged_modes.update(new.get("modes", {}))
    merged["modes"] = merged_modes
    for key, value in new.items():
        if key != "modes":
            merged[key] = value
    gold = merged.get("gold_answer")
    if "prefill_full_decode_full" in merged_modes:
        merged["full_perplexity"] = merged_modes["prefill_full_decode_full"]["perplexity"]
    if "prefill_sparse_decode_sparse" in merged_modes:
        merged["sparse_perplexity"] = merged_modes["prefill_sparse_decode_sparse"]["perplexity"]
    if (
        "prefill_full_decode_full" in merged_modes
        and "prefill_sparse_decode_sparse" in merged_modes
    ):
        ff_ppl = merged["full_perplexity"]["perplexity"]
        ss_ppl = merged["sparse_perplexity"]["perplexity"]
        merged["perplexity_delta"] = ss_ppl - ff_ppl
        merged["loss_delta"] = (
            merged["sparse_perplexity"]["loss"] - merged["full_perplexity"]["loss"]
        )
        merged["generation"] = legacy_generation_from_modes(merged_modes, gold)
    return merged


def _split_ff_reference_roots(spec: Optional[str]) -> List[Path]:
    if spec is None or str(spec).strip() == "":
        return []
    return [
        Path(part.strip()).expanduser()
        for part in str(spec).split(",")
        if part.strip()
    ]


def _iter_task_eval_jsons(root: Path) -> List[Path]:
    if root.is_file():
        return [root]
    roots = [root]
    if root.name not in {"task_eval", "eval"}:
        roots.extend([root / "task_eval", root / "eval"])
    paths: List[Path] = []
    for candidate_root in roots:
        if candidate_root.is_dir():
            paths.extend(sorted(candidate_root.glob("sample_*/task_eval.json")))
    return paths


def _build_ff_reference_index(args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
    cached = getattr(args, "_ff_reference_index", None)
    if cached is not None:
        return cached

    index: Dict[str, Dict[str, Any]] = {}
    for root in _split_ff_reference_roots(getattr(args, "ff_reference_roots", None)):
        loaded = 0
        for path in _iter_task_eval_jsons(root):
            try:
                result = ensure_modes_from_legacy(
                    json.loads(path.read_text(encoding="utf-8"))
                )
            except Exception as exc:
                logger.warning("Skipping invalid ff reference %s: %s", path, exc)
                continue
            sample_id = str(result.get("sample_id", ""))
            mode = result.get("modes", {}).get("prefill_full_decode_full")
            if sample_id and mode:
                index[sample_id] = mode
                loaded += 1
        logger.info("Loaded %d ff reference modes from %s", loaded, root)

    setattr(args, "_ff_reference_index", index)
    return index


def load_ff_reference_mode(
    args: argparse.Namespace,
    sample: Dict[str, Any],
    *,
    gold_answer: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    sample_id = str(sample.get("_id", ""))
    if not sample_id:
        return None
    mode = _build_ff_reference_index(args).get(sample_id)
    if mode is None:
        return None
    reused = json.loads(json.dumps(mode))
    reused["reused_from_ff_reference"] = True
    if gold_answer is not None:
        gen = reused.get("generation", {})
        if gen.get("correct") is None and gen.get("answer") is not None:
            gen["correct"] = gen.get("answer") == gold_answer
    return reused


def compare_generation_outputs(
    full_result: Dict[str, Any],
    sparse_result: Dict[str, Any],
    gold_answer: Optional[str],
) -> Dict[str, Any]:
    full_text = full_result["generated_text"]
    sparse_text = sparse_result["generated_text"]
    full_answer = extract_mcq_answer(full_text)
    sparse_answer = extract_mcq_answer(sparse_text)
    gold = gold_answer.strip().upper() if gold_answer else None

    full_ids = full_result["new_token_ids"]
    sparse_ids = sparse_result["new_token_ids"]
    prefix_match = 0
    for a, b in zip(full_ids, sparse_ids):
        if a == b:
            prefix_match += 1
        else:
            break
    min_len = min(len(full_ids), len(sparse_ids))
    matching = sum(1 for i in range(min_len) if full_ids[i] == sparse_ids[i])
    max_len = max(len(full_ids), len(sparse_ids), 1)

    return {
        "full_generated_text": full_text,
        "sparse_generated_text": sparse_text,
        "full_answer": full_answer,
        "sparse_answer": sparse_answer,
        "gold_answer": gold,
        "full_correct": full_answer == gold if gold else None,
        "sparse_correct": sparse_answer == gold if gold else None,
        "answer_letters_match": full_answer == sparse_answer,
        "exact_text_match": full_text.strip() == sparse_text.strip(),
        "token_match_ratio": matching / max_len,
        "token_prefix_match_length": prefix_match,
        "full_num_new_tokens": len(full_ids),
        "sparse_num_new_tokens": len(sparse_ids),
    }


def run_task_eval_for_sample(
    model,
    tokenizer,
    prompt: str,
    sample: Dict[str, Any],
    layer_to_mask: Dict[int, torch.Tensor],
    *,
    seq_len: int,
    last_q: int,
    args: argparse.Namespace,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    sparse_head_indices: Optional[List[int]] = None,
    debug_layers: Optional[set] = None,
) -> Dict[str, Any]:
    device = next(model.parameters()).device
    if input_ids is None or attention_mask is None:
        input_ids, attention_mask = tokenize_prompt(tokenizer, prompt, args, device)

    mode_combos = parse_eval_mode_combos(args.eval_mode_combos)
    ppl_chunk = resolve_eval_chunk_size(args, seq_len)
    gold = str(sample.get("answer", "")).strip().upper() or None

    modes: Dict[str, Dict[str, Any]] = {}
    ff_key = EVAL_MODE_COMBO_SPECS["ff"]["key"]
    if "ff" in mode_combos:
        ff_reference = load_ff_reference_mode(args, sample, gold_answer=gold)
        if ff_reference is not None:
            modes[ff_key] = ff_reference
            logger.info("  reused ff reference for sample_id=%s", sample.get("_id"))

    # Speed optimization (semantics-preserving):
    # - Prefill_full KV is identical for ff and fs -> compute once, reuse for both decodes
    # - Prefill_sparse KV is identical for sf and ss -> compute once, reuse for both decodes
    # - PPL depends only on prefill, not decode -> compute once per prefill kind and share
    combos_by_prefill: Dict[bool, List[str]] = {False: [], True: []}
    for combo in mode_combos:
        if combo == "ff" and ff_key in modes:
            continue
        combos_by_prefill[bool(EVAL_MODE_COMBO_SPECS[combo]["apply_prefill"])].append(combo)

    for apply_prefill in (False, True):
        combos = combos_by_prefill[apply_prefill]
        if not combos:
            continue

        # Build KV cache + (optional) PPL for this prefill variant.
        prefill_controller = make_mode_controller(
            layer_to_mask,
            apply_prefill=apply_prefill,
            apply_decode=False,
            last_q=last_q,
            analysis_seq_len=seq_len,
            sparse_head_indices=sparse_head_indices,
            debug_layers=debug_layers,
        )

        ppl = {"loss": float("nan"), "perplexity": float("nan"), "num_tokens": 0}
        if args.eval_compute_ppl:
            ppl = compute_sequence_nll_and_ppl(
                model,
                input_ids,
                attention_mask,
                controller=prefill_controller,
                seq_len=seq_len,
                chunk_size=ppl_chunk,
            )
            free_cuda_cache()

        sparse_prefill_patched = apply_prefill and prefill_controller is not None
        if sparse_prefill_patched:
            patch_model_attention(model, prefill_controller)
        try:
            past_key_values, prefill_logits = _chunked_prefill_past_key_values(
                model,
                input_ids,
                attention_mask,
                prefill_controller if apply_prefill else None,
                analysis_seq_len=seq_len,
                chunk_size=ppl_chunk,
            )
        finally:
            if sparse_prefill_patched:
                set_attention_forward_context(None)
                unpatch_model_attention(model)
        free_cuda_cache()

        # Run all decode variants that share this prefill KV.
        for combo in combos:
            spec = EVAL_MODE_COMBO_SPECS[combo]
            mode_key = spec["key"]
            apply_decode = bool(spec["apply_decode"])
            decode_controller = make_mode_controller(
                layer_to_mask,
                apply_prefill=False,
                apply_decode=apply_decode,
                last_q=last_q,
                analysis_seq_len=seq_len,
                sparse_head_indices=sparse_head_indices,
                debug_layers=debug_layers,
            )

            gen = generate_new_tokens_from_past(
                model,
                tokenizer,
                input_ids,
                attention_mask,
                past_key_values,
                controller=decode_controller,
                max_new_tokens=args.eval_max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                analysis_seq_len=seq_len,
                prefill_logits=prefill_logits,
            )
            free_cuda_cache()

            modes[mode_key] = {
                "combo": combo,
                "apply_prefill": bool(spec["apply_prefill"]),
                "apply_decode": apply_decode,
                "perplexity": ppl,
                "generation": build_mode_generation_metrics(gen, gold),
            }

        del past_key_values
        free_cuda_cache()

    result: Dict[str, Any] = {
        "domain": sample.get("domain", sample.get("_selected_domain")),
        "sub_domain": sample.get("sub_domain"),
        "sample_id": str(sample.get("_id")),
        "gold_answer": gold,
        "seq_len": seq_len,
        "eval_mode_combos": mode_combos,
        "eval_apply_prefill": args.eval_apply_prefill,
        "eval_apply_decode": args.eval_apply_decode,
        "modes": modes,
    }

    if "prefill_full_decode_full" in modes:
        result["full_perplexity"] = modes["prefill_full_decode_full"]["perplexity"]
    if "prefill_sparse_decode_sparse" in modes:
        result["sparse_perplexity"] = modes["prefill_sparse_decode_sparse"]["perplexity"]
    if "prefill_full_decode_full" in modes and "prefill_sparse_decode_sparse" in modes:
        result["perplexity_delta"] = (
            result["sparse_perplexity"]["perplexity"]
            - result["full_perplexity"]["perplexity"]
        )
        result["loss_delta"] = (
            result["sparse_perplexity"]["loss"] - result["full_perplexity"]["loss"]
        )
        result["generation"] = legacy_generation_from_modes(modes, gold)
    return result


def run_head_selection_analysis(
    sample: Dict[str, Any],
    sample_idx: int,
    model,
    tokenizer,
    args: argparse.Namespace,
    mask_builder: SparseMaskBuilder,
    similarity_mask_builder: SparseMaskBuilder,
    *,
    output_subdir: str = "head_selection",
) -> Dict[str, Any]:
    """Phase 1: full analysis on one sample to help confirm per-layer representative heads."""
    result = process_sample(
        sample,
        sample_idx,
        model,
        tokenizer,
        args,
        mask_builder,
        similarity_mask_builder,
        sample_dir_name=f"{output_subdir}/sample_{sample_idx:03d}",
        run_task_eval=False,
        return_analysis=True,
    )
    assert result is not None
    return result


def run_task_eval_with_fixed_heads(
    sample: Dict[str, Any],
    eval_idx: int,
    model,
    tokenizer,
    args: argparse.Namespace,
    mask_builder: SparseMaskBuilder,
    global_representative_heads: Dict[int, int],
    *,
    global_heads_meta: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Phase 2: use globally confirmed representative heads (no re-selection),
    build per-sample shared masks, then run accuracy / ppl / consistency eval.
    """
    sample_id = str(sample.get("_id", eval_idx))
    prompt = build_prompt(sample)
    sample_dir = Path(args.output_dir) / "task_eval" / f"sample_{eval_idx:03d}"

    logger.info("=" * 60)
    logger.info(
        "Task eval %03d | id=%s | using global representative heads",
        eval_idx,
        sample_id,
    )

    attentions, seq_len, input_ids, attention_mask = collect_last_q_attentions(
        model, tokenizer, prompt, args
    )
    last_q = attentions.shape[2]
    query_abs_positions = query_abs_positions_from_last_q(seq_len, last_q)

    fixed_heads = {
        int(k): int(v) for k, v in global_representative_heads.items()
    }
    layer_to_mask = build_layer_shared_masks(
        attentions,
        fixed_heads,
        mask_builder,
        query_abs_positions=query_abs_positions,
    )
    del attentions
    free_cuda_cache()

    logger.info(
        "  input_length=%d | fixed_heads from %d selection samples",
        seq_len,
        global_heads_meta.get("num_selection_samples", 0),
    )

    task_eval_result = run_task_eval_for_sample(
        model,
        tokenizer,
        prompt,
        sample,
        layer_to_mask,
        seq_len=seq_len,
        last_q=last_q,
        args=args,
        input_ids=input_ids,
        attention_mask=attention_mask,
        sparse_head_indices=getattr(args, "resolved_sparse_head_indices", None),
        debug_layers=getattr(args, "debug_layers", None),
    )
    if args.save_masks:
        masks_to_save = {str(k): v.cpu() for k, v in layer_to_mask.items()}
    del layer_to_mask
    free_cuda_cache()
    task_eval_result["eval_phase"] = "task_eval_with_fixed_heads"
    task_eval_result["global_representative_heads"] = {
        str(k): v for k, v in fixed_heads.items()
    }
    task_eval_result["head_selection_meta"] = global_heads_meta

    mode_logs: List[str] = []
    for mode_key, mode_data in task_eval_result.get("modes", {}).items():
        gen = mode_data.get("generation", {})
        ppl = mode_data.get("perplexity", {}).get("perplexity", float("nan"))
        mode_logs.append(
            f"{mode_key}: acc={gen.get('correct')} ppl={ppl:.4f} ans={gen.get('answer')}"
        )
    logger.info("  task eval modes | %s", " | ".join(mode_logs))

    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / "input.txt").write_text(prompt, encoding="utf-8")
    task_eval_path = sample_dir / "task_eval.json"
    if task_eval_path.is_file():
        existing = json.loads(task_eval_path.read_text(encoding="utf-8"))
        task_eval_result = merge_task_eval_results(existing, task_eval_result)
    task_eval_path.write_text(
        json.dumps(task_eval_result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (sample_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "sample_id": sample_id,
                "domain": sample.get("domain", sample.get("_selected_domain")),
                "seq_len": seq_len,
                "gold_answer": sample.get("answer"),
                "global_representative_heads": {str(k): v for k, v in fixed_heads.items()},
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    if args.save_masks:
        torch.save(masks_to_save, sample_dir / "shared_masks.pt")

    logger.info("Saved task eval to %s", sample_dir)
    return task_eval_result


def aggregate_global_representative_heads(
    selection_results: List[Dict[str, Any]],
    num_layers: int,
    *,
    representative_selection: str = "coverage",
) -> Dict[str, Any]:
    """
    Confirm per-layer representative heads from multiple head-selection samples.

    Strategy per layer:
      1. Majority vote across selection samples.
      2. Tie-break by summed representative_score among voters for that head.
    """
    per_layer_votes: Dict[int, Dict[int, List[float]]] = {
        layer: {} for layer in range(num_layers)
    }

    for result in selection_results:
        rep_heads = result["representative_heads"]
        scores = result["coverage_scores"]
        for layer_idx in range(num_layers):
            head = int(rep_heads[layer_idx])
            score = float(scores[layer_idx]["representative_score"])
            per_layer_votes[layer_idx].setdefault(head, []).append(score)

    global_heads: Dict[int, int] = {}
    vote_detail: Dict[str, Any] = {}

    for layer_idx in range(num_layers):
        candidates = per_layer_votes[layer_idx]
        if not candidates:
            global_heads[layer_idx] = 0
            continue

        def sort_key(item: Tuple[int, List[float]]) -> Tuple[int, float]:
            head, score_list = item
            return (len(score_list), sum(score_list))

        best_head, best_scores = max(candidates.items(), key=sort_key)
        global_heads[layer_idx] = int(best_head)
        vote_detail[str(layer_idx)] = {
            "representative_head": int(best_head),
            "vote_count": len(best_scores),
            "vote_total": len(selection_results),
            "mean_score_among_voters": sum(best_scores) / len(best_scores),
            "all_candidates": {
                str(h): {
                    "votes": len(sc),
                    "mean_score": sum(sc) / len(sc),
                }
                for h, sc in candidates.items()
            },
        }

    return {
        "representative_heads": global_heads,
        "representative_selection": representative_selection,
        "num_selection_samples": len(selection_results),
        "selection_sample_ids": [r["sample_id"] for r in selection_results],
        "per_layer_vote_detail": vote_detail,
    }


def summarize_task_eval(output_dir: Path, eval_num_samples: int) -> Dict[str, Any]:
    entries: List[Dict[str, Any]] = []
    search_roots = [
        output_dir / "task_eval",
        output_dir,
    ]
    seen: set[str] = set()
    for root in search_roots:
        if not root.is_dir():
            continue
        for sample_dir in sorted(root.glob("sample_*")):
            path = sample_dir / "task_eval.json"
            if path.is_file() and str(path) not in seen:
                seen.add(str(path))
                entries.append(json.loads(path.read_text(encoding="utf-8")))

    entries = entries[:eval_num_samples]
    if not entries:
        return {"evaluated_samples": 0}

    def _mean(key_path: List[str]) -> float:
        vals: List[float] = []
        for e in entries:
            obj: Any = e
            for k in key_path:
                if not isinstance(obj, dict) or k not in obj:
                    obj = None
                    break
                obj = obj[k]
            if obj is not None and math.isfinite(float(obj)):
                vals.append(float(obj))
        return sum(vals) / max(len(vals), 1)

    def _mode_accuracy(mode_key: str) -> Dict[str, Any]:
        correct = [
            e["modes"][mode_key]["generation"]["correct"]
            for e in entries
            if mode_key in e.get("modes", {})
            and e["modes"][mode_key]["generation"].get("correct") is not None
        ]
        return {
            "accuracy": sum(correct) / max(len(correct), 1),
            "correct_count": sum(correct),
            "total_with_gold": len(correct),
        }

    def _mode_ppl_mean(mode_key: str) -> float:
        vals: List[float] = []
        for e in entries:
            mode = e.get("modes", {}).get(mode_key)
            if not mode:
                continue
            ppl = mode.get("perplexity", {}).get("perplexity")
            if ppl is not None and math.isfinite(float(ppl)):
                vals.append(float(ppl))
        return sum(vals) / max(len(vals), 1)

    global_heads_path = output_dir / "global_representative_heads.json"
    global_heads_meta = None
    if global_heads_path.is_file():
        global_heads_meta = json.loads(global_heads_path.read_text(encoding="utf-8"))

    mode_keys = sorted(
        {
            mode_key
            for e in entries
            for mode_key in e.get("modes", {})
        }
    )
    mode_summary: Dict[str, Dict[str, Any]] = {}
    for mode_key in mode_keys:
        mode_summary[mode_key] = {
            **_mode_accuracy(mode_key),
            "perplexity_mean": _mode_ppl_mean(mode_key),
        }

    full_correct = [
        e["generation"]["full_correct"]
        for e in entries
        if isinstance(e.get("generation"), dict)
        and e.get("generation", {}).get("full_correct") is not None
    ]
    sparse_correct = [
        e["generation"]["sparse_correct"]
        for e in entries
        if isinstance(e.get("generation"), dict)
        and e.get("generation", {}).get("sparse_correct") is not None
    ]

    summary = {
        "evaluated_samples": len(entries),
        "head_selection": global_heads_meta,
        "mode_summary": mode_summary,
        "accuracy": {
            "full": sum(full_correct) / max(len(full_correct), 1),
            "sparse": sum(sparse_correct) / max(len(sparse_correct), 1),
            "full_correct_count": sum(full_correct),
            "sparse_correct_count": sum(sparse_correct),
            "total_with_gold": len(full_correct),
        },
        "perplexity": {
            "full_mean": _mean(["full_perplexity", "perplexity"]),
            "sparse_mean": _mean(["sparse_perplexity", "perplexity"]),
            "full_loss_mean": _mean(["full_perplexity", "loss"]),
            "sparse_loss_mean": _mean(["sparse_perplexity", "loss"]),
            "delta_mean": _mean(["perplexity_delta"]),
        },
        "consistency": {
            "answer_letters_match_rate": _mean(["generation", "answer_letters_match"]),
            "exact_text_match_rate": _mean(["generation", "exact_text_match"]),
            "mean_token_match_ratio": _mean(["generation", "token_match_ratio"]),
        },
        "per_sample": entries,
    }

    by_domain: Dict[str, Dict[str, Any]] = {}
    for e in entries:
        domain = str(e.get("domain", "unknown"))
        if domain not in by_domain:
            by_domain[domain] = {
                "count": 0,
                "sparse_correct": 0,
                "full_correct": 0,
                "mode_correct": {mode_key: 0 for mode_key in mode_keys},
            }
        by_domain[domain]["count"] += 1
        if e.get("generation", {}).get("sparse_correct"):
            by_domain[domain]["sparse_correct"] += 1
        if e.get("generation", {}).get("full_correct"):
            by_domain[domain]["full_correct"] += 1
        for mode_key in mode_keys:
            mode_gen = e.get("modes", {}).get(mode_key, {}).get("generation", {})
            if mode_gen.get("correct"):
                by_domain[domain]["mode_correct"][mode_key] += 1
    for domain, stats in by_domain.items():
        stats["sparse_accuracy"] = stats["sparse_correct"] / max(stats["count"], 1)
        stats["full_accuracy"] = stats["full_correct"] / max(stats["count"], 1)
        stats["mode_accuracy"] = {
            mode_key: stats["mode_correct"][mode_key] / max(stats["count"], 1)
            for mode_key in mode_keys
        }
    summary["by_domain"] = by_domain

    out_path = output_dir / "task_eval_summary.json"
    out_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logger.info("Task eval summary (%d samples):", len(entries))
    for mode_key, stats in mode_summary.items():
        logger.info(
            "  %s: acc=%.2f%% ppl=%.4f",
            mode_key,
            100 * stats["accuracy"],
            stats["perplexity_mean"],
        )
    if full_correct:
        logger.info(
            "  legacy accuracy: full=%.2f%% sparse=%.2f%%",
            100 * summary["accuracy"]["full"],
            100 * summary["accuracy"]["sparse"],
        )
    logger.info(
        "  perplexity: full=%.4f sparse=%.4f (delta=%.4f)",
        summary["perplexity"]["full_mean"],
        summary["perplexity"]["sparse_mean"],
        summary["perplexity"]["delta_mean"],
    )
    logger.info(
        "  consistency: answer_match=%.2f%% exact_text=%.2f%% token_match=%.4f",
        100 * summary["consistency"]["answer_letters_match_rate"],
        100 * summary["consistency"]["exact_text_match_rate"],
        summary["consistency"]["mean_token_match_ratio"],
    )
    return summary


# ---------------------------------------------------------------------------
# Post-run filtering
# ---------------------------------------------------------------------------


def _aggregate_sample_metrics(sample_dir: Path) -> Dict[str, Any]:
    layer_stats_path = sample_dir / "layer_stats.json"
    run_meta_path = sample_dir / "run_meta.json"
    if not layer_stats_path.is_file():
        raise FileNotFoundError(f"Missing {layer_stats_path}")

    layer_stats = json.loads(layer_stats_path.read_text(encoding="utf-8"))
    run_meta = (
        json.loads(run_meta_path.read_text(encoding="utf-8"))
        if run_meta_path.is_file()
        else {}
    )

    mean_coverages = [v["mean_coverage"] for v in layer_stats.values()]
    rep_scores = [v["representative_score"] for v in layer_stats.values()]
    sparsities = [v["sparsity"] for v in layer_stats.values()]

    return {
        "sample_dir": str(sample_dir),
        "sample_id": run_meta.get("sample_id"),
        "domain": run_meta.get("domain"),
        "sub_domain": run_meta.get("sub_domain"),
        "seq_len": run_meta.get("seq_len"),
        "prompt_token_len": run_meta.get("prompt_token_len"),
        "num_layers": len(layer_stats),
        "min_layer_mean_coverage": min(mean_coverages) if mean_coverages else 0.0,
        "avg_layer_mean_coverage": sum(mean_coverages) / max(len(mean_coverages), 1),
        "avg_representative_score": sum(rep_scores) / max(len(rep_scores), 1),
        "avg_sparsity": sum(sparsities) / max(len(sparsities), 1),
        "min_sparsity": min(sparsities) if sparsities else 0.0,
        "max_sparsity": max(sparsities) if sparsities else 0.0,
    }


def _passes_filter(metrics: Dict[str, Any], args: argparse.Namespace) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if metrics["min_layer_mean_coverage"] < args.min_layer_mean_coverage:
        reasons.append(
            f"min_layer_mean_coverage {metrics['min_layer_mean_coverage']:.4f} "
            f"< {args.min_layer_mean_coverage}"
        )
    if metrics["avg_representative_score"] < args.min_avg_representative_score:
        reasons.append(
            f"avg_representative_score {metrics['avg_representative_score']:.4f} "
            f"< {args.min_avg_representative_score}"
        )
    if metrics["avg_sparsity"] < args.min_avg_sparsity:
        reasons.append(
            f"avg_sparsity {metrics['avg_sparsity']:.4f} < {args.min_avg_sparsity}"
        )
    if metrics["avg_sparsity"] > args.max_avg_sparsity:
        reasons.append(
            f"avg_sparsity {metrics['avg_sparsity']:.4f} > {args.max_avg_sparsity}"
        )
    return len(reasons) == 0, reasons


def filter_results_after_run(output_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    sample_dirs = sorted(
        [p for p in output_dir.glob("sample_*") if p.is_dir()],
        key=lambda p: p.name,
    )
    all_metrics: List[Dict[str, Any]] = []
    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []

    for sample_dir in sample_dirs:
        metrics = _aggregate_sample_metrics(sample_dir)
        passed, reasons = _passes_filter(metrics, args)
        entry = {**metrics, "passed": passed, "drop_reasons": reasons}
        all_metrics.append(entry)
        if passed:
            kept.append(entry)
        else:
            dropped.append(entry)

        # Per-sample filter sidecar for convenience.
        (sample_dir / "filter_eval.json").write_text(
            json.dumps(entry, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    summary = {
        "output_dir": str(output_dir),
        "criteria": {
            "min_layer_mean_coverage": args.min_layer_mean_coverage,
            "min_avg_representative_score": args.min_avg_representative_score,
            "min_avg_sparsity": args.min_avg_sparsity,
            "max_avg_sparsity": args.max_avg_sparsity,
        },
        "total_samples": len(all_metrics),
        "kept_count": len(kept),
        "dropped_count": len(dropped),
        "kept": kept,
        "dropped": dropped,
        "all": all_metrics,
    }
    summary_path = output_dir / "filtered_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logger.info("Post-run filter: kept %d / %d samples", len(kept), len(all_metrics))
    for entry in all_metrics:
        status = "KEEP" if entry["passed"] else "DROP"
        logger.info(
            "  [%s] %s | domain=%s | min_cov=%.4f | avg_rep=%.4f | avg_sparsity=%.4f",
            status,
            Path(entry["sample_dir"]).name,
            entry.get("domain"),
            entry["min_layer_mean_coverage"],
            entry["avg_representative_score"],
            entry["avg_sparsity"],
        )
        if entry["drop_reasons"]:
            logger.info("       reasons: %s", "; ".join(entry["drop_reasons"]))

    return summary


# ---------------------------------------------------------------------------
# Main per-sample pipeline
# ---------------------------------------------------------------------------


def process_sample(
    sample: Dict[str, Any],
    sample_idx: int,
    model,
    tokenizer,
    args: argparse.Namespace,
    mask_builder: SparseMaskBuilder,
    similarity_mask_builder: SparseMaskBuilder,
    *,
    sample_dir_name: Optional[str] = None,
    run_task_eval: Optional[bool] = None,
    return_analysis: bool = False,
) -> Optional[Dict[str, Any]]:
    sample_id = str(sample.get("_id", sample_idx))
    prompt = build_prompt(sample)
    dir_name = sample_dir_name or f"sample_{sample_idx:03d}"
    sample_dir = Path(args.output_dir) / dir_name

    logger.info("=" * 60)
    logger.info("Sample %03d | id=%s", sample_idx, sample_id)

    # 1) Full attention baseline + last_q extraction.
    attentions, seq_len, input_ids, attention_mask = collect_last_q_attentions(
        model, tokenizer, prompt, args
    )
    last_q = attentions.shape[2]
    query_abs_positions = query_abs_positions_from_last_q(seq_len, last_q)

    logger.info(
        "input_length=%d | last_q=%d | attentions=%s",
        seq_len,
        last_q,
        tuple(attentions.shape),
    )

    num_layers = attentions.shape[0]
    num_heads = attentions.shape[1]
    representative_heads: Dict[int, int] = {}
    coverage_scores: Dict[int, Dict[str, Any]] = {}
    similarity_matrices: Dict[int, torch.Tensor] = {}

    # 2-5) Per-layer similarity + representative head selection.
    for layer_idx in range(num_layers):
        layer_attn = attentions[layer_idx]
        sim = compute_directional_coverage_similarity(
            layer_attn,
            similarity_mask_builder,
            query_abs_positions=query_abs_positions,
        )
        rep_head, score_per_head = select_representative_head_by_mode(
            sim,
            mode=args.representative_selection,
            layer_idx=layer_idx,
            num_heads=num_heads,
            num_layers=num_layers,
            seed=args.seed,
        )

        representative_heads[layer_idx] = rep_head
        similarity_matrices[layer_idx] = sim
        coverage_scores[layer_idx] = {
            "coverage_score_per_head": score_per_head.tolist(),
            "mean_coverage": float(sim.mean().item()),
            "min_coverage": float(sim.min().item()),
            "std_coverage": float(sim.std(unbiased=False).item()),
            "representative_head": rep_head,
            "representative_score": float(score_per_head[rep_head].item()),
        }

        logger.info(
            "  layer %02d | rep_head=%d | rep_score=%.4f | sim_mean=%.4f",
            layer_idx,
            rep_head,
            score_per_head[rep_head].item(),
            sim.mean().item(),
        )

    # 6) Build shared masks.
    layer_to_mask = build_layer_shared_masks(
        attentions,
        representative_heads,
        mask_builder,
        query_abs_positions=query_abs_positions,
    )

    layer_coverage_stats = compute_layer_coverage_stats(
        attentions, layer_to_mask, query_abs_positions=query_abs_positions
    )
    sparsity_stats = compute_sparsity_stats(
        layer_to_mask,
        query_abs_positions=query_abs_positions,
        representative_heads=representative_heads,
        coverage_scores=coverage_scores,
        layer_coverage_stats=layer_coverage_stats,
    )

    layer_stats: Dict[str, Any] = {}
    for layer_idx in range(num_layers):
        layer_key = str(layer_idx)
        layer_stats[layer_key] = {
            "representative_head": representative_heads[layer_idx],
            "representative_score": coverage_scores[layer_idx]["representative_score"],
            "mean_coverage": layer_coverage_stats[layer_idx]["mean_coverage"],
            "min_coverage": layer_coverage_stats[layer_idx]["min_coverage"],
            "std_coverage": layer_coverage_stats[layer_idx]["std_coverage"],
            "keep_ratio": sparsity_stats[layer_key]["keep_ratio"],
            "sparsity": sparsity_stats[layer_key]["sparsity"],
        }
        logger.info(
            "  layer %02d | sparsity=%.4f | mask_mean_cov=%.4f",
            layer_idx,
            layer_stats[layer_key]["sparsity"],
            layer_stats[layer_key]["mean_coverage"],
        )

    logger.info(
        "apply_prefill=%s | apply_decode=%s | generate=%s",
        args.apply_prefill,
        args.apply_decode,
        args.generate,
    )

    # Optional representative head attention maps for saving.
    rep_attn_maps = None
    if args.save_attention_maps:
        rep_attn_maps = torch.stack(
            [
                attentions[layer_idx, representative_heads[layer_idx]]
                for layer_idx in range(num_layers)
            ],
            dim=0,
        )

    full_output = None
    sparse_output = None

    # Optional generation / sparse intervention.
    if args.generate:
        # Full attention generation baseline.
        full_output = run_generation_with_optional_sparse(
            model,
            tokenizer,
            input_ids,
            attention_mask,
            controller=None,
            args=args,
        )

        if args.apply_prefill or args.apply_decode:
            controller = SharedMaskAttentionController(
                layer_to_mask={k: v.clone() for k, v in layer_to_mask.items()},
                apply_prefill=args.apply_prefill,
                apply_decode=args.apply_decode,
                last_q=last_q,
                analysis_seq_len=seq_len,
            )
            patch_model_attention(model, controller)
            try:
                sparse_output = run_generation_with_optional_sparse(
                    model,
                    tokenizer,
                    input_ids,
                    attention_mask,
                    controller=controller,
                    args=args,
                )
            finally:
                unpatch_model_attention(model)
        else:
            sparse_output = None

    save_results(
        sample_dir,
        sample_id=sample_id,
        prompt=prompt,
        seq_len=seq_len,
        sample_meta=sample,
        representative_heads=representative_heads,
        coverage_scores=coverage_scores,
        similarity_matrices=similarity_matrices,
        layer_to_mask=layer_to_mask,
        layer_stats=layer_stats,
        sparsity_stats=sparsity_stats,
        args=args,
        full_output=full_output,
        sparse_output=sparse_output,
        representative_head_attentions=rep_attn_maps,
    )

    logger.info("Saved results to %s", sample_dir)

    if return_analysis:
        return {
            "sample_id": sample_id,
            "sample_idx": sample_idx,
            "representative_heads": representative_heads,
            "coverage_scores": coverage_scores,
            "layer_to_mask": layer_to_mask,
            "seq_len": seq_len,
            "last_q": last_q,
        }
    return None


def main() -> None:
    args = parse_args()
    baseline_cfg: Optional[Dict[str, Any]] = None
    if args.baseline_id:
        baseline_cfg = apply_baseline_config(args)

    args.debug_layers = parse_debug_layers(args.debug_layers)
    if args.debug_layers is not None:
        logger.info("debug_layers=%s (sparse only on these layers)", sorted(args.debug_layers))

    torch.manual_seed(args.seed)

    if args.run_task_eval and not args.skip_head_selection:
        required = args.head_selection_num_samples + args.eval_num_samples
        if args.num_samples < required:
            logger.info(
                "Expanding num_samples from %d to %d (= head_selection %d + eval %d)",
                args.num_samples,
                required,
                args.head_selection_num_samples,
                args.eval_num_samples,
            )
            args.num_samples = required
    elif args.run_task_eval and args.skip_head_selection:
        required = args.head_selection_num_samples + args.eval_num_samples
        if args.num_samples < required:
            args.num_samples = required

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    logger.info("Loading model: %s", args.model_name_or_path)
    model, tokenizer = load_model_and_tokenizer(args)
    args.resolved_sparse_head_indices = parse_sparse_head_indices(
        args.sparse_head_indices, int(model.config.num_attention_heads)
    )
    if args.resolved_sparse_head_indices is not None:
        logger.info(
            "Sparse head routing (DuoAttention proxy): heads=%s",
            args.resolved_sparse_head_indices,
        )

    samples = load_longbench_v2_samples(args)
    mask_builder = get_mask_builder(args)
    # Similarity uses the same top-p style mass rule by default; decoupled via builder.
    similarity_mask_builder = TopPMaskBuilder(top_p=args.top_p)

    if args.run_task_eval:
        if args.skip_head_selection:
            global_heads_path = output_dir / "global_representative_heads.json"
            if not global_heads_path.is_file():
                raise FileNotFoundError(
                    f"--skip_head_selection requires {global_heads_path}"
                )
            serializable = json.loads(global_heads_path.read_text(encoding="utf-8"))
            global_heads_meta = {
                **serializable,
                "representative_heads": {
                    int(k): int(v)
                    for k, v in serializable["representative_heads"].items()
                },
            }
            role_eval = [s for s in samples if s.get("_sample_role") == "task_eval"]
            if len(role_eval) >= args.eval_num_samples:
                eval_samples = role_eval[: args.eval_num_samples]
            else:
                eval_samples = samples[
                    args.head_selection_num_samples : args.head_selection_num_samples
                    + args.eval_num_samples
                ]
            logger.info(
                "Skipping phase-1; loaded global heads from %s",
                global_heads_path,
            )
        else:
            head_samples = samples[: args.head_selection_num_samples]
            eval_samples = samples[
                args.head_selection_num_samples : args.head_selection_num_samples
                + args.eval_num_samples
            ]
            if len(head_samples) < args.head_selection_num_samples:
                raise ValueError(
                    f"Need {args.head_selection_num_samples} head-selection samples, got {len(head_samples)}"
                )
            if len(eval_samples) < args.eval_num_samples:
                raise ValueError(
                    f"Need {args.eval_num_samples} eval samples, got {len(eval_samples)}"
                )

            head_domains = [
                str(s.get("domain", s.get("_selected_domain", "unknown")))
                for s in head_samples
            ]
            logger.info("=" * 60)
            logger.info(
                "Phase 1: head selection on %d samples | domains=%s",
                args.head_selection_num_samples,
                head_domains,
            )
            if len(set(head_domains)) < len(head_domains):
                logger.warning(
                    "Head-selection samples are not from distinct domains: %s",
                    head_domains,
                )
            selection_results: List[Dict[str, Any]] = []
            for idx, sample in enumerate(head_samples):
                selection_results.append(
                    run_head_selection_analysis(
                        sample,
                        idx,
                        model,
                        tokenizer,
                        args,
                        mask_builder,
                        similarity_mask_builder,
                    )
                )

            num_layers = len(selection_results[0]["representative_heads"])
            global_heads_meta = aggregate_global_representative_heads(
                selection_results,
                num_layers,
                representative_selection=args.representative_selection,
            )
            global_heads_path = output_dir / "global_representative_heads.json"
            serializable = {
                **global_heads_meta,
                "representative_heads": {
                    str(k): v for k, v in global_heads_meta["representative_heads"].items()
                },
            }
            global_heads_path.write_text(
                json.dumps(serializable, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info("Saved global representative heads to %s", global_heads_path)

            del selection_results
            free_cuda_cache()
            logger.info("Freed GPU cache before phase-2 task eval")

        eval_domains = [
            str(s.get("domain", s.get("_selected_domain", "unknown")))
            for s in eval_samples
        ]
        if len(eval_samples) < args.eval_num_samples:
            raise ValueError(
                f"Need {args.eval_num_samples} eval samples, got {len(eval_samples)}"
            )

        from collections import Counter

        eval_domain_counts = Counter(eval_domains)
        logger.info("=" * 60)
        logger.info(
            "Phase 2: sparse task eval on %d samples | domain_coverage=%s",
            args.eval_num_samples,
            dict(eval_domain_counts),
        )
        logger.info(
            "  eval domains (%d unique): %s",
            len(eval_domain_counts),
            sorted(eval_domain_counts.keys()),
        )
        for eval_idx, sample in enumerate(eval_samples):
            run_task_eval_with_fixed_heads(
                sample,
                eval_idx,
                model,
                tokenizer,
                args,
                mask_builder,
                global_heads_meta["representative_heads"],
                global_heads_meta=serializable,
            )

        if args.filter_after_run:
            filter_results_after_run(output_dir / "head_selection", args)

        summarize_task_eval(output_dir, args.eval_num_samples)
    else:
        for idx, sample in enumerate(samples):
            process_sample(
                sample,
                idx,
                model,
                tokenizer,
                args,
                mask_builder,
                similarity_mask_builder,
            )

        if args.filter_after_run:
            filter_results_after_run(output_dir, args)

    if args.baseline_id:
        manifest_path = write_experiment_manifest(output_dir, args, baseline_cfg)
        logger.info("Wrote experiment manifest: %s", manifest_path)

    logger.info("Done. Outputs: %s", output_dir)


if __name__ == "__main__":
    main()
