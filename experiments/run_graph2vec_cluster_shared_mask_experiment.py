#!/usr/bin/env python3
"""
Graph2Vec cluster shared sparse mask experiment on LongBench-v2.

Extends the single-representative-head experiment to:
  - Graph2Vec cluster heads into K groups per layer
  - One representative head + shared mask per cluster
  - Per-head mask routing via ClusterSharedMaskAttentionController

Reuses model loading, data loading, attention extraction, mask builders,
and four-mode eval patterns from run_shared_layer_mask_experiment.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.graph2vec_cluster_shared_mask import (
    ClusterSharedMaskAttentionController,
    aggregate_global_cluster_assignments,
    aggregate_global_cluster_representatives,
    build_layer_cluster_masks,
    compute_cluster_layer_stats,
    patch_model_attention_cluster,
    run_layer_clustering_and_selection,
    set_cluster_attention_forward_context,
    unpatch_model_attention_cluster,
)
from experiments.run_shared_layer_mask_experiment import (
    EVAL_MODE_COMBO_SPECS,
    AttentionForwardContext,
    build_prompt,
    collect_last_q_attentions,
    extract_mcq_answer,
    free_cuda_cache,
    get_mask_builder,
    legacy_generation_from_modes,
    load_longbench_v2_samples,
    load_ff_reference_mode,
    load_model_and_tokenizer,
    merge_task_eval_results,
    parse_eval_mode_combos,
    query_abs_positions_from_last_q,
    resolve_eval_chunk_size,
    resolve_max_attn_score_bytes,
    _next_prefill_chunk_end,
    str_to_bool,
    tokenize_prompt,
    TopPMaskBuilder,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Graph2Vec cluster shared sparse mask experiment"
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
        default="/home/ubuntu/work/experiments/outputs/graph2vec_cluster2",
    )
    p.add_argument("--num_samples", type=int, default=53)
    p.add_argument(
        "--samples_per_domain",
        type=int,
        default=None,
        help="If set, load up to N samples per --domains.",
    )
    p.add_argument(
        "--domains",
        type=str,
        nargs="*",
        default=None,
        help="Optional domain filter for data loading.",
    )
    p.add_argument("--sub_domain", type=str, default=None)
    p.add_argument("--difficulty", type=str, default=None)
    p.add_argument("--length", type=str, default=None)
    p.add_argument("--start_line", type=int, default=0)
    p.add_argument("--head_selection_num_samples", type=int, default=3)
    p.add_argument("--eval_num_samples", type=int, default=50)
    p.add_argument("--skip_head_selection", type=str_to_bool, default=False)
    p.add_argument("--max_input_length", type=int, default=32768)
    p.add_argument("--last_q", type=int, default=32)
    p.add_argument("--chunk_size", type=int, default=512)
    p.add_argument("--dtype", type=str, choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--mask_method",
        type=str,
        choices=["top_p", "top_k", "top_p_local"],
        default="top_p",
    )
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=128)
    p.add_argument("--local_window", type=int, default=256)

    p.add_argument("--run_task_eval", type=str_to_bool, default=True)
    p.add_argument(
        "--eval_mode_combos",
        type=str,
        default="ff,sf,fs,ss",
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
    p.add_argument("--eval_max_new_tokens", type=int, default=8)
    p.add_argument("--eval_compute_ppl", type=str_to_bool, default=True)
    p.add_argument("--eval_ppl_chunk_size", type=int, default=None)
    p.add_argument("--do_sample", type=str_to_bool, default=False)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--save_masks", type=str_to_bool, default=False)
    p.add_argument("--save_similarity", type=str_to_bool, default=True)
    p.add_argument("--filter_after_run", type=str_to_bool, default=False)

    # Cluster / Graph2Vec parameters
    p.add_argument(
        "--cluster_method",
        type=str,
        choices=["graph2vec", "svd_kmeans", "bmm"],
        default="graph2vec",
    )
    p.add_argument("--num_head_clusters", type=int, default=2)
    p.add_argument(
        "--binarize_method",
        type=str,
        choices=["top_p", "top_k", "threshold"],
        default="top_p",
    )
    p.add_argument("--binarize_top_p", type=float, default=0.95)
    p.add_argument("--binarize_top_k", type=int, default=128)
    p.add_argument("--binarize_threshold", type=float, default=0.0)
    p.add_argument(
        "--graph_type",
        type=str,
        choices=["bipartite", "directed_token"],
        default="bipartite",
    )
    p.add_argument("--graph2vec_dim", type=int, default=128)
    p.add_argument("--graph2vec_wl_iterations", type=int, default=2)
    p.add_argument("--graph2vec_workers", type=int, default=1)
    p.add_argument("--cluster_seed", type=int, default=42)
    p.add_argument(
        "--svd_components",
        type=int,
        default=8,
        help="TruncatedSVD components for cluster_method=svd_kmeans",
    )
    p.add_argument(
        "--bmm_max_iter",
        type=int,
        default=100,
        help="Max EM iterations for cluster_method=bmm",
    )
    p.add_argument(
        "--bmm_tol",
        type=float,
        default=1e-4,
        help="EM convergence tolerance for cluster_method=bmm",
    )
    p.add_argument(
        "--bmm_n_init",
        type=int,
        default=5,
        help="Number of random initializations for cluster_method=bmm",
    )
    p.add_argument(
        "--sink_tokens",
        type=int,
        default=4,
        help="Number of initial tokens treated as sink in graph node labels",
    )
    p.add_argument("--save_graph2vec_embeddings", type=str_to_bool, default=True)
    p.add_argument(
        "--debug_layers",
        type=str,
        default=None,
        help=(
            "Comma-separated layer indices for Graph2Vec clustering only, e.g. '0,1'. "
            "Other layers use head-id split fallback (for smoke tests)."
        ),
    )

    return p.parse_args()


def parse_debug_layers(spec: Optional[str]) -> Optional[set]:
    if spec is None or not str(spec).strip():
        return None
    layers = {int(x.strip()) for x in str(spec).split(",") if x.strip()}
    if not layers:
        return None
    return layers


SPARSE_BACKEND_META = {
    "sparse_backend": "dense_masked_attention",
    "note": (
        "This implementation evaluates quality impact and sparsity potential, "
        "not real wall-clock speedup."
    ),
}


# ---------------------------------------------------------------------------
# Cluster-aware eval helpers
# ---------------------------------------------------------------------------


def make_cluster_mode_controller(
    layer_to_cluster_masks: Dict[int, Dict[int, torch.Tensor]],
    layer_to_head_cluster: Dict[int, Dict[int, int]],
    *,
    apply_prefill: bool,
    apply_decode: bool,
    last_q: int,
    analysis_seq_len: int,
) -> Optional[ClusterSharedMaskAttentionController]:
    if not apply_prefill and not apply_decode:
        return None
    cloned_masks: Dict[int, Dict[int, torch.Tensor]] = {}
    for layer_idx, cm in layer_to_cluster_masks.items():
        cloned_masks[layer_idx] = {cid: m.clone() for cid, m in cm.items()}
    return ClusterSharedMaskAttentionController(
        layer_to_cluster_masks=cloned_masks,
        layer_to_head_cluster=layer_to_head_cluster,
        apply_prefill=apply_prefill,
        apply_decode=apply_decode,
        last_q=last_q,
        analysis_seq_len=analysis_seq_len,
    )


@torch.inference_mode()
def compute_sequence_nll_and_ppl_cluster(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    controller: Optional[ClusterSharedMaskAttentionController],
    *,
    seq_len: int,
    chunk_size: int,
) -> Dict[str, float]:
    device = input_ids.device
    use_sparse = controller is not None and (
        controller.apply_prefill or controller.apply_decode
    )
    patched = use_sparse
    if patched:
        patch_model_attention_cluster(model, controller)

    num_heads = int(getattr(model.config, "num_attention_heads", 28))
    max_attn_bytes = resolve_max_attn_score_bytes(seq_len)
    requested_chunk = max(1, int(chunk_size))

    if seq_len > 32768:
        logger.info(
            "Long-context PPL prefill (cluster): seq_len=%d requested_chunk=%d max_attn_score_mb=%.0f",
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
                set_cluster_attention_forward_context(
                    AttentionForwardContext(
                        stage="prefill",
                        q_len=chunk_len,
                        kv_len=end,
                        query_abs_start=start,
                        analysis_seq_len=controller.analysis_seq_len,
                    )
                )
            else:
                set_cluster_attention_forward_context(None)

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
        set_cluster_attention_forward_context(None)
        if patched:
            unpatch_model_attention_cluster(model)
        del past_key_values
        free_cuda_cache()

    if total_tokens <= 0:
        return {"loss": float("nan"), "perplexity": float("nan"), "num_tokens": 0}
    loss = total_nll / total_tokens
    ppl = float(math.exp(loss)) if math.isfinite(loss) else float("nan")
    return {"loss": loss, "perplexity": ppl, "num_tokens": total_tokens}


def _decode_loop_cluster(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    past_key_values: Any,
    *,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    controller: Optional[ClusterSharedMaskAttentionController],
    analysis_seq_len: int,
) -> Dict[str, Any]:
    device = input_ids.device
    generated = input_ids
    attn_mask = attention_mask
    new_token_ids: List[int] = []
    eos_id = tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        if controller is not None and controller.apply_decode:
            set_cluster_attention_forward_context(
                AttentionForwardContext(
                    stage="decode",
                    q_len=1,
                    kv_len=generated.shape[1],
                    query_abs_start=generated.shape[1] - 1,
                    analysis_seq_len=analysis_seq_len,
                )
            )
        else:
            set_cluster_attention_forward_context(None)

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

    set_cluster_attention_forward_context(None)
    new_ids_tensor = torch.tensor(new_token_ids, dtype=torch.long)
    text = tokenizer.decode(new_ids_tensor, skip_special_tokens=True)
    return {
        "generated_text": text,
        "new_token_ids": new_token_ids,
        "past_key_values": past_key_values,
    }


@torch.inference_mode()
def _chunked_prefill_cluster(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    controller: Optional[ClusterSharedMaskAttentionController],
    *,
    analysis_seq_len: int,
    chunk_size: int,
) -> Any:
    seq_len = input_ids.shape[1]
    device = input_ids.device
    past_key_values = None
    use_sparse = controller is not None and controller.apply_prefill
    num_heads = int(getattr(model.config, "num_attention_heads", 28))
    max_attn_bytes = resolve_max_attn_score_bytes(seq_len)
    requested_chunk = max(1, int(chunk_size))

    if seq_len > 32768:
        logger.info(
            "Long-context eval prefill (cluster): seq_len=%d requested_chunk=%d max_attn_score_mb=%.0f",
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
            set_cluster_attention_forward_context(
                AttentionForwardContext(
                    stage="prefill",
                    q_len=chunk_len,
                    kv_len=end,
                    query_abs_start=start,
                    analysis_seq_len=analysis_seq_len,
                )
            )
        else:
            set_cluster_attention_forward_context(None)

        out = model(
            input_ids=input_ids[:, start:end],
            attention_mask=attention_mask[:, :end],
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=False,
        )
        past_key_values = out.past_key_values
        del out
        if device.type == "cuda":
            free_cuda_cache()
        start = end

    set_cluster_attention_forward_context(None)
    return past_key_values


@torch.inference_mode()
def generate_new_tokens_cluster(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    controller: Optional[ClusterSharedMaskAttentionController],
    *,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    analysis_seq_len: int,
    prefill_chunk_size: int = 4096,
) -> Dict[str, Any]:
    input_len = input_ids.shape[1]
    use_sparse = controller is not None and (
        controller.apply_prefill or controller.apply_decode
    )
    patched = use_sparse
    if patched:
        patch_model_attention_cluster(model, controller)

    try:
        if input_len > prefill_chunk_size or input_len > 32768:
            past_key_values = _chunked_prefill_cluster(
                model,
                input_ids,
                attention_mask,
                controller if controller is not None and controller.apply_prefill else None,
                analysis_seq_len=analysis_seq_len,
                chunk_size=prefill_chunk_size,
            )
        else:
            if controller is not None and controller.apply_prefill:
                set_cluster_attention_forward_context(
                    AttentionForwardContext(
                        stage="prefill",
                        q_len=input_len,
                        kv_len=input_len,
                        query_abs_start=0,
                        analysis_seq_len=analysis_seq_len,
                    )
                )
            else:
                set_cluster_attention_forward_context(None)
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                output_attentions=False,
            )
            past_key_values = out.past_key_values
            del out

        free_cuda_cache()
        result = _decode_loop_cluster(
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
        )
        del result["past_key_values"]
        free_cuda_cache()
        return result
    finally:
        set_cluster_attention_forward_context(None)
        if patched:
            unpatch_model_attention_cluster(model)
        free_cuda_cache()


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
def generate_new_tokens_cluster_from_past(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    past_key_values: Any,
    controller: Optional[ClusterSharedMaskAttentionController],
    *,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    analysis_seq_len: int,
) -> Dict[str, Any]:
    """Decode-only generation starting from an existing KV cache (cluster controller)."""
    use_sparse = controller is not None and controller.apply_decode
    patched = use_sparse
    if patched:
        patch_model_attention_cluster(model, controller)
    try:
        result = _decode_loop_cluster(
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
        )
        del result["past_key_values"]
        free_cuda_cache()
        return result
    finally:
        set_cluster_attention_forward_context(None)
        if patched:
            unpatch_model_attention_cluster(model)
        free_cuda_cache()


def evaluate_cluster_mode_combo(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_to_cluster_masks: Dict[int, Dict[int, torch.Tensor]],
    layer_to_head_cluster: Dict[int, Dict[int, int]],
    *,
    combo: str,
    seq_len: int,
    last_q: int,
    args: argparse.Namespace,
    gold_answer: Optional[str],
    ppl_chunk: int,
) -> Dict[str, Any]:
    spec = EVAL_MODE_COMBO_SPECS[combo]
    controller = make_cluster_mode_controller(
        layer_to_cluster_masks,
        layer_to_head_cluster,
        apply_prefill=spec["apply_prefill"],
        apply_decode=spec["apply_decode"],
        last_q=last_q,
        analysis_seq_len=seq_len,
    )
    ppl = {"loss": float("nan"), "perplexity": float("nan"), "num_tokens": 0}
    if args.eval_compute_ppl:
        ppl = compute_sequence_nll_and_ppl_cluster(
            model,
            input_ids,
            attention_mask,
            controller=controller,
            seq_len=seq_len,
            chunk_size=ppl_chunk,
        )
        free_cuda_cache()

    gen = generate_new_tokens_cluster(
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


def run_cluster_task_eval_for_sample(
    model,
    tokenizer,
    sample: Dict[str, Any],
    layer_to_cluster_masks: Dict[int, Dict[int, torch.Tensor]],
    layer_to_head_cluster: Dict[int, Dict[int, int]],
    *,
    seq_len: int,
    last_q: int,
    args: argparse.Namespace,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    device = next(model.parameters()).device
    prompt = build_prompt(sample)
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

    combos_by_prefill: Dict[bool, List[str]] = {False: [], True: []}
    for combo in mode_combos:
        if combo == "ff" and ff_key in modes:
            continue
        combos_by_prefill[bool(EVAL_MODE_COMBO_SPECS[combo]["apply_prefill"])].append(combo)

    for apply_prefill in (False, True):
        combos = combos_by_prefill[apply_prefill]
        if not combos:
            continue

        prefill_controller = make_cluster_mode_controller(
            layer_to_cluster_masks,
            layer_to_head_cluster,
            apply_prefill=apply_prefill,
            apply_decode=False,
            last_q=last_q,
            analysis_seq_len=seq_len,
        )

        ppl = {"loss": float("nan"), "perplexity": float("nan"), "num_tokens": 0}
        if args.eval_compute_ppl:
            ppl = compute_sequence_nll_and_ppl_cluster(
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
            patch_model_attention_cluster(model, prefill_controller)
        try:
            past_key_values = _chunked_prefill_cluster(
                model,
                input_ids,
                attention_mask,
                prefill_controller if apply_prefill else None,
                analysis_seq_len=seq_len,
                chunk_size=ppl_chunk,
            )
        finally:
            if sparse_prefill_patched:
                set_cluster_attention_forward_context(None)
                unpatch_model_attention_cluster(model)
        free_cuda_cache()

        for combo in combos:
            spec = EVAL_MODE_COMBO_SPECS[combo]
            mode_key = spec["key"]
            apply_decode = bool(spec["apply_decode"])
            decode_controller = make_cluster_mode_controller(
                layer_to_cluster_masks,
                layer_to_head_cluster,
                apply_prefill=False,
                apply_decode=apply_decode,
                last_q=last_q,
                analysis_seq_len=seq_len,
            )
            gen = generate_new_tokens_cluster_from_past(
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
        "modes": modes,
        "experiment_type": "graph2vec_cluster_shared_mask",
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
        result["generation"] = legacy_generation_from_modes(modes, gold)
    return result


# ---------------------------------------------------------------------------
# Phase 1: head clustering + representative selection
# ---------------------------------------------------------------------------


def run_head_selection_sample(
    sample: Dict[str, Any],
    sample_idx: int,
    model,
    tokenizer,
    args: argparse.Namespace,
    mask_builder,
    similarity_mask_builder,
) -> Dict[str, Any]:
    sample_id = str(sample.get("_id", sample_idx))
    prompt = build_prompt(sample)
    sample_dir = Path(args.output_dir) / "head_selection" / f"sample_{sample_idx:03d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    old_bmm_mu_dir = getattr(args, "_bmm_mu_dir", None)
    args._bmm_mu_dir = str(sample_dir / "bmm_mu")

    logger.info("=" * 60)
    logger.info("Head selection %03d | id=%s", sample_idx, sample_id)

    try:
        attentions, seq_len, _, _ = collect_last_q_attentions(model, tokenizer, prompt, args)
        last_q = attentions.shape[2]
        query_abs_positions = query_abs_positions_from_last_q(seq_len, last_q)
        num_layers, num_heads = attentions.shape[0], attentions.shape[1]

        logger.info(
            "  input_length=%d | last_q=%d | layers=%d | heads=%d",
            seq_len,
            last_q,
            num_layers,
            num_heads,
        )

        layer_clusters: Dict[int, Dict[int, List[int]]] = {}
        layer_cluster_reps: Dict[int, Dict[int, Dict[str, Any]]] = {}
        layer_labels: Dict[int, np.ndarray] = {}
        layer_embeddings: Dict[int, np.ndarray] = {}
        layer_similarity: Dict[int, Dict[int, torch.Tensor]] = {}
        layer_stats: Dict[str, Any] = {}
        layer_graph_stats: Dict[int, Dict[str, Any]] = {}

        for layer_idx in range(num_layers):
            layer_result = run_layer_clustering_and_selection(
                attentions[layer_idx],
                layer_idx,
                query_abs_positions,
                args,
                similarity_mask_builder,
            )
            layer_clusters[layer_idx] = layer_result["clusters"]
            layer_cluster_reps[layer_idx] = layer_result["cluster_representatives"]
            layer_labels[layer_idx] = layer_result["labels"]
            layer_embeddings[layer_idx] = layer_result["embeddings"]
            layer_similarity[layer_idx] = layer_result["similarity_matrices"]
            layer_graph_stats[layer_idx] = layer_result["graph_stats"]

            gs = layer_result["graph_stats"]
            logger.info(
                "  layer %02d graph_stats: avg_nodes=%.1f max_nodes=%d backend=%s",
                layer_idx,
                gs.get("avg_num_nodes", 0),
                gs.get("max_num_nodes", 0),
                gs.get("graph_embedding_backend", "unknown"),
            )

            for cid, stats in layer_result["cluster_representatives"].items():
                logger.info(
                    "  layer %02d cluster %d | heads=%s | rep=%d | score=%.4f",
                    layer_idx,
                    cid,
                    stats["cluster_heads"],
                    stats["representative_head"],
                    stats["representative_score"],
                )

        global_cluster_reps = {
            layer_idx: {
                cid: stats["representative_head"]
                for cid, stats in reps.items()
            }
            for layer_idx, reps in layer_cluster_reps.items()
        }
        layer_to_cluster_masks = build_layer_cluster_masks(
            attentions,
            global_cluster_reps,
            mask_builder,
            query_abs_positions=query_abs_positions,
        )
        layer_to_head_cluster = {
            layer_idx: {h: int(layer_labels[layer_idx][h]) for h in range(num_heads)}
            for layer_idx in range(num_layers)
        }
        layer_stats = compute_cluster_layer_stats(
            attentions,
            layer_to_cluster_masks,
            layer_to_head_cluster,
            global_cluster_reps,
            cluster_rep_stats=layer_cluster_reps,
            query_abs_positions=query_abs_positions,
        )
        for layer_idx, gs in layer_graph_stats.items():
            layer_key = str(layer_idx)
            if layer_key in layer_stats:
                layer_stats[layer_key]["graph_stats"] = gs
            else:
                layer_stats[layer_key] = {"graph_stats": gs}

        (sample_dir / "input.txt").write_text(prompt, encoding="utf-8")

        cluster_assignments_serial = {
            str(layer): {str(cid): heads for cid, heads in clusters.items()}
            for layer, clusters in layer_clusters.items()
        }
        (sample_dir / "cluster_assignments.json").write_text(
            json.dumps(cluster_assignments_serial, indent=2),
            encoding="utf-8",
        )

        cluster_reps_serial = {
            str(layer): {
                str(cid): {
                    k: v
                    for k, v in stats.items()
                    if k != "similarity_matrix"
                }
                for cid, stats in reps.items()
            }
            for layer, reps in layer_cluster_reps.items()
        }
        (sample_dir / "cluster_representatives.json").write_text(
            json.dumps(cluster_reps_serial, indent=2),
            encoding="utf-8",
        )
        (sample_dir / "layer_cluster_stats.json").write_text(
            json.dumps(layer_stats, indent=2),
            encoding="utf-8",
        )

        if args.save_graph2vec_embeddings:
            emb_path = sample_dir / "graph2vec_embeddings.npy"
            stacked = np.stack(
                [layer_embeddings[i] for i in range(num_layers)],
                axis=0,
            )
            np.save(emb_path, stacked)

        if args.save_similarity:
            sim_save = {
                str(layer): {
                    str(cid): layer_similarity[layer][cid].cpu()
                    for cid in layer_similarity[layer]
                }
                for layer in range(num_layers)
                if layer in layer_similarity and layer_similarity[layer]
            }
            if sim_save:
                torch.save(sim_save, sample_dir / "similarity_matrices.pt")

        rep_heads_compat = {
            str(layer): next(iter(reps.values()))["representative_head"]
            for layer, reps in layer_cluster_reps.items()
        }
        (sample_dir / "representative_heads.json").write_text(
            json.dumps(rep_heads_compat, indent=2),
            encoding="utf-8",
        )

        del attentions, layer_to_cluster_masks
        free_cuda_cache()

        return {
            "sample_id": sample_id,
            "sample_idx": sample_idx,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "layer_clusters": layer_clusters,
            "layer_cluster_reps": layer_cluster_reps,
            "layer_stats": layer_stats,
            "layer_graph_stats": layer_graph_stats,
            "seq_len": seq_len,
            "last_q": last_q,
        }
    finally:
        if old_bmm_mu_dir is None:
            if hasattr(args, "_bmm_mu_dir"):
                delattr(args, "_bmm_mu_dir")
        else:
            args._bmm_mu_dir = old_bmm_mu_dir


# ---------------------------------------------------------------------------
# Phase 2: eval with fixed global clusters
# ---------------------------------------------------------------------------


def run_eval_sample_with_global_clusters(
    sample: Dict[str, Any],
    eval_idx: int,
    model,
    tokenizer,
    args: argparse.Namespace,
    mask_builder,
    global_assignments: Dict[str, Any],
    global_reps_meta: Dict[str, Any],
) -> Dict[str, Any]:
    sample_id = str(sample.get("_id", eval_idx))
    prompt = build_prompt(sample)
    sample_dir = Path(args.output_dir) / "eval" / f"sample_{eval_idx:03d}"

    logger.info("=" * 60)
    logger.info("Eval %03d | id=%s | global cluster masks", eval_idx, sample_id)

    attentions, seq_len, input_ids, attention_mask = collect_last_q_attentions(
        model, tokenizer, prompt, args
    )
    last_q = attentions.shape[2]
    query_abs_positions = query_abs_positions_from_last_q(seq_len, last_q)

    layer_to_head_cluster = global_assignments["layer_to_head_cluster"]
    global_cluster_reps = global_reps_meta["layer_cluster_representatives"]

    layer_to_cluster_masks = build_layer_cluster_masks(
        attentions,
        global_cluster_reps,
        mask_builder,
        query_abs_positions=query_abs_positions,
    )

    layer_stats = compute_cluster_layer_stats(
        attentions,
        layer_to_cluster_masks,
        layer_to_head_cluster,
        global_cluster_reps,
        query_abs_positions=query_abs_positions,
    )

    sparsity_stats: Dict[str, Any] = {}
    for layer_key, ldata in layer_stats.items():
        clusters = ldata.get("clusters", {})
        avg_sparsity = sum(c["sparsity"] for c in clusters.values()) / max(
            len(clusters), 1
        )
        avg_keep = sum(c["keep_ratio"] for c in clusters.values()) / max(
            len(clusters), 1
        )
        sparsity_stats[layer_key] = {
            "avg_sparsity": avg_sparsity,
            "avg_keep_ratio": avg_keep,
            "clusters": {
                cid: {"sparsity": c["sparsity"], "keep_ratio": c["keep_ratio"]}
                for cid, c in clusters.items()
            },
        }

    del attentions
    free_cuda_cache()

    task_eval_result = run_cluster_task_eval_for_sample(
        model,
        tokenizer,
        sample,
        layer_to_cluster_masks,
        layer_to_head_cluster,
        seq_len=seq_len,
        last_q=last_q,
        args=args,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    task_eval_result["global_cluster_assignments"] = {
        str(layer): {str(h): cid for h, cid in hc.items()}
        for layer, hc in layer_to_head_cluster.items()
    }
    task_eval_result["global_cluster_representatives"] = {
        str(layer): {str(cid): rep for cid, rep in reps.items()}
        for layer, reps in global_cluster_reps.items()
    }

    mode_logs: List[str] = []
    for mode_key, mode_data in task_eval_result.get("modes", {}).items():
        gen = mode_data.get("generation", {})
        ppl = mode_data.get("perplexity", {}).get("perplexity", float("nan"))
        mode_logs.append(
            f"{mode_key}: acc={gen.get('correct')} ppl={ppl:.4f} ans={gen.get('answer')}"
        )
    logger.info("  eval modes | %s", " | ".join(mode_logs))

    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / "input.txt").write_text(prompt, encoding="utf-8")
    (sample_dir / "layer_cluster_stats.json").write_text(
        json.dumps(layer_stats, indent=2),
        encoding="utf-8",
    )
    (sample_dir / "sparsity_stats.json").write_text(
        json.dumps(sparsity_stats, indent=2),
        encoding="utf-8",
    )

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
                "global_cluster_representatives": {
                    str(layer): {str(cid): rep for cid, rep in reps.items()}
                    for layer, reps in global_cluster_reps.items()
                },
                **SPARSE_BACKEND_META,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    if args.save_masks:
        masks_to_save = {
            str(layer): {str(cid): m.cpu() for cid, m in cm.items()}
            for layer, cm in layer_to_cluster_masks.items()
        }
        torch.save(masks_to_save, sample_dir / "cluster_masks.pt")

    del layer_to_cluster_masks
    free_cuda_cache()
    logger.info("Saved eval results to %s", sample_dir)
    return task_eval_result


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def summarize_eval_results(
    output_dir: Path,
    eval_num_samples: int,
    global_assignments: Optional[Dict[str, Any]] = None,
    global_reps: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    eval_root = output_dir / "eval"
    entries: List[Dict[str, Any]] = []
    if eval_root.is_dir():
        for sample_dir in sorted(eval_root.glob("sample_*")):
            path = sample_dir / "task_eval.json"
            if path.is_file():
                entries.append(json.loads(path.read_text(encoding="utf-8")))
    entries = entries[:eval_num_samples]

    if not entries:
        return {"evaluated_samples": 0}

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
        vals = []
        for e in entries:
            mode = e.get("modes", {}).get(mode_key)
            if not mode:
                continue
            ppl = mode.get("perplexity", {}).get("perplexity")
            if ppl is not None and math.isfinite(float(ppl)):
                vals.append(float(ppl))
        return sum(vals) / max(len(vals), 1)

    mode_keys = sorted({mk for e in entries for mk in e.get("modes", {})})
    mode_summary: Dict[str, Dict[str, Any]] = {}
    ff_key = "prefill_full_decode_full"
    ff_acc = _mode_accuracy(ff_key).get("accuracy", 0.0)

    for mode_key in mode_keys:
        acc_info = _mode_accuracy(mode_key)
        mode_summary[mode_key] = {
            **acc_info,
            "perplexity_mean": _mode_ppl_mean(mode_key),
            "accuracy_delta_vs_ff": acc_info["accuracy"] - ff_acc if ff_key in mode_keys else None,
        }

    # Output consistency vs ff
    consistency: Dict[str, Any] = {}
    if ff_key in mode_keys:
        for mode_key in mode_keys:
            if mode_key == ff_key:
                continue
            match_ratios = []
            letter_matches = []
            for e in entries:
                ff_gen = e["modes"][ff_key]["generation"]
                mode_gen = e["modes"][mode_key]["generation"]
                ff_ids = ff_gen.get("num_new_tokens", 0)
                mode_ids = mode_gen.get("num_new_tokens", 0)
                letter_matches.append(
                    ff_gen.get("answer") == mode_gen.get("answer")
                )
                ff_text = ff_gen.get("generated_text", "").strip()
                mode_text = mode_gen.get("generated_text", "").strip()
                match_ratios.append(1.0 if ff_text == mode_text else 0.0)
            consistency[mode_key] = {
                "answer_letter_match_ratio": sum(letter_matches) / max(len(letter_matches), 1),
                "exact_text_match_ratio": sum(match_ratios) / max(len(match_ratios), 1),
            }

    summary: Dict[str, Any] = {
        "evaluated_samples": len(entries),
        "experiment_type": "graph2vec_cluster_shared_mask",
        "mode_summary": mode_summary,
        "consistency_vs_ff": consistency,
        **SPARSE_BACKEND_META,
    }
    if global_assignments is not None:
        summary["global_cluster_assignments"] = {
            str(layer): {str(cid): heads for cid, heads in clusters.items()}
            for layer, clusters in global_assignments.get("layer_clusters", {}).items()
        }
    if global_reps is not None:
        summary["global_cluster_representatives"] = {
            str(layer): {str(cid): rep for cid, rep in reps.items()}
            for layer, reps in global_reps.get("layer_cluster_representatives", {}).items()
        }

    (output_dir / "eval_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Saved eval_summary.json (%d samples)", len(entries))
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    args.debug_layers = parse_debug_layers(args.debug_layers)
    torch.manual_seed(args.seed)

    if args.debug_layers is not None:
        logger.info(
            "Debug mode: Graph2Vec clustering only on layers %s",
            sorted(args.debug_layers),
        )

    if not args.skip_head_selection:
        required = args.head_selection_num_samples + args.eval_num_samples
        if args.num_samples < required:
            logger.info(
                "Expanding num_samples from %d to %d",
                args.num_samples,
                required,
            )
            args.num_samples = required

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {**vars(args), **SPARSE_BACKEND_META}
    if args.debug_layers is not None:
        run_config["debug_layers"] = sorted(args.debug_layers)
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    logger.info("Loading model: %s", args.model_name_or_path)
    model, tokenizer = load_model_and_tokenizer(args)
    samples = load_longbench_v2_samples(args)
    mask_builder = get_mask_builder(args)
    similarity_mask_builder = TopPMaskBuilder(top_p=args.top_p)

    global_assignments: Optional[Dict[str, Any]] = None
    global_reps_meta: Optional[Dict[str, Any]] = None

    if args.skip_head_selection:
        assign_path = output_dir / "global_cluster_assignments.json"
        reps_path = output_dir / "global_cluster_representatives.json"
        if not assign_path.is_file() or not reps_path.is_file():
            raise FileNotFoundError(
                f"--skip_head_selection requires {assign_path} and {reps_path}"
            )
        global_assignments = json.loads(assign_path.read_text(encoding="utf-8"))
        global_reps_meta = json.loads(reps_path.read_text(encoding="utf-8"))
        global_assignments["layer_to_head_cluster"] = {
            int(layer): {int(h): int(cid) for h, cid in hc.items()}
            for layer, hc in global_assignments["layer_to_head_cluster"].items()
        }
        global_assignments["layer_clusters"] = {
            int(layer): {int(cid): heads for cid, heads in clusters.items()}
            for layer, clusters in global_assignments["layer_clusters"].items()
        }
        global_reps_meta["layer_cluster_representatives"] = {
            int(layer): {int(cid): int(rep) for cid, rep in reps.items()}
            for layer, reps in global_reps_meta["layer_cluster_representatives"].items()
        }
        eval_samples = samples[
            args.head_selection_num_samples : args.head_selection_num_samples
            + args.eval_num_samples
        ]
        logger.info("Skipping phase-1; loaded global clusters from %s", output_dir)
    else:
        head_samples = samples[: args.head_selection_num_samples]
        eval_samples = samples[
            args.head_selection_num_samples : args.head_selection_num_samples
            + args.eval_num_samples
        ]
        if len(head_samples) < args.head_selection_num_samples:
            raise ValueError(
                f"Need {args.head_selection_num_samples} head-selection samples"
            )
        if len(eval_samples) < args.eval_num_samples:
            raise ValueError(f"Need {args.eval_num_samples} eval samples")

        head_domains = [
            str(s.get("domain", s.get("_selected_domain", "unknown")))
            for s in head_samples
        ]
        logger.info("=" * 60)
        logger.info(
            "Phase 1: %s head clustering (k=%d) on %d samples | domains=%s",
            args.cluster_method,
            args.num_head_clusters,
            args.head_selection_num_samples,
            head_domains,
        )

        selection_results: List[Dict[str, Any]] = []
        for idx, sample in enumerate(head_samples):
            selection_results.append(
                run_head_selection_sample(
                    sample,
                    idx,
                    model,
                    tokenizer,
                    args,
                    mask_builder,
                    similarity_mask_builder,
                )
            )

        num_layers = selection_results[0]["num_layers"]
        num_heads = selection_results[0]["num_heads"]

        global_assignments = aggregate_global_cluster_assignments(
            selection_results,
            num_layers,
            num_heads,
            args.num_head_clusters,
        )
        global_reps_meta = aggregate_global_cluster_representatives(
            selection_results,
            global_assignments,
            num_layers,
            args.num_head_clusters,
        )

        assign_serial = {
            "layer_to_head_cluster": {
                str(layer): {str(h): cid for h, cid in hc.items()}
                for layer, hc in global_assignments["layer_to_head_cluster"].items()
            },
            "layer_clusters": {
                str(layer): {str(cid): heads for cid, heads in clusters.items()}
                for layer, clusters in global_assignments["layer_clusters"].items()
            },
            "num_selection_samples": global_assignments["num_selection_samples"],
            "per_head_vote_detail": global_assignments["per_head_vote_detail"],
            "cluster_alignment_records": global_assignments.get(
                "cluster_alignment_records", []
            ),
        }
        reps_serial = {
            "layer_cluster_representatives": {
                str(layer): {str(cid): rep for cid, rep in reps.items()}
                for layer, reps in global_reps_meta[
                    "layer_cluster_representatives"
                ].items()
            },
            "per_cluster_vote_detail": global_reps_meta["per_cluster_vote_detail"],
            "num_selection_samples": global_reps_meta["num_selection_samples"],
        }

        (output_dir / "global_cluster_assignments.json").write_text(
            json.dumps(assign_serial, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (output_dir / "global_cluster_representatives.json").write_text(
            json.dumps(reps_serial, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        head_selection_summary = {
            "num_selection_samples": len(selection_results),
            "sample_ids": [r["sample_id"] for r in selection_results],
            "num_layers": num_layers,
            "num_heads": num_heads,
            "num_head_clusters": args.num_head_clusters,
            "cluster_method": args.cluster_method,
            "binarize_method": args.binarize_method,
            "binarize_top_p": args.binarize_top_p,
            "mask_top_p": args.top_p,
            "debug_layers": sorted(args.debug_layers) if args.debug_layers else None,
            "cluster_alignment_records": global_assignments.get(
                "cluster_alignment_records", []
            ),
            "per_sample_layer_stats": {
                f"sample_{r['sample_idx']:03d}": r["layer_stats"] for r in selection_results
            },
            "per_sample_graph_backends": {
                f"sample_{r['sample_idx']:03d}": {
                    str(layer): gs.get("graph_embedding_backend", "unknown")
                    for layer, gs in r.get("layer_graph_stats", {}).items()
                }
                for r in selection_results
            },
            **SPARSE_BACKEND_META,
        }
        (output_dir / "head_selection_summary.json").write_text(
            json.dumps(head_selection_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Saved global cluster assignments and representatives")

        del selection_results
        free_cuda_cache()

    eval_domains = [
        str(s.get("domain", s.get("_selected_domain", "unknown"))) for s in eval_samples
    ]
    logger.info("=" * 60)
    logger.info(
        "Phase 2: cluster sparse task eval on %d samples | domains=%s",
        args.eval_num_samples,
        dict(Counter(eval_domains)),
    )

    for eval_idx, sample in enumerate(eval_samples):
        run_eval_sample_with_global_clusters(
            sample,
            eval_idx,
            model,
            tokenizer,
            args,
            mask_builder,
            global_assignments,
            global_reps_meta,
        )

    summarize_eval_results(
        output_dir,
        args.eval_num_samples,
        global_assignments=global_assignments,
        global_reps=global_reps_meta,
    )
    logger.info("Done. Outputs: %s", output_dir)


if __name__ == "__main__":
    main()
