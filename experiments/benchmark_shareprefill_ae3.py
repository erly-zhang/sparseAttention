#!/usr/bin/env python3
"""Shared sparse-method benchmark utilities.

Benchmark adapters keep official prompts, truncation, and metrics. This module
installs the same sparse implementations used by the aligned LongBench runner
and records synchronized online inference metrics for every method.
"""

from __future__ import annotations

import json
import hashlib
import math
import sys
import time
import types
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import torch

from experiments.jsd_grouped_sparse import install_grouped_flexprefill
from experiments.token_compacted_sparse import (
    install_grouped_token_compacted_flexprefill,
)


DEFAULT_GROUP_CONFIG = Path(
    "/home/ubuntu/work/experiments/outputs/"
    "stage3_shareprefill_ae_k3_full_blocks_top_p_0.90_500eval/"
    "jsd_head_groups.json"
)
DEFAULT_FLEXPREFILL_ROOT = Path("/home/ubuntu/work/FlexPrefill")
METHODS = (
    "dense",
    "minference",
    "flexprefill",
    "shareprefill_ae3_full",
    "shareprefill_ae3_compact",
    "shareprefill_ae3_hisa",
    "shareprefill_ae3_hisa_mass",
    "shareprefill_ae3_token_block32",
    "shareprefill_ae3_token_block64",
    "shareprefill_ae3_token_block128",
    "shareprefill_ae3_token_block_auto",
    "shareprefill_ae3_token_block_auto_oracle_residual",
    "shareprefill_ae3_token_block_auto_equal_probe",
    "shareprefill_ae3_token_block_auto_full_query_mean",
    "shareprefill_ae3_token_block_auto_equal_probe_fixed_mass_profile",
    "shareprefill_ae3_token_block_auto_equal_probe_member_vs",
    "shareprefill_ae3_token_block_auto_equal_probe_member_vs_mid8_23",
    "shareprefill_ae3_token_block_auto_topp_member_vs",
    "shareprefill_ae8_token_block_auto",
    "shareprefill_ae3_token_block_auto_topp",
    "shareprefill_ae3_token_block_auto_topp_matched",
    "shareprefill_ae3_token_block_auto_topp_unbounded",
    "shareprefill_ae3_token_block_auto_fbeta",
    "shareprefill_ae3_token_block_auto_fixed_mass_profile",
    "shareprefill_ae3_token_block_auto_topp95_mass80",
    "shareprefill_ae3_token_block_auto_topp95_mass80_count67",
    "shareprefill_ae3_token_block_auto_hybrid",
    "shareprefill_ae3_token_block_auto_hybrid_fixed10_topp90",
    "shareprefill_ae3_token_block_auto_dense_topp_mass",
    "shareprefill_per_head_token_topk",
    "shareprefill_ae3_representative_token_topk",
    "shareprefill_ae3_representative_token_topk_protected",
    "shareprefill_ae3_token_block_auto_target_protected",
    "shareprefill_per_head_token_block_auto",
)
BENCHMARK_GROUP_CONFIGS = {
    "infinitebench": Path(
        "/home/ubuntu/work/experiments/outputs/"
        "infinitebench_shareprefill_ae_calibration_k3/"
        "shareprefill_ae_k3_head_groups.json"
    ),
    "ruler": Path(
        "/home/ubuntu/work/experiments/outputs/"
        "ruler_shareprefill_ae_calibration_k3/"
        "shareprefill_ae_k3_head_groups.json"
    ),
}


def resolve_group_config(
    benchmark: str,
    calibration_scope: str,
    explicit_path: Optional[str],
) -> Path:
    if explicit_path:
        return Path(explicit_path)
    if calibration_scope == "longbench_fixed":
        return DEFAULT_GROUP_CONFIG
    try:
        return BENCHMARK_GROUP_CONFIGS[benchmark]
    except KeyError as error:
        raise ValueError(f"No group config registered for {benchmark}") from error


def validate_group_config_scope(
    path: str | Path,
    *,
    benchmark: str,
    calibration_scope: str,
) -> Dict[str, Any]:
    config = load_group_config(path)
    data_path = str(config.get("data_path", ""))
    expected = (
        f"{benchmark}_benchmark_specific_calibration"
        if calibration_scope == "benchmark_specific"
        else "longbench_v2_32k_full_7b.jsonl"
    )
    if expected not in data_path:
        raise ValueError(
            f"Group config scope mismatch: expected {expected!r} in "
            f"data_path, found {data_path!r}"
        )
    return config


def patch_transformers_flash_window_flag() -> None:
    """Support the transformers version used by the official runtimes."""

    try:
        import transformers.modeling_flash_attention_utils as flash_utils

        if not hasattr(flash_utils, "_flash_supports_window_size"):
            flash_utils._flash_supports_window_size = False
    except Exception:
        pass


def patch_minference_dense_decode_fallback() -> None:
    """Use SDPA for the dense decode call when FA2 is unavailable."""

    import torch.nn.functional as F
    import minference.modules.forward as mf_forward

    def sdpa_flash_compatible(
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        position_ids=None,
        dropout=0.0,
        sliding_window=None,
        is_causal=True,
        **kwargs,
    ):
        q = query_states.transpose(1, 2)
        k = key_states.transpose(1, 2)
        v = value_states.transpose(1, 2)
        causal = bool(
            is_causal
            and attention_mask is None
            and q.shape[-2] > 1
        )
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=dropout,
            is_causal=causal,
        )
        return output.transpose(1, 2)

    mf_forward._flash_attention_forward = sdpa_flash_compatible


def require_flash_attention_2() -> str:
    """Require the backend used by the official long-context runtimes."""

    try:
        import flash_attn
    except ImportError as exc:
        raise RuntimeError(
            "MInference long-context evaluation requires FlashAttention-2. "
            "Install flash-attn in the active official runtime environment."
        ) from exc
    return str(flash_attn.__version__)


def load_group_config(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("classification_metric") != (
        "shareprefill_attention_map_autoencoder"
    ):
        raise ValueError("Expected a SharePrefill-AE group configuration")
    num_groups = int(config.get("num_groups_per_layer", -1))
    if num_groups < 1:
        raise ValueError("SharePrefill-AE group configuration has invalid K")
    return config


def install_shareprefill_ae3_full(
    model,
    *,
    group_config_path: str | Path = DEFAULT_GROUP_CONFIG,
    flexprefill_root: str | Path = DEFAULT_FLEXPREFILL_ROOT,
    block_size: int = 128,
    block_top_p: float = 0.9,
    tau: float = 0.1,
    min_budget: int = 1024,
    max_budget: Optional[int] = None,
    force_sink_block: bool = True,
    force_diagonal_block: bool = True,
    dense_layers: Optional[Sequence[int]] = None,
):
    """Install the exact Stage-3 AE3 Full selected-block implementation."""

    flex_root = Path(flexprefill_root)
    if not flex_root.is_dir():
        raise FileNotFoundError(f"FlexPrefill root not found: {flex_root}")
    if str(flex_root) not in sys.path:
        sys.path.insert(0, str(flex_root))
    config = load_group_config(group_config_path)
    patch = install_grouped_flexprefill(
        model,
        config,
        block_size=block_size,
        gamma=block_top_p,
        tau=tau,
        min_budget=min_budget,
        max_budget=max_budget,
        force_sink_block=force_sink_block,
        force_diagonal_block=force_diagonal_block,
        dense_layers=dense_layers,
    )
    return patch, config


def install_shareprefill_ae3_compact(
    model,
    *,
    group_config_path: str | Path = DEFAULT_GROUP_CONFIG,
    flexprefill_root: str | Path = DEFAULT_FLEXPREFILL_ROOT,
    block_size: int = 128,
    block_top_p: float = 0.9,
    tau: float = 0.1,
    min_budget: int = 1024,
    max_budget: Optional[int] = None,
    token_top_p: float = 0.9,
    min_tokens_per_selected_block: int = 16,
    token_chunk_size: int = 32,
    force_sink_block: bool = True,
    force_diagonal_block: bool = True,
):
    """Install AE3 block sparse attention with within-block K compaction."""

    flex_root = Path(flexprefill_root)
    if not flex_root.is_dir():
        raise FileNotFoundError(f"FlexPrefill root not found: {flex_root}")
    if str(flex_root) not in sys.path:
        sys.path.insert(0, str(flex_root))
    config = load_group_config(group_config_path)
    patch = install_grouped_token_compacted_flexprefill(
        model,
        config,
        block_size=block_size,
        gamma=block_top_p,
        tau=tau,
        min_budget=min_budget,
        max_budget=max_budget,
        token_top_p=token_top_p,
        min_tokens_per_selected_block=min_tokens_per_selected_block,
        token_chunk_size=token_chunk_size,
        force_sink_block=force_sink_block,
        force_diagonal_block=force_diagonal_block,
    )
    return patch, config


def install_shareprefill_ae3_hisa(
    model,
    *,
    group_config_path: str | Path = DEFAULT_GROUP_CONFIG,
    flexprefill_root: str | Path = DEFAULT_FLEXPREFILL_ROOT,
    block_size: int = 128,
    candidate_block_count: int = 256,
    final_token_budget: Optional[int] = 8192,
    token_chunk_size: int = 32,
    force_sink_block: bool = True,
    force_diagonal_block: bool = True,
):
    """Install AE3 grouping with HISA-style two-stage global token top-k."""

    flex_root = Path(flexprefill_root)
    if not flex_root.is_dir():
        raise FileNotFoundError(f"FlexPrefill root not found: {flex_root}")
    if str(flex_root) not in sys.path:
        sys.path.insert(0, str(flex_root))
    config = load_group_config(group_config_path)
    patch = install_grouped_token_compacted_flexprefill(
        model,
        config,
        block_size=block_size,
        gamma=0.9,
        tau=0.1,
        min_budget=1024,
        max_budget=None,
        token_top_p=0.9,
        min_tokens_per_selected_block=16,
        token_chunk_size=token_chunk_size,
        force_sink_block=force_sink_block,
        force_diagonal_block=force_diagonal_block,
        selection_mode="hisa_style_global_topk",
        candidate_block_count=candidate_block_count,
        final_token_budget=final_token_budget,
    )
    return patch, config


def install_shareprefill_ae3_hisa_mass(
    model,
    *,
    group_config_path: str | Path = DEFAULT_GROUP_CONFIG,
    flexprefill_root: str | Path = DEFAULT_FLEXPREFILL_ROOT,
    block_size: int = 128,
    candidate_block_count: int = 256,
    final_token_budget: int = 8192,
    probe_weights: tuple[float, float, float, float] = (0.2, 0.3, 0.4, 0.1),
    token_chunk_size: int = 32,
    force_sink_block: bool = True,
    force_diagonal_block: bool = True,
):
    """Install mass-allocated AE3 HISA-style token selection."""

    flex_root = Path(flexprefill_root)
    if not flex_root.is_dir():
        raise FileNotFoundError(f"FlexPrefill root not found: {flex_root}")
    if str(flex_root) not in sys.path:
        sys.path.insert(0, str(flex_root))
    config = load_group_config(group_config_path)
    patch = install_grouped_token_compacted_flexprefill(
        model,
        config,
        block_size=block_size,
        gamma=0.9,
        tau=0.1,
        min_budget=1024,
        max_budget=None,
        token_top_p=0.9,
        min_tokens_per_selected_block=1,
        token_chunk_size=token_chunk_size,
        force_sink_block=force_sink_block,
        force_diagonal_block=force_diagonal_block,
        selection_mode="hisa_style_mass_allocated",
        candidate_block_count=candidate_block_count,
        final_token_budget=final_token_budget,
        probe_weights=probe_weights,
    )
    return patch, config


def install_shareprefill_ae3_token_block_cover(
    model,
    *,
    logical_key_block_size: int | tuple[int, ...],
    group_config_path: str | Path = DEFAULT_GROUP_CONFIG,
    flexprefill_root: str | Path = DEFAULT_FLEXPREFILL_ROOT,
    target_token_budget: Optional[int] = 8192,
    target_token_top_p: Optional[float] = None,
    target_token_min_budget: Optional[int] = None,
    target_top_p_start_layer: Optional[int] = None,
    final_token_budget: Optional[int] = 8192,
    maximum_selected_blocks: Optional[int] = None,
    fill_final_token_budget: bool = False,
    minimum_block_coverage_ratio: float = 0.125,
    minimum_block_target_probability_ratio: Optional[float] = None,
    block_target_probability_top_p: Optional[float] = None,
    block_target_count_coverage_ratio: Optional[float] = None,
    block_target_cost_aware: bool = False,
    block_f_beta: Optional[float] = None,
    measure_fixed_target_probability_mass: bool = False,
    query_score_mode: str = "four_probe_weighted",
    probe_weights: tuple[float, float, float, float] = (0.2, 0.3, 0.4, 0.1),
    token_chunk_size: int = 32,
    force_sink_block: bool = True,
    force_diagonal_block: bool = True,
    force_sink_target_tokens: bool = False,
    force_diagonal_target_tokens: bool = False,
    target_protected_token_span: int = 128,
    adaptive_coverage_tolerance: float = 0.0,
    adaptive_kernel_cost_tolerance: float = 0.0,
    watched_key_ranges: tuple[tuple[int, int], ...] = (),
    force_watched_key_range_blocks: bool = False,
    profile_member_mask_fidelity: bool = False,
    dense_layers: Optional[Sequence[int]] = None,
    per_head_selection: bool = False,
    project_target_to_blocks: bool = True,
    member_vertical_slash: bool = False,
    member_vertical_slash_gamma: float = 0.95,
    member_vertical_slash_min_tokens: int = 1024,
    member_vertical_slash_max_tokens: int = 2048,
    member_vertical_slash_layers: Optional[Sequence[int]] = None,
    residual_mode: str = "none",
    oracle_residual_tokens: int = 0,
    oracle_member_topk_budget: int = 8192,
):
    """Install token-first selection followed by whole-K-block coverage."""

    flex_root = Path(flexprefill_root)
    if not flex_root.is_dir():
        raise FileNotFoundError(f"FlexPrefill root not found: {flex_root}")
    if str(flex_root) not in sys.path:
        sys.path.insert(0, str(flex_root))
    config = load_group_config(group_config_path)
    patch = install_grouped_token_compacted_flexprefill(
        model,
        config,
        block_size=128,
        gamma=0.9,
        tau=0.1,
        min_budget=1024,
        max_budget=None,
        token_top_p=0.9,
        min_tokens_per_selected_block=1,
        token_chunk_size=token_chunk_size,
        force_sink_block=force_sink_block,
        force_diagonal_block=force_diagonal_block,
        force_sink_target_tokens=force_sink_target_tokens,
        force_diagonal_target_tokens=force_diagonal_target_tokens,
        target_protected_token_span=target_protected_token_span,
        selection_mode="token_first_block_cover",
        final_token_budget=final_token_budget,
        maximum_selected_blocks=maximum_selected_blocks,
        fill_final_token_budget=fill_final_token_budget,
        target_token_budget=target_token_budget,
        target_token_top_p=target_token_top_p,
        target_token_min_budget=target_token_min_budget,
        target_top_p_start_layer=target_top_p_start_layer,
        query_score_mode=query_score_mode,
        probe_weights=probe_weights,
        logical_key_block_size=logical_key_block_size,
        minimum_block_coverage_ratio=minimum_block_coverage_ratio,
        minimum_block_target_probability_ratio=(
            minimum_block_target_probability_ratio
        ),
        block_target_probability_top_p=block_target_probability_top_p,
        block_target_count_coverage_ratio=(
            block_target_count_coverage_ratio
        ),
        block_target_cost_aware=block_target_cost_aware,
        block_f_beta=block_f_beta,
        measure_fixed_target_probability_mass=(
            measure_fixed_target_probability_mass
        ),
        adaptive_coverage_tolerance=adaptive_coverage_tolerance,
        adaptive_kernel_cost_tolerance=adaptive_kernel_cost_tolerance,
        watched_key_ranges=watched_key_ranges,
        force_watched_key_range_blocks=force_watched_key_range_blocks,
        profile_member_mask_fidelity=profile_member_mask_fidelity,
        dense_layers=dense_layers,
        per_head_selection=per_head_selection,
        project_target_to_blocks=project_target_to_blocks,
        member_vertical_slash=member_vertical_slash,
        member_vertical_slash_gamma=member_vertical_slash_gamma,
        member_vertical_slash_min_tokens=member_vertical_slash_min_tokens,
        member_vertical_slash_max_tokens=member_vertical_slash_max_tokens,
        member_vertical_slash_layers=member_vertical_slash_layers,
        residual_mode=residual_mode,
        oracle_residual_tokens=oracle_residual_tokens,
        oracle_member_topk_budget=oracle_member_topk_budget,
    )
    return patch, config


def install_benchmark_method(
    model,
    method: str,
    *,
    model_name: str,
    group_config_path: str | Path = DEFAULT_GROUP_CONFIG,
    flexprefill_root: str | Path = DEFAULT_FLEXPREFILL_ROOT,
    dense_layers: Optional[Sequence[int]] = None,
    selector_dump_path: Optional[str | Path] = None,
    record_sparsity: bool = False,
    block_f_beta: float = 2.0,
    fixed_topk_budget: int = 8192,
    target_token_top_p: Optional[float] = None,
    target_top_p_start_layer: Optional[int] = None,
    flexprefill_min_budget: int = 1024,
    watched_key_ranges: tuple[tuple[int, int], ...] = (),
    force_watched_key_range_blocks: bool = False,
    profile_member_mask_fidelity: bool = False,
    oracle_residual_tokens: int = 0,
    oracle_member_topk_budget: int = 8192,
):
    """Install a validated method without changing benchmark inputs.

    The method-specific sparse algorithm and parameters match the existing
    aligned LongBench comparison. The returned model may differ from the input
    object because the official MInference API returns its patched model.
    """

    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    if int(fixed_topk_budget) <= 0:
        raise ValueError("fixed_topk_budget must be positive")
    patch_transformers_flash_window_flag()
    if method == "dense":
        return model, None, {
            "implementation": "dense_reference",
            "model": model_name,
        }
    if method == "minference":
        from minference import MInference
        from experiments.baseline_sparsity import (
            install_baseline_sparsity_instrumentation,
        )

        flash_attn_version = require_flash_attention_2()
        model_key = Path(model_name).name.lower()
        model_type = str(getattr(model.config, "model_type", "")).lower()
        if model_type == "llama" and "llama-3.1-8b-instruct" in model_key:
            sparse_pattern_model = "meta-llama/Llama-3.1-8B-Instruct"
        elif model_type == "qwen2" and "qwen2-7b-instruct" in model_key:
            sparse_pattern_model = "Qwen/Qwen2-7B-Instruct"
        elif model_type == "qwen2" and "qwen2.5-7b-instruct" in model_key:
            sparse_pattern_model = "Qwen/Qwen2.5-7B-Instruct"
        else:
            raise ValueError(
                "No validated MInference sparse pattern for model "
                f"{model_name!r} (model_type={model_type!r})"
            )
        patched_model = MInference("minference", sparse_pattern_model)(model)
        instrumentation = (
            install_baseline_sparsity_instrumentation(
                patched_model, method, selector_dump_path
            )
            if record_sparsity or selector_dump_path is not None
            else None
        )
        return patched_model, instrumentation, {
            "implementation": "official_minference_runtime",
            "model": model_name,
            "sparse_pattern_model": sparse_pattern_model,
            "sparse_pattern_note": (
                "Official model-specific 128k sparse pattern selected from "
                "the checkpoint architecture and local model directory name."
            ),
            "attention_backend": "flash_attention_2",
            "flash_attn_version": flash_attn_version,
            "selector_dump": str(selector_dump_path) if selector_dump_path else None,
            "sparsity_definition": (
                "final-kernel causally valid token pairs / dense causal pairs"
                if record_sparsity or selector_dump_path
                else None
            ),
        }
    if method == "flexprefill":
        flex_root = Path(flexprefill_root)
        if str(flex_root) not in sys.path:
            sys.path.insert(0, str(flex_root))
        from flex_prefill import patch_model
        from experiments.baseline_sparsity import (
            install_baseline_sparsity_instrumentation,
        )

        model_type = str(getattr(model.config, "model_type", "")).lower()
        gamma = 0.95 if model_type == "llama" else 0.9
        config = {
            "block_size": 128,
            "flex_prefill_gamma": gamma,
            "flex_prefill_tau": 0.1,
            "flex_prefill_min_budget": int(flexprefill_min_budget),
            "flex_prefill_max_budget": None,
        }
        patch_model(model, "flex_prefill", config)
        instrumentation = (
            install_baseline_sparsity_instrumentation(
                model, method, selector_dump_path
            )
            if record_sparsity or selector_dump_path is not None
            else None
        )
        return model, instrumentation, {
            "implementation": "official_flexprefill_runtime",
            "model": model_name,
            "online_config": config,
            "selector_dump": str(selector_dump_path) if selector_dump_path else None,
            "sparsity_definition": (
                "final-kernel causally valid token pairs / dense causal pairs"
                if record_sparsity or selector_dump_path
                else None
            ),
        }
    if method in {
        "shareprefill_ae3_compact",
        "shareprefill_ae3_hisa",
        "shareprefill_ae3_hisa_mass",
        "shareprefill_ae3_token_block32",
        "shareprefill_ae3_token_block64",
        "shareprefill_ae3_token_block128",
        "shareprefill_ae3_token_block_auto",
        "shareprefill_ae3_token_block_auto_oracle_residual",
        "shareprefill_ae3_token_block_auto_equal_probe",
        "shareprefill_ae3_token_block_auto_full_query_mean",
        "shareprefill_ae3_token_block_auto_equal_probe_fixed_mass_profile",
        "shareprefill_ae3_token_block_auto_equal_probe_member_vs",
        "shareprefill_ae3_token_block_auto_equal_probe_member_vs_mid8_23",
        "shareprefill_ae3_token_block_auto_topp_member_vs",
        "shareprefill_ae8_token_block_auto",
        "shareprefill_ae3_token_block_auto_topp",
        "shareprefill_ae3_token_block_auto_topp_matched",
        "shareprefill_ae3_token_block_auto_topp_unbounded",
        "shareprefill_ae3_token_block_auto_fbeta",
        "shareprefill_ae3_token_block_auto_fixed_mass_profile",
        "shareprefill_ae3_token_block_auto_topp95_mass80",
        "shareprefill_ae3_token_block_auto_topp95_mass80_count67",
        "shareprefill_ae3_token_block_auto_hybrid",
        "shareprefill_ae3_token_block_auto_hybrid_fixed10_topp90",
        "shareprefill_ae3_token_block_auto_dense_topp_mass",
        "shareprefill_per_head_token_topk",
        "shareprefill_ae3_representative_token_topk",
        "shareprefill_ae3_representative_token_topk_protected",
        "shareprefill_ae3_token_block_auto_target_protected",
        "shareprefill_per_head_token_block_auto",
    }:
        if dense_layers:
            raise ValueError(
                "Dense-layer overrides are not supported by the compact path"
            )
        if method == "shareprefill_ae3_hisa_mass":
            patch, config = install_shareprefill_ae3_hisa_mass(
                model,
                group_config_path=group_config_path,
                flexprefill_root=flexprefill_root,
            )
            return model, patch, {
                "implementation": "shareprefill_ae_k3_hisa_mass_allocated",
                "model": model_name,
                "group_config_path": str(group_config_path),
                "classification_metric": config.get("classification_metric"),
                "online_config": {
                    "block_size": 128,
                    "candidate_block_count": 256,
                    "candidate_token_ceiling": 32768,
                    "block_budget_allocation": "normalized_attention_mass",
                    "minimum_tokens_per_ordinary_candidate_block": 1,
                    "query_probes": ["one_third", "two_thirds", "last", "mean"],
                    "probe_aggregation": "weighted_average",
                    "probe_weights": {
                        "one_third": 0.2,
                        "two_thirds": 0.3,
                        "last": 0.4,
                        "mean": 0.1,
                    },
                    "final_token_budget": 8192,
                    "token_chunk_size": 32,
                    "force_sink_block": True,
                    "force_diagonal_block": True,
                    "protected_tokens_count_within_budget": True,
                    "dense_layers": [],
                },
            }
        diagnostic_methods = {
            "shareprefill_per_head_token_topk": {
                "per_head": True,
                "project_to_blocks": False,
                "protect_within_budget": False,
                "implementation": "per_q_head_fixed_topk_token_mask",
            },
            "shareprefill_ae3_representative_token_topk": {
                "per_head": False,
                "project_to_blocks": False,
                "protect_within_budget": False,
                "implementation": "ae3_representative_fixed_topk_token_mask",
            },
            "shareprefill_ae3_representative_token_topk_protected": {
                "per_head": False,
                "project_to_blocks": False,
                "protect_within_budget": True,
                "implementation": (
                    "ae3_representative_fixed_topk_token_mask_with_"
                    "sink_local_within_budget"
                ),
            },
            "shareprefill_ae3_token_block_auto_target_protected": {
                "per_head": False,
                "project_to_blocks": True,
                "protect_within_budget": False,
                "protect_target_tokens": True,
                "force_blocks": False,
                "implementation": (
                    "ae3_fixed_topk_with_sink_local_target_protection_"
                    "and_unforced_autoblock_projection"
                ),
            },
            "shareprefill_per_head_token_block_auto": {
                "per_head": True,
                "project_to_blocks": True,
                "protect_within_budget": False,
                "implementation": "per_q_head_fixed_topk_autoblock_projection",
            },
        }
        if method in diagnostic_methods:
            diagnostic = diagnostic_methods[method]
            patch, config = install_shareprefill_ae3_token_block_cover(
                model,
                logical_key_block_size=(32, 64, 128),
                group_config_path=group_config_path,
                flexprefill_root=flexprefill_root,
                target_token_budget=8192,
                target_token_top_p=None,
                target_top_p_start_layer=None,
                final_token_budget=(
                    8192
                    if diagnostic["project_to_blocks"]
                    or diagnostic["protect_within_budget"]
                    else None
                ),
                maximum_selected_blocks=None,
                fill_final_token_budget=False,
                minimum_block_coverage_ratio=0.125,
                adaptive_coverage_tolerance=0.02,
                watched_key_ranges=watched_key_ranges,
                force_watched_key_range_blocks=(
                    force_watched_key_range_blocks
                ),
                profile_member_mask_fidelity=profile_member_mask_fidelity,
                force_sink_block=bool(
                    diagnostic.get(
                        "force_blocks", diagnostic["project_to_blocks"]
                    )
                ),
                force_diagonal_block=bool(
                    diagnostic.get(
                        "force_blocks", diagnostic["project_to_blocks"]
                    )
                ),
                force_sink_target_tokens=bool(
                    diagnostic.get("protect_target_tokens", False)
                ),
                force_diagonal_target_tokens=bool(
                    diagnostic.get("protect_target_tokens", False)
                ),
                target_protected_token_span=128,
                per_head_selection=bool(diagnostic["per_head"]),
                project_target_to_blocks=bool(
                    diagnostic["project_to_blocks"]
                ),
            )
            return model, patch, {
                "implementation": diagnostic["implementation"],
                "model": model_name,
                "group_config_path": str(group_config_path),
                "classification_metric": config.get("classification_metric"),
                "num_groups_per_layer": int(config["num_groups_per_layer"]),
                "online_config": {
                    "execution_query_tile_size": 128,
                    "mask_source": (
                        "each_query_head"
                        if diagnostic["per_head"]
                        else "ae_group_representative"
                    ),
                    "group_mask_broadcast": not diagnostic["per_head"],
                    "per_head_selector_chunk_size": (
                        4 if diagnostic["per_head"] else None
                    ),
                    "target_selection": "fixed_top_k",
                    "target_token_budget": 8192,
                    "whole_block_projection": bool(
                        diagnostic["project_to_blocks"]
                    ),
                    "logical_key_block_sizes": (
                        [32, 64, 128]
                        if diagnostic["project_to_blocks"]
                        else []
                    ),
                    "final_whole_block_token_budget": (
                        8192 if diagnostic["project_to_blocks"] else None
                    ),
                    "final_token_budget": (
                        8192 if diagnostic["protect_within_budget"] else None
                    ),
                    "protected_token_span": (
                        128
                        if diagnostic["protect_within_budget"]
                        or diagnostic.get("protect_target_tokens", False)
                        else None
                    ),
                    "protected_tokens_count_within_budget": bool(
                        diagnostic["protect_within_budget"]
                        or diagnostic.get("protect_target_tokens", False)
                    ),
                    "force_sink_target_tokens": bool(
                        diagnostic.get("protect_target_tokens", False)
                    ),
                    "force_diagonal_target_tokens": bool(
                        diagnostic.get("protect_target_tokens", False)
                    ),
                    "minimum_block_coverage_ratio": (
                        0.125 if diagnostic["project_to_blocks"] else None
                    ),
                    "adaptive_coverage_tolerance": (
                        0.02 if diagnostic["project_to_blocks"] else None
                    ),
                    "query_probes": [
                        "one_third",
                        "two_thirds",
                        "last",
                        "mean",
                    ],
                    "probe_aggregation": "weighted_average",
                    "probe_weights": {
                        "one_third": 0.2,
                        "two_thirds": 0.3,
                        "last": 0.4,
                        "mean": 0.1,
                    },
                    "force_sink_block": bool(
                        diagnostic.get(
                            "force_blocks",
                            diagnostic["project_to_blocks"]
                            or diagnostic["protect_within_budget"],
                        )
                    ),
                    "force_diagonal_block": bool(
                        diagnostic.get(
                            "force_blocks",
                            diagnostic["project_to_blocks"]
                            or diagnostic["protect_within_budget"],
                        )
                    ),
                    "force_watched_key_range_blocks": bool(
                        force_watched_key_range_blocks
                    ),
                    "token_chunk_size": 32,
                    "dense_layers": [],
                },
            }
        if method.startswith(
            ("shareprefill_ae3_token_block", "shareprefill_ae8_token_block")
        ):
            block_suffix = method.split("_token_block", 1)[1].lstrip("_")
            oracle_residual = block_suffix == "auto_oracle_residual"
            hybrid_layerwise = block_suffix in {
                "auto_hybrid",
                "auto_hybrid_fixed10_topp90",
            }
            dense_prefix_top_p = block_suffix == "auto_dense_topp_mass"
            configured_top_p_start_layer = (
                int(target_top_p_start_layer)
                if block_suffix in {"auto_hybrid", "auto_dense_topp_mass"}
                and target_top_p_start_layer is not None
                else 10
                if block_suffix == "auto_hybrid_fixed10_topp90"
                else None
            )
            if (
                block_suffix in {"auto_hybrid", "auto_dense_topp_mass"}
                and configured_top_p_start_layer is None
            ):
                raise ValueError(
                    "The configurable hybrid method requires "
                    "target_top_p_start_layer"
                )
            if configured_top_p_start_layer is not None and not (
                0 <= configured_top_p_start_layer <= int(model.config.num_hidden_layers)
            ):
                raise ValueError(
                    "target_top_p_start_layer must be between zero and the "
                    "number of hidden layers"
                )
            probability_top_p = block_suffix in {
                "auto_topp",
                "auto_topp_member_vs",
                "auto_topp_matched",
                "auto_topp_unbounded",
                "auto_fbeta",
                "auto_topp95_mass80",
                "auto_topp95_mass80_count67",
                "auto_hybrid",
                "auto_hybrid_fixed10_topp90",
                "auto_dense_topp_mass",
            }
            equal_probe_topk = block_suffix in {
                "auto_equal_probe",
                "auto_equal_probe_fixed_mass_profile",
                "auto_equal_probe_member_vs",
                "auto_equal_probe_member_vs_mid8_23",
            }
            full_query_mean = block_suffix == "auto_full_query_mean"
            member_vertical_slash = block_suffix in {
                "auto_equal_probe_member_vs",
                "auto_equal_probe_member_vs_mid8_23",
                "auto_topp_member_vs",
            }
            configured_probe_weights = (
                (0.25, 0.25, 0.25, 0.25)
                if probability_top_p or equal_probe_topk
                else (0.2, 0.3, 0.4, 0.1)
            )
            configured_target_min_budget = 1024 if probability_top_p else None
            target_mass_selection = block_suffix in {
                "auto_topp95_mass80",
                "auto_topp95_mass80_count67",
                "auto_dense_topp_mass",
            }
            dual_coverage_selection = (
                block_suffix in {
                    "auto_topp95_mass80_count67",
                    "auto_dense_topp_mass",
                }
            )
            unbounded_final_keys = block_suffix in {
                "auto_topp_unbounded",
                "auto_topp95_mass80",
                "auto_topp95_mass80_count67",
                "auto_dense_topp_mass",
            }
            fbeta_selection = block_suffix == "auto_fbeta"
            fixed_mass_profile = block_suffix in {
                "auto_fixed_mass_profile",
                "auto_equal_probe_fixed_mass_profile",
                "auto_equal_probe_member_vs",
                "auto_equal_probe_member_vs_mid8_23",
            }
            matched_final_budget = block_suffix == "auto_topp_matched"
            if target_token_top_p is not None:
                if not probability_top_p:
                    raise ValueError(
                        "target_token_top_p is only valid for probability top-p methods"
                    )
                if not 0.0 < target_token_top_p <= 1.0:
                    raise ValueError("target_token_top_p must be in (0, 1]")
            configured_target_top_p = (
                float(target_token_top_p)
                if target_token_top_p is not None
                else 0.95
                if target_mass_selection
                else 0.90
                if probability_top_p
                else None
            )
            adaptive = block_suffix in {
                "auto",
                "auto_oracle_residual",
                "auto_equal_probe",
                "auto_full_query_mean",
                "auto_equal_probe_fixed_mass_profile",
                "auto_equal_probe_member_vs",
                "auto_equal_probe_member_vs_mid8_23",
                "auto_topp",
                "auto_topp_member_vs",
                "auto_topp_matched",
                "auto_topp_unbounded",
                "auto_fbeta",
                "auto_fixed_mass_profile",
                "auto_topp95_mass80",
                "auto_topp95_mass80_count67",
                "auto_hybrid",
                "auto_hybrid_fixed10_topp90",
                "auto_dense_topp_mass",
            }
            logical_key_block_sizes = (
                (32, 64, 128) if adaptive else (int(block_suffix),)
            )
            patch, config = install_shareprefill_ae3_token_block_cover(
                model,
                logical_key_block_size=logical_key_block_sizes,
                group_config_path=group_config_path,
                flexprefill_root=flexprefill_root,
                target_token_budget=(
                    int(fixed_topk_budget)
                    if hybrid_layerwise
                    else None
                    if probability_top_p
                    else int(fixed_topk_budget)
                ),
                target_token_top_p=configured_target_top_p,
                target_token_min_budget=configured_target_min_budget,
                target_top_p_start_layer=(
                    None if dense_prefix_top_p else configured_top_p_start_layer
                ),
                final_token_budget=(
                    None
                    if hybrid_layerwise or unbounded_final_keys or fbeta_selection
                    else int(fixed_topk_budget)
                ),
                maximum_selected_blocks=(
                    256 if hybrid_layerwise or dense_prefix_top_p else None
                ),
                fill_final_token_budget=matched_final_budget,
                minimum_block_coverage_ratio=0.125,
                minimum_block_target_probability_ratio=None,
                block_target_probability_top_p=(
                    0.95
                    if dense_prefix_top_p
                    else 0.80
                    if target_mass_selection
                    else None
                ),
                block_target_count_coverage_ratio=(
                    0.80
                    if dense_prefix_top_p
                    else 0.67
                    if dual_coverage_selection
                    else None
                ),
                block_target_cost_aware=dense_prefix_top_p,
                block_f_beta=block_f_beta if fbeta_selection else None,
                measure_fixed_target_probability_mass=fixed_mass_profile,
                query_score_mode=(
                    "full_query_mean"
                    if full_query_mean
                    else "four_probe_weighted"
                ),
                probe_weights=configured_probe_weights,
                adaptive_coverage_tolerance=0.02 if adaptive else 0.0,
                adaptive_kernel_cost_tolerance=(
                    0.0
                    if dual_coverage_selection
                    else 0.02
                    if target_mass_selection
                    else 0.0
                ),
                watched_key_ranges=watched_key_ranges,
                force_watched_key_range_blocks=(
                    force_watched_key_range_blocks
                ),
                profile_member_mask_fidelity=profile_member_mask_fidelity,
                residual_mode="oracle" if oracle_residual else "none",
                oracle_residual_tokens=(
                    int(oracle_residual_tokens) if oracle_residual else 0
                ),
                oracle_member_topk_budget=int(oracle_member_topk_budget),
                member_vertical_slash=member_vertical_slash,
                member_vertical_slash_gamma=0.95,
                member_vertical_slash_min_tokens=1024,
                member_vertical_slash_max_tokens=2048,
                member_vertical_slash_layers=(
                    tuple(range(8, 24))
                    if block_suffix == "auto_equal_probe_member_vs_mid8_23"
                    else None
                ),
                dense_layers=(
                    tuple(range(configured_top_p_start_layer))
                    if dense_prefix_top_p
                    else ()
                ),
            )
            return model, patch, {
                "implementation": (
                    "shareprefill_ae_k3_token_first_block_cover_plus_exact_member_oracle_residual"
                    if oracle_residual
                    else "shareprefill_ae_k3_token_first_full_query_mean_block_cover"
                    if full_query_mean
                    else "shareprefill_ae_k3_dense_prefix_then_probability_top_p_cost_aware_block_cover"
                    if dense_prefix_top_p
                    else "shareprefill_ae_k3_token_first_layerwise_fixed_topk_then_probability_top_p_block_cover"
                    if hybrid_layerwise
                    else "shareprefill_ae_k3_token_first_target_mass_block_cover"
                    if target_mass_selection and not dual_coverage_selection
                    else "shareprefill_ae_k3_token_first_dual_coverage_block_cover"
                    if dual_coverage_selection
                    else (
                        "shareprefill_ae_k3_token_first_fbeta_block_cover"
                        if fbeta_selection
                        else (
                            "shareprefill_ae_k3_token_first_probability_unbounded_block_cover"
                            if unbounded_final_keys
                            else (
                                "shareprefill_ae_k3_token_first_probability_matched_budget_block_cover"
                                if matched_final_budget
                                else (
                                    "shareprefill_ae_k3_token_first_probability_block_cover_plus_member_vertical_slash"
                                    if probability_top_p and member_vertical_slash
                                    else "shareprefill_ae_k3_token_first_probability_block_cover"
                                    if probability_top_p
                                    else "shareprefill_ae_k3_token_first_block_cover_plus_member_vertical_slash"
                                    if member_vertical_slash
                                    else "shareprefill_ae_k3_token_first_block_cover"
                                )
                            )
                        )
                    )
                ).replace(
                    "shareprefill_ae_k3",
                    f"shareprefill_ae_k{int(config['num_groups_per_layer'])}",
                ),
                "model": model_name,
                "group_config_path": str(group_config_path),
                "classification_metric": config.get("classification_metric"),
                "num_groups_per_layer": int(config["num_groups_per_layer"]),
                "online_config": {
                    "execution_query_tile_size": 128,
                    "logical_key_block_sizes": list(logical_key_block_sizes),
                    "block_size_selection": (
                        "fewest_causal_qk_pairs_exact"
                        if dual_coverage_selection
                        else "fewest_causal_qk_pairs_then_larger_within_2pct"
                        if target_mass_selection
                        else "highest_f_beta_then_fewest_kernel_keys"
                        if fbeta_selection
                        else "largest_within_2pp_of_best_coverage"
                        if adaptive
                        else "fixed"
                    ),
                    "adaptive_coverage_tolerance": (
                        0.005 if fbeta_selection else 0.02 if adaptive else 0.0
                    ),
                    "adaptive_kernel_cost_tolerance": (
                        0.0
                        if dual_coverage_selection
                        else 0.02
                        if target_mass_selection
                        else None
                    ),
                    "target_selection": (
                        "dense_prefix_then_minimum_probability_prefix_with_token_floor"
                        if dense_prefix_top_p
                        else "layerwise_fixed_top_k_then_minimum_probability_prefix_with_token_floor"
                        if hybrid_layerwise
                        else "minimum_probability_prefix_with_token_floor"
                        if probability_top_p
                        else "fixed_top_k"
                    ),
                    "target_top_p_start_layer_zero_based": (
                        configured_top_p_start_layer
                        if hybrid_layerwise or dense_prefix_top_p
                        else None
                    ),
                    "target_top_p_start_layer_one_based": (
                        configured_top_p_start_layer + 1
                        if (hybrid_layerwise or dense_prefix_top_p)
                        and configured_top_p_start_layer
                        < int(model.config.num_hidden_layers)
                        else None
                    ),
                    "measure_fixed_target_probability_mass": (
                        fixed_mass_profile
                    ),
                    "watched_key_ranges": [
                        [start, end] for start, end in watched_key_ranges
                    ],
                    "force_watched_key_range_blocks": bool(
                        force_watched_key_range_blocks
                    ),
                    "profile_member_mask_fidelity": (
                        profile_member_mask_fidelity
                    ),
                    "residual_mode": "oracle" if oracle_residual else "none",
                    "shared_topk_budget": (
                        int(fixed_topk_budget) if oracle_residual else None
                    ),
                    "oracle_residual_tokens_per_nonrepresentative_head_row": (
                        int(oracle_residual_tokens) if oracle_residual else 0
                    ),
                    "oracle_member_topk_budget": (
                        int(oracle_member_topk_budget)
                        if oracle_residual
                        else None
                    ),
                    "oracle_nominal_total_target_budget": (
                        int(fixed_topk_budget) + int(oracle_residual_tokens)
                        if oracle_residual
                        else None
                    ),
                    "oracle_representative_residual_policy": (
                        "zero" if oracle_residual else None
                    ),
                    "oracle_residual_projection_policy": (
                        "exact_member_tokens_appended_after_shared_block_projection"
                        if oracle_residual
                        else None
                    ),
                    "oracle_selector_timing_caveat": (
                        "dense_member_qk_oracle_scoring_is_diagnostic_not_optimized"
                        if oracle_residual
                        else None
                    ),
                    "member_vertical_slash": member_vertical_slash,
                    "member_vertical_slash_source": (
                        "each_query_head_last_128_queries"
                        if member_vertical_slash
                        else None
                    ),
                    "member_vertical_slash_gamma": (
                        0.95
                        if member_vertical_slash
                        else None
                    ),
                    "member_vertical_slash_min_tokens": (
                        1024
                        if member_vertical_slash
                        else None
                    ),
                    "member_vertical_slash_max_tokens": (
                        2048
                        if member_vertical_slash
                        else None
                    ),
                    "member_vertical_slash_budget_policy": (
                        "union_outside_autoblock_8192_budget"
                        if member_vertical_slash
                        else None
                    ),
                    "member_vertical_slash_layers_zero_based": (
                        list(range(8, 24))
                        if block_suffix == "auto_equal_probe_member_vs_mid8_23"
                        else list(range(int(model.config.num_hidden_layers)))
                        if member_vertical_slash
                        else []
                    ),
                    "target_token_budget": (
                        int(fixed_topk_budget)
                        if hybrid_layerwise
                        else None
                        if probability_top_p
                        else int(fixed_topk_budget)
                    ),
                    "target_token_top_p": configured_target_top_p,
                    "target_token_min_budget": configured_target_min_budget,
                    "final_whole_block_token_budget": (
                        None
                        if hybrid_layerwise or unbounded_final_keys or fbeta_selection
                        else int(fixed_topk_budget)
                    ),
                    "final_budget_policy": (
                        "maximum_block_count_only"
                        if hybrid_layerwise or dense_prefix_top_p
                        else "fill_to_budget_up_to_causal_capacity"
                        if matched_final_budget
                        else "maximum_only"
                        if not unbounded_final_keys and not fbeta_selection
                        else None
                    ),
                    "maximum_selected_blocks": (
                        {str(size): 256 for size in logical_key_block_sizes}
                        if hybrid_layerwise or dense_prefix_top_p
                        else None
                        if unbounded_final_keys or fbeta_selection
                        else {
                            str(size): int(fixed_topk_budget) // size
                            for size in logical_key_block_sizes
                        }
                    ),
                    "minimum_block_coverage_ratio": (
                        None
                        if target_mass_selection or fbeta_selection
                        else 0.125
                    ),
                    "minimum_block_coverage_policy": (
                        "priority_boundary_then_budget_fill"
                        if matched_final_budget
                        else "hard_filter"
                        if not target_mass_selection and not fbeta_selection
                        else None
                    ),
                    "minimum_block_target_probability_ratio": None,
                    "block_selection_objective": (
                        "cost_aware_cover_95pct_target_mass_and_80pct_target_count"
                        if dense_prefix_top_p
                        else "cover_80pct_target_mass_and_67pct_target_count"
                        if dual_coverage_selection
                        else "cover_80pct_of_target_probability_mass"
                        if target_mass_selection
                        else "maximize_f_beta_over_density_sorted_block_prefixes"
                        if fbeta_selection
                        else None
                    ),
                    "block_target_probability_top_p": (
                        0.95
                        if dense_prefix_top_p
                        else 0.80
                        if target_mass_selection
                        else None
                    ),
                    "block_target_count_coverage_ratio": (
                        0.80
                        if dense_prefix_top_p
                        else 0.67
                        if dual_coverage_selection
                        else None
                    ),
                    "block_target_cost_aware": dense_prefix_top_p,
                    "block_f_beta": block_f_beta if fbeta_selection else None,
                    "f_beta_tie_tolerance": 0.005 if fbeta_selection else None,
                    "block_probability_ratio_denominator": None,
                    "minimum_target_tokens_per_block": (
                        None
                        if target_mass_selection or fbeta_selection
                        else {
                            str(size): math.ceil(size * 0.125)
                            for size in logical_key_block_sizes
                        }
                    ),
                    "query_score_mode": (
                        "full_query_causal_logit_mean"
                        if full_query_mean
                        else "four_probe_weighted"
                    ),
                    "query_probes": (
                        ["all_legal_query_tokens_in_tile"]
                        if full_query_mean
                        else ["one_third", "two_thirds", "last", "mean"]
                    ),
                    "probe_aggregation": (
                        "exact_arithmetic_mean_over_causally_legal_query_logits"
                        if full_query_mean
                        else "weighted_average"
                    ),
                    "probe_weights": (
                        None
                        if full_query_mean
                        else {
                            "one_third": configured_probe_weights[0],
                            "two_thirds": configured_probe_weights[1],
                            "last": configured_probe_weights[2],
                            "mean": configured_probe_weights[3],
                        }
                    ),
                    "force_sink_block": True,
                    "force_diagonal_block": True,
                    "protected_blocks_count_within_budget": (
                        dense_prefix_top_p
                        or (not unbounded_final_keys and not fbeta_selection)
                    ),
                    "protected_blocks_included_in_f_beta": fbeta_selection,
                    "protected_blocks_included_in_target_mass_goal": (
                        target_mass_selection
                    ),
                    "dense_layers": (
                        list(range(configured_top_p_start_layer))
                        if dense_prefix_top_p
                        else []
                    ),
                },
            }
        if method == "shareprefill_ae3_hisa":
            patch, config = install_shareprefill_ae3_hisa(
                model,
                group_config_path=group_config_path,
                flexprefill_root=flexprefill_root,
            )
            return model, patch, {
                "implementation": "shareprefill_ae_k3_hisa_style_global_topk",
                "model": model_name,
                "group_config_path": str(group_config_path),
                "classification_metric": config.get("classification_metric"),
                "online_config": {
                    "block_size": 128,
                    "candidate_block_count": 256,
                    "candidate_token_ceiling": 32768,
                    "query_probes": ["one_third", "two_thirds", "last", "mean"],
                    "probe_aggregation": "maximum",
                    "final_token_budget": 8192,
                    "token_chunk_size": 32,
                    "force_sink_block": True,
                    "force_diagonal_block": True,
                    "protected_tokens_count_within_budget": True,
                    "dense_layers": [],
                },
            }
        patch, config = install_shareprefill_ae3_compact(
            model,
            group_config_path=group_config_path,
            flexprefill_root=flexprefill_root,
        )
        return model, patch, {
            "implementation": "shareprefill_ae_k3_token_compacted_blocks",
            "model": model_name,
            "group_config_path": str(group_config_path),
            "classification_metric": config.get("classification_metric"),
            "online_config": {
                "block_size": 128,
                "block_top_p": 0.9,
                "tau": 0.1,
                "min_budget": 1024,
                "token_top_p": 0.9,
                "min_tokens_per_selected_block": 16,
                "token_chunk_size": 32,
                "force_sink_block": True,
                "force_diagonal_block": True,
                "within_block_token_compaction": True,
                "dense_layers": [],
            },
        }

    patch, config = install_shareprefill_ae3_full(
        model,
        group_config_path=group_config_path,
        flexprefill_root=flexprefill_root,
        dense_layers=dense_layers,
    )
    return model, patch, {
        "implementation": "shareprefill_ae_k3_full_blocks",
        "model": model_name,
        "group_config_path": str(group_config_path),
        "classification_metric": config.get("classification_metric"),
        "online_config": {
            "block_size": 128,
            "block_top_p": 0.9,
            "tau": 0.1,
            "min_budget": 1024,
            "force_sink_block": True,
            "force_diagonal_block": True,
            "within_block_token_compaction": False,
            "dense_layers": list(patch.dense_layers),
            "dense_implementation": "original_flash_attention_2_forward",
        },
    }


def _stats_delta(
    after: Mapping[str, Any], before: Mapping[str, Any], key: str
) -> float:
    return float(after.get(key, 0)) - float(before.get(key, 0))


def _nested_stats_delta(
    after: Mapping[Any, Any], before: Mapping[Any, Any]
) -> Dict[str, Any]:
    delta: Dict[str, Any] = {}
    for key in set(before) | set(after):
        after_value = after.get(key)
        before_value = before.get(key)
        if isinstance(after_value, Mapping) or isinstance(before_value, Mapping):
            delta[str(key)] = _nested_stats_delta(
                after_value if isinstance(after_value, Mapping) else {},
                before_value if isinstance(before_value, Mapping) else {},
            )
        else:
            delta[str(key)] = float(after_value or 0.0) - float(
                before_value or 0.0
            )
    return delta


def _finalize_watched_range_stats(
    values: Mapping[str, Any]
) -> Dict[str, float]:
    finalized = {str(key): float(value) for key, value in values.items()}
    selector_rows = finalized.get("selector_rows", 0.0)
    valid_slots = finalized.get("valid_token_slots", 0.0)
    finalized["mean_probability_mass_per_selector_row"] = (
        finalized.get("probability_mass", 0.0) / selector_rows
        if selector_rows
        else float("nan")
    )
    for source, output in (
        ("target_token_slots", "target_keep_ratio"),
        ("selected_token_slots", "final_kernel_keep_ratio"),
        ("sink_token_slots", "sink_protection_ratio"),
    ):
        finalized[output] = (
            finalized.get(source, 0.0) / valid_slots
            if valid_slots
            else float("nan")
        )
    valid_pairs = finalized.get("valid_causal_pairs", 0.0)
    if valid_pairs:
        finalized["exact_causal_pair_coverage_ratio"] = (
            finalized.get("selected_causal_pairs", 0.0) / valid_pairs
        )
    return finalized


def input_ids_sha256(input_ids: torch.Tensor) -> str:
    """Create a stable cross-method identity for a tokenized prompt."""

    values = input_ids.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(values).hexdigest()


def read_jsonl(path: str | Path) -> list[Dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate_input_alignment(
    reference_path: str | Path,
    actual_rows: list[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Require matching tokenized inputs without assuming a limited-run prefix."""

    reference_rows = read_jsonl(reference_path)
    if len(actual_rows) > len(reference_rows):
        raise RuntimeError(
            "Input alignment count mismatch: "
            f"reference={len(reference_rows)} actual={len(actual_rows)}"
        )
    identity_fields = (
        "benchmark",
        "task",
        "sequence_length",
        "sample_index",
        "input_tokens",
        "input_ids_sha256",
    )
    def row_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return tuple(row.get(field) for field in identity_fields)

    if len(actual_rows) == len(reference_rows):
        for index, (reference, actual) in enumerate(
            zip(reference_rows, actual_rows)
        ):
            if row_identity(reference) != row_identity(actual):
                for field in identity_fields:
                    if reference.get(field) != actual.get(field):
                        raise RuntimeError(
                            "Input alignment mismatch at call "
                            f"{index}, field={field}: "
                            f"reference={reference.get(field)!r} "
                            f"actual={actual.get(field)!r}"
                        )
    else:
        # lm-eval sorts generation requests by tokenized length. A limited smoke
        # run therefore forms an ordered subsequence, not a prefix, of a full run.
        reference_cursor = 0
        for actual_index, actual in enumerate(actual_rows):
            actual_identity = row_identity(actual)
            while (
                reference_cursor < len(reference_rows)
                and row_identity(reference_rows[reference_cursor])
                != actual_identity
            ):
                reference_cursor += 1
            if reference_cursor == len(reference_rows):
                raise RuntimeError(
                    "Input alignment mismatch: smoke call "
                    f"{actual_index} is not present in the remaining ordered "
                    "reference rows"
                )
            reference_cursor += 1
    return {
        "status": "passed",
        "reference_metrics": str(reference_path),
        "count": len(actual_rows),
        "reference_count": len(reference_rows),
        "alignment_scope": (
            "full"
            if len(reference_rows) == len(actual_rows)
            else "ordered_subsequence"
        ),
        "compared_fields": list(identity_fields),
    }


class GenerationMetricsRecorder:
    """Wrap ``model.generate`` and append one online metric row per call."""

    def __init__(
        self,
        model,
        patch,
        output_path: str | Path,
        *,
        method: str = "shareprefill_ae3_full",
        watched_range_resolver: Optional[
            Callable[[torch.Tensor], tuple[tuple[int, int], ...]]
        ] = None,
    ) -> None:
        self.model = model
        self.patch = patch
        self.method = method
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.rows: list[Dict[str, Any]] = []
        self.context: Dict[str, Any] = {}
        self._original_generate = model.generate
        self._installed = False
        self.watched_range_resolver = watched_range_resolver

    def set_context(self, **context: Any) -> None:
        self.context = dict(context)

    def reset(self) -> None:
        self.rows = []
        if self.patch is not None and hasattr(self.patch, "reset_stats"):
            self.patch.reset_stats()
        elif self.patch is not None and hasattr(self.patch, "selector"):
            stats = self.patch.selector.stats
            self.patch.selector.stats = type(stats)()
        self.output_path.write_text("", encoding="utf-8")

    def install(self) -> None:
        if self._installed:
            return
        recorder = self

        def wrapped_generate(model_self, *args, **kwargs):
            input_ids = kwargs.get("input_ids")
            if input_ids is None and args:
                input_ids = args[0]
            if input_ids is None:
                raise ValueError("generate call did not provide input_ids")
            token_hash = input_ids_sha256(input_ids)
            resolved_watched_ranges: tuple[tuple[int, int], ...] = ()
            if recorder.watched_range_resolver is not None:
                resolved_watched_ranges = recorder.watched_range_resolver(
                    input_ids
                )
                if (
                    recorder.patch is None
                    or not hasattr(recorder.patch, "selector")
                ):
                    raise RuntimeError(
                        "Watched-range profiling requires a selector patch"
                    )
                recorder.patch.selector.watched_key_ranges = (
                    resolved_watched_ranges
                )
            if recorder.patch is not None and hasattr(
                recorder.patch, "begin_sample"
            ):
                recorder.patch.begin_sample(
                    call_index=len(recorder.rows),
                    input_ids_sha256=token_hash,
                    input_tokens=int(input_ids.shape[1]),
                    context=recorder.context,
                )

            stats_source = (
                recorder.patch.selector.stats
                if recorder.patch is not None
                and hasattr(recorder.patch, "selector")
                else None
            )
            before = stats_source.snapshot() if stats_source else {}
            timing = {"started": False, "finished": False}
            prefill_start = torch.cuda.Event(enable_timing=True)
            prefill_end = torch.cuda.Event(enable_timing=True)

            def before_forward(_module, _args, forward_kwargs):
                ids = forward_kwargs.get("input_ids")
                if (
                    not timing["started"]
                    and ids is not None
                    and ids.ndim == 2
                    and ids.shape[1] > 1
                ):
                    prefill_start.record()
                    timing["started"] = True

            def after_forward(_module, _args, _kwargs, _output):
                if timing["started"] and not timing["finished"]:
                    prefill_end.record()
                    timing["finished"] = True

            pre_handle = model_self.register_forward_pre_hook(
                before_forward, with_kwargs=True
            )
            post_handle = model_self.register_forward_hook(
                after_forward, with_kwargs=True
            )
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            total_start = time.perf_counter()
            try:
                output_ids = recorder._original_generate(*args, **kwargs)
            except BaseException:
                if recorder.patch is not None and hasattr(
                    recorder.patch, "abort_sample"
                ):
                    recorder.patch.abort_sample()
                raise
            finally:
                pre_handle.remove()
                post_handle.remove()
            torch.cuda.synchronize()
            total_seconds = time.perf_counter() - total_start
            if not timing["finished"]:
                raise RuntimeError("Could not identify the prefill forward")
            prefill_seconds = prefill_start.elapsed_time(prefill_end) / 1000.0
            if recorder.patch is not None and hasattr(
                recorder.patch, "flush_stats"
            ):
                recorder.patch.flush_stats()
            after = stats_source.snapshot() if stats_source else {}
            dump_metadata = (
                recorder.patch.end_sample()
                if recorder.patch is not None
                and hasattr(recorder.patch, "end_sample")
                else {}
            )
            selected_blocks = _stats_delta(after, before, "selected_blocks")
            causal_blocks = _stats_delta(after, before, "causal_blocks")
            selected_token_pairs = _stats_delta(
                after, before, "selected_token_pairs"
            )
            causal_token_pairs = _stats_delta(
                after, before, "causal_token_pairs"
            )
            compacted_key_tokens = _stats_delta(
                after, before, "compacted_key_tokens"
            )
            candidate_key_tokens = _stats_delta(
                after, before, "candidate_key_tokens"
            )
            target_mask_tokens = _stats_delta(
                after, before, "target_mask_tokens"
            )
            covered_target_tokens = _stats_delta(
                after, before, "covered_target_tokens"
            )
            target_probability_mass = _stats_delta(
                after, before, "target_probability_mass"
            )
            covered_target_probability_mass = _stats_delta(
                after, before, "covered_target_probability_mass"
            )
            watched_key_range_stats = _nested_stats_delta(
                after.get("watched_key_range_stats", {}),
                before.get("watched_key_range_stats", {}),
            )
            layer_watched_key_range_stats = _nested_stats_delta(
                after.get("layer_watched_key_range_stats", {}),
                before.get("layer_watched_key_range_stats", {}),
            )
            member_mask_fidelity_stats = _nested_stats_delta(
                after.get("member_mask_fidelity_stats", {}),
                before.get("member_mask_fidelity_stats", {}),
            )
            for layer, members in member_mask_fidelity_stats.items():
                after_members = after.get("member_mask_fidelity_stats", {}).get(
                    int(layer), {}
                )
                for head, values in members.items():
                    after_values = after_members.get(int(head), {})
                    for name in (
                        "group_index",
                        "representative_head",
                        "kv_head",
                        "is_representative",
                    ):
                        if name in after_values:
                            values[name] = int(after_values[name])
            layer_target_selection_mode_rows = _nested_stats_delta(
                after.get("layer_target_selection_mode_rows", {}),
                before.get("layer_target_selection_mode_rows", {}),
            )
            before_block_sizes = before.get("chosen_block_size_rows", {})
            after_block_sizes = after.get("chosen_block_size_rows", {})
            chosen_block_size_rows = {
                str(size): int(after_block_sizes.get(size, 0))
                - int(before_block_sizes.get(size, 0))
                for size in set(before_block_sizes) | set(after_block_sizes)
            }
            before_candidate_pairs = before.get(
                "candidate_kernel_pairs_by_size", {}
            )
            after_candidate_pairs = after.get(
                "candidate_kernel_pairs_by_size", {}
            )
            candidate_kernel_pairs_by_size = {
                str(size): int(after_candidate_pairs.get(size, 0))
                - int(before_candidate_pairs.get(size, 0))
                for size in set(before_candidate_pairs) | set(after_candidate_pairs)
            }
            layer_integer_stat_names = (
                "layer_selected_key_tokens",
                "layer_candidate_key_tokens",
                "layer_target_mask_tokens",
                "layer_covered_target_tokens",
                "layer_selected_blocks",
                "layer_selected_token_pairs",
                "layer_selector_rows",
            )
            per_layer_stats: Dict[str, Dict[str, Any]] = {}
            for stat_name in layer_integer_stat_names:
                before_values = before.get(stat_name, {})
                after_values = after.get(stat_name, {})
                for layer in set(before_values) | set(after_values):
                    layer_key = str(layer)
                    per_layer_stats.setdefault(layer_key, {})[
                        stat_name.removeprefix("layer_")
                    ] = int(after_values.get(layer, 0)) - int(
                        before_values.get(layer, 0)
                    )
            for stat_name in (
                "layer_target_probability_mass",
                "layer_covered_target_probability_mass",
            ):
                before_values = before.get(stat_name, {})
                after_values = after.get(stat_name, {})
                for layer in set(before_values) | set(after_values):
                    layer_key = str(layer)
                    per_layer_stats.setdefault(layer_key, {})[
                        stat_name.removeprefix("layer_")
                    ] = float(after_values.get(layer, 0.0)) - float(
                        before_values.get(layer, 0.0)
                    )
            for stat_name, output_name in (
                ("layer_chosen_block_size_rows", "chosen_block_size_rows"),
                (
                    "layer_candidate_kernel_pairs_by_size",
                    "candidate_kernel_pairs_by_size",
                ),
            ):
                before_layers = before.get(stat_name, {})
                after_layers = after.get(stat_name, {})
                for layer in set(before_layers) | set(after_layers):
                    before_values = before_layers.get(layer, {})
                    after_values = after_layers.get(layer, {})
                    per_layer_stats.setdefault(str(layer), {})[output_name] = {
                        str(size): int(after_values.get(size, 0))
                        - int(before_values.get(size, 0))
                        for size in set(before_values) | set(after_values)
                    }
            for layer, mode_rows in layer_target_selection_mode_rows.items():
                per_layer_stats.setdefault(str(layer), {})[
                    "target_selection_mode_rows"
                ] = {
                    str(mode): int(count) for mode, count in mode_rows.items()
                }
            for values in per_layer_stats.values():
                rows_for_layer = values.get("selector_rows", 0)
                values["mean_selected_key_tokens_per_row"] = (
                    values.get("selected_key_tokens", 0) / rows_for_layer
                    if rows_for_layer
                    else float("nan")
                )
                values["mean_target_mask_tokens_per_row"] = (
                    values.get("target_mask_tokens", 0) / rows_for_layer
                    if rows_for_layer
                    else float("nan")
                )
                target_mass = values.get("target_probability_mass", 0.0)
                values["target_probability_coverage_ratio"] = (
                    values.get("covered_target_probability_mass", 0.0)
                    / target_mass
                    if target_mass
                    else float("nan")
                )
            for layer, watched_ranges in layer_watched_key_range_stats.items():
                per_layer_stats.setdefault(str(layer), {})[
                    "watched_key_ranges"
                ] = {
                    range_name: _finalize_watched_range_stats(range_values)
                    for range_name, range_values in watched_ranges.items()
                }
            batch_size = int(input_ids.shape[0])
            generated_tokens = int(
                output_ids.shape[1] - input_ids.shape[1]
            )
            row = {
                "call_index": len(recorder.rows),
                "method": recorder.method,
                **recorder.context,
                "batch_size": batch_size,
                "input_tokens": int(input_ids.shape[1]),
                "input_ids_sha256": token_hash,
                "generated_tokens": generated_tokens,
                "prefill_latency_sec": prefill_seconds,
                "decode_latency_sec": max(
                    total_seconds - prefill_seconds, 0.0
                ),
                "total_latency_sec": total_seconds,
                "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
                **dump_metadata,
            }
            if resolved_watched_ranges:
                row["resolved_watched_key_ranges"] = [
                    [int(start), int(end)]
                    for start, end in resolved_watched_ranges
                ]
            if stats_source is not None:
                row.update(
                    {
                        "selected_blocks": int(selected_blocks),
                        "causal_blocks": int(causal_blocks),
                        "block_keep_ratio": (
                            selected_blocks / causal_blocks
                            if causal_blocks
                            else float("nan")
                        ),
                    }
                )
                if "selected_token_pairs" in after:
                    global_keep = (
                        selected_token_pairs / causal_token_pairs
                        if causal_token_pairs
                        else float("nan")
                    )
                    within_block_keep = (
                        compacted_key_tokens / candidate_key_tokens
                        if candidate_key_tokens
                        else float("nan")
                    )
                    row.update(
                        {
                            "selected_token_pairs": int(selected_token_pairs),
                            "causal_token_pairs": int(causal_token_pairs),
                            "global_token_keep_ratio": global_keep,
                            "global_token_sparsity": 1.0 - global_keep,
                            "compacted_key_tokens": int(compacted_key_tokens),
                            "candidate_key_tokens": int(candidate_key_tokens),
                            "within_selected_block_token_keep_ratio": (
                                within_block_keep
                            ),
                            "within_selected_block_token_sparsity": (
                                1.0 - within_block_keep
                            ),
                        }
                    )
                if target_mask_tokens:
                    row.update(
                        {
                            "target_mask_tokens": int(target_mask_tokens),
                            "covered_target_tokens": int(
                                covered_target_tokens
                            ),
                            "target_mask_coverage_ratio": (
                                covered_target_tokens / target_mask_tokens
                            ),
                        }
                    )
                if target_probability_mass:
                    row.update(
                        {
                            "target_probability_mass": float(
                                target_probability_mass
                            ),
                            "covered_target_probability_mass": float(
                                covered_target_probability_mass
                            ),
                            "target_probability_coverage_ratio": (
                                covered_target_probability_mass
                                / target_probability_mass
                            ),
                        }
                    )
                if any(chosen_block_size_rows.values()):
                    row["chosen_block_size_rows"] = chosen_block_size_rows
                if any(candidate_kernel_pairs_by_size.values()):
                    row["candidate_kernel_pairs_by_size"] = (
                        candidate_kernel_pairs_by_size
                    )
                if watched_key_range_stats:
                    row["watched_key_range_stats"] = {
                        range_name: _finalize_watched_range_stats(range_values)
                        for range_name, range_values in watched_key_range_stats.items()
                    }
                if per_layer_stats:
                    row["per_layer_selector_stats"] = per_layer_stats
                if member_mask_fidelity_stats:
                    row["member_mask_fidelity_stats"] = (
                        member_mask_fidelity_stats
                    )
            recorder.rows.append(row)
            with recorder.output_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            return output_ids

        self.model.generate = types.MethodType(wrapped_generate, self.model)
        self._installed = True

    def uninstall(self) -> None:
        if self._installed:
            self.model.generate = self._original_generate
            self._installed = False

    def summary(self) -> Dict[str, Any]:
        rows = self.rows
        global_keep = _ratio_of_sums(
            rows, "selected_token_pairs", "causal_token_pairs"
        )
        within_block_keep = _ratio_of_sums(
            rows, "compacted_key_tokens", "candidate_key_tokens"
        )
        target_coverage = _ratio_of_sums(
            rows, "covered_target_tokens", "target_mask_tokens"
        )
        target_probability_coverage = _ratio_of_sums(
            rows,
            "covered_target_probability_mass",
            "target_probability_mass",
        )
        chosen_block_size_rows: Dict[str, int] = {}
        candidate_kernel_pairs_by_size: Dict[str, int] = {}
        watched_key_range_totals: Dict[str, Dict[str, float]] = {}
        for row in rows:
            for size, count in row.get("chosen_block_size_rows", {}).items():
                chosen_block_size_rows[str(size)] = (
                    chosen_block_size_rows.get(str(size), 0) + int(count)
                )
            for size, count in row.get(
                "candidate_kernel_pairs_by_size", {}
            ).items():
                candidate_kernel_pairs_by_size[str(size)] = (
                    candidate_kernel_pairs_by_size.get(str(size), 0)
                    + int(count)
                )
            for range_name, values in row.get(
                "watched_key_range_stats", {}
            ).items():
                aggregate = watched_key_range_totals.setdefault(
                    str(range_name), {}
                )
                for name in (
                    "probability_mass",
                    "selector_rows",
                    "valid_token_slots",
                    "target_token_slots",
                    "selected_token_slots",
                    "sink_token_slots",
                ):
                    aggregate[name] = aggregate.get(name, 0.0) + float(
                        values.get(name, 0.0)
                    )
        summary = {
            "count": len(rows),
            "avg_input_tokens": _mean(rows, "input_tokens"),
            "avg_generated_tokens": _mean(rows, "generated_tokens"),
            "avg_prefill_latency_sec": _mean(rows, "prefill_latency_sec"),
            "avg_decode_latency_sec": _mean(rows, "decode_latency_sec"),
            "avg_total_latency_sec": _mean(rows, "total_latency_sec"),
            "avg_block_keep_ratio": _optional_mean(
                rows, "block_keep_ratio"
            ),
            "avg_global_token_keep_ratio": _optional_mean(
                rows, "global_token_keep_ratio"
            ),
            "avg_global_token_sparsity": _optional_mean(
                rows, "global_token_sparsity"
            ),
            "avg_within_selected_block_token_keep_ratio": _optional_mean(
                rows, "within_selected_block_token_keep_ratio"
            ),
            "avg_target_mask_coverage_ratio": _optional_mean(
                rows, "target_mask_coverage_ratio"
            ),
            "avg_target_probability_coverage_ratio": _optional_mean(
                rows, "target_probability_coverage_ratio"
            ),
            "chosen_block_size_rows": chosen_block_size_rows,
            "candidate_kernel_pairs_by_size": (
                candidate_kernel_pairs_by_size
            ),
            "watched_key_range_stats": {
                range_name: _finalize_watched_range_stats(values)
                for range_name, values in watched_key_range_totals.items()
            },
            "max_peak_memory_bytes": max(
                (int(row["peak_memory_bytes"]) for row in rows), default=0
            ),
        }
        if global_keep is not None:
            summary.update(
                {
                    "selected_token_pairs": int(
                        sum(int(row["selected_token_pairs"]) for row in rows)
                    ),
                    "causal_token_pairs": int(
                        sum(int(row["causal_token_pairs"]) for row in rows)
                    ),
                    "global_token_keep_ratio": global_keep,
                    "global_token_sparsity": 1.0 - global_keep,
                }
            )
        if within_block_keep is not None:
            summary.update(
                {
                    "compacted_key_tokens": int(
                        sum(int(row["compacted_key_tokens"]) for row in rows)
                    ),
                    "candidate_key_tokens": int(
                        sum(int(row["candidate_key_tokens"]) for row in rows)
                    ),
                    "within_selected_block_token_keep_ratio": (
                        within_block_keep
                    ),
                    "within_selected_block_token_sparsity": (
                        1.0 - within_block_keep
                    ),
                }
            )
        if target_coverage is not None:
            summary.update(
                {
                    "target_mask_tokens": int(
                        sum(int(row["target_mask_tokens"]) for row in rows)
                    ),
                    "covered_target_tokens": int(
                        sum(int(row["covered_target_tokens"]) for row in rows)
                    ),
                    "target_mask_coverage_ratio": target_coverage,
                }
            )
        if target_probability_coverage is not None:
            summary.update(
                {
                    "target_probability_mass": sum(
                        float(row["target_probability_mass"])
                        for row in rows
                    ),
                    "covered_target_probability_mass": sum(
                        float(row["covered_target_probability_mass"])
                        for row in rows
                    ),
                    "target_probability_coverage_ratio": (
                        target_probability_coverage
                    ),
                }
            )
        return summary


def _mean(rows: list[Mapping[str, Any]], key: str) -> float:
    if not rows:
        return float("nan")
    return sum(float(row[key]) for row in rows) / len(rows)


def _optional_mean(rows: list[Mapping[str, Any]], key: str) -> Optional[float]:
    values = [float(row[key]) for row in rows if key in row]
    return sum(values) / len(values) if values else None


def _ratio_of_sums(
    rows: list[Mapping[str, Any]], numerator: str, denominator: str
) -> Optional[float]:
    selected = sum(float(row[numerator]) for row in rows if numerator in row)
    possible = sum(float(row[denominator]) for row in rows if denominator in row)
    return selected / possible if possible else None


def write_benchmark_summary(
    path: str | Path,
    *,
    benchmark: str,
    method: str,
    recorder: GenerationMetricsRecorder,
    run_args: Mapping[str, Any],
    method_metadata: Mapping[str, Any],
    extra: Optional[Mapping[str, Any]] = None,
) -> None:
    summary = {
        "benchmark": benchmark,
        "method": method,
        "method_metadata": dict(method_metadata),
        "hardware": {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_total_memory_bytes": torch.cuda.get_device_properties(
                0
            ).total_memory,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
        "timing_scope": "synchronized model.generate; prefill via CUDA events",
        "runtime": recorder.summary(),
        "run_args": dict(run_args),
    }
    if extra:
        summary.update(dict(extra))
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
