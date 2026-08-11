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
from typing import Any, Dict, Mapping, Optional, Sequence

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
    if int(config.get("num_groups_per_layer", -1)) != 3:
        raise ValueError("The requested benchmark configuration requires K=3")
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
    final_token_budget: int = 8192,
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
    target_token_budget: int = 8192,
    minimum_block_coverage_ratio: float = 0.125,
    probe_weights: tuple[float, float, float, float] = (0.2, 0.3, 0.4, 0.1),
    token_chunk_size: int = 32,
    force_sink_block: bool = True,
    force_diagonal_block: bool = True,
    adaptive_coverage_tolerance: float = 0.0,
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
        selection_mode="token_first_block_cover",
        final_token_budget=target_token_budget,
        probe_weights=probe_weights,
        logical_key_block_size=logical_key_block_size,
        minimum_block_coverage_ratio=minimum_block_coverage_ratio,
        adaptive_coverage_tolerance=adaptive_coverage_tolerance,
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
):
    """Install a validated method without changing benchmark inputs.

    The method-specific sparse algorithm and parameters match the existing
    aligned LongBench comparison. The returned model may differ from the input
    object because the official MInference API returns its patched model.
    """

    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
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
        patched_model = MInference(
            "minference", "Qwen/Qwen2.5-7B-Instruct"
        )(model)
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

        config = {
            "block_size": 128,
            "flex_prefill_gamma": 0.9,
            "flex_prefill_tau": 0.1,
            "flex_prefill_min_budget": 512,
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
        if method.startswith("shareprefill_ae3_token_block"):
            block_suffix = method.rsplit("block", 1)[1].lstrip("_")
            adaptive = block_suffix == "auto"
            logical_key_block_sizes = (
                (32, 64, 128) if adaptive else (int(block_suffix),)
            )
            patch, config = install_shareprefill_ae3_token_block_cover(
                model,
                logical_key_block_size=logical_key_block_sizes,
                group_config_path=group_config_path,
                flexprefill_root=flexprefill_root,
                adaptive_coverage_tolerance=0.02 if adaptive else 0.0,
            )
            return model, patch, {
                "implementation": "shareprefill_ae_k3_token_first_block_cover",
                "model": model_name,
                "group_config_path": str(group_config_path),
                "classification_metric": config.get("classification_metric"),
                "online_config": {
                    "execution_query_tile_size": 128,
                    "logical_key_block_sizes": list(logical_key_block_sizes),
                    "block_size_selection": (
                        "largest_within_2pp_of_best_coverage"
                        if adaptive
                        else "fixed"
                    ),
                    "adaptive_coverage_tolerance": 0.02 if adaptive else 0.0,
                    "target_token_budget": 8192,
                    "final_whole_block_token_budget": 8192,
                    "maximum_selected_blocks": {
                        str(size): 8192 // size
                        for size in logical_key_block_sizes
                    },
                    "minimum_block_coverage_ratio": 0.125,
                    "minimum_target_tokens_per_block": {
                        str(size): math.ceil(size * 0.125)
                        for size in logical_key_block_sizes
                    },
                    "query_probes": ["one_third", "two_thirds", "last", "mean"],
                    "probe_aggregation": "weighted_average",
                    "probe_weights": {
                        "one_third": 0.2,
                        "two_thirds": 0.3,
                        "last": 0.4,
                        "mean": 0.1,
                    },
                    "force_sink_block": True,
                    "force_diagonal_block": True,
                    "protected_blocks_count_within_budget": True,
                    "dense_layers": [],
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
            before_block_sizes = before.get("chosen_block_size_rows", {})
            after_block_sizes = after.get("chosen_block_size_rows", {})
            chosen_block_size_rows = {
                str(size): int(after_block_sizes.get(size, 0))
                - int(before_block_sizes.get(size, 0))
                for size in set(before_block_sizes) | set(after_block_sizes)
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
                if any(chosen_block_size_rows.values()):
                    row["chosen_block_size_rows"] = chosen_block_size_rows
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
        chosen_block_size_rows: Dict[str, int] = {}
        for row in rows:
            for size, count in row.get("chosen_block_size_rows", {}).items():
                chosen_block_size_rows[str(size)] = (
                    chosen_block_size_rows.get(str(size), 0) + int(count)
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
            "chosen_block_size_rows": chosen_block_size_rows,
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
