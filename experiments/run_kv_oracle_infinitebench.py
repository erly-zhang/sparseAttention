#!/usr/bin/env python3
"""Run sparse methods on the official InfiniteBench adapter."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import re
import sys
from pathlib import Path

import lm_eval
import torch
import torch.nn.functional as F
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.benchmark_shareprefill_ae3 import (
    DEFAULT_FLEXPREFILL_ROOT,
    DEFAULT_GROUP_CONFIG,
    GenerationMetricsRecorder,
    METHODS,
    install_benchmark_method,
    resolve_group_config,
    validate_group_config_scope,
    validate_input_alignment,
    write_benchmark_summary,
)
from experiments.dense_full_query_answer_profiler import (
    DenseFullQueryAnswerProfiler,
)


TASKS = [
    "longbook_sum_eng",
    "longbook_qa_eng",
    "longbook_choice_eng",
    "longdialogue_qa_eng",
    "longbook_qa_chn",
    "code_debug",
    "math_find",
    "passkey",
    "number_string",
    "kv_retrieval",
]
SHAREPREFILL_METHODS = {
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
    "shareprefill_per_head_token_block_auto",
}

# Use one generation budget for every method. The budgets are shared across
# methods and leave enough room for each task's answer parser to observe a
# complete answer.
INFINITEBENCH_MAX_NEW_TOKENS = {
    "longbook_sum_eng": 1200,
    "longbook_qa_eng": 40,
    "longbook_choice_eng": 40,
    "longdialogue_qa_eng": 40,
    "longbook_qa_chn": 40,
    "code_debug": 128,
    "math_find": 10,
    "passkey": 15,
    "number_string": 128,
    "kv_retrieval": 128,
}

ANSWER_DIGIT_LENGTH = {
    "passkey": 5,
    "number_string": 10,
}

UUID_PATTERN = (
    r"(?<![0-9a-f])"
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}"
    r"(?![0-9a-f])"
)
UUID_EXPRESSION = re.compile(UUID_PATTERN, re.IGNORECASE)


def _find_subsequence_ranges(
    values: list[int], pattern: list[int]
) -> list[tuple[int, int]]:
    if not pattern:
        return []
    width = len(pattern)
    return [
        (start, start + width)
        for start in range(0, len(values) - width + 1)
        if values[start : start + width] == pattern
    ]


def build_answer_range_resolver(tokenizer, task: str):
    """Locate the repeated synthetic retrieval answer in exact model input IDs."""

    try:
        digit_length = ANSWER_DIGIT_LENGTH[task]
    except KeyError as error:
        raise ValueError(
            "Automatic answer-span profiling only supports PassKey and "
            "Number String"
        ) from error
    expression = re.compile(rf"(?<!\d)\d{{{digit_length}}}(?!\d)")

    def resolve(input_ids: torch.Tensor) -> tuple[tuple[int, int], ...]:
        values = [int(value) for value in input_ids[0].detach().cpu().tolist()]
        prefix_text = tokenizer.decode(
            values[: min(512, len(values))], skip_special_tokens=False
        )
        candidates = expression.findall(prefix_text)
        counts = Counter(candidates)
        repeated = [value for value in candidates if counts[value] >= 2]
        if not repeated:
            raise RuntimeError(
                f"Could not identify the repeated {digit_length}-digit answer "
                f"for {task}"
            )
        answer = repeated[0]
        ranges: set[tuple[int, int]] = set()
        for prefix in ("", " ", "\n"):
            encoded = tokenizer.encode(
                prefix + answer, add_special_tokens=False
            )
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
            for start, end in _find_subsequence_ranges(values, encoded):
                answer_start = start + len(prefix_ids)
                if answer_start < end:
                    ranges.add((answer_start, end))
        if len(ranges) < 2:
            direct = tokenizer.encode(answer, add_special_tokens=False)
            ranges.update(_find_subsequence_ranges(values, direct))
        ordered = tuple(sorted(ranges))
        if len(ordered) < 2:
            raise RuntimeError(
                f"Expected at least two tokenized occurrences of {answer!r}, "
                f"found {ordered}"
            )
        return ordered

    return resolve


def build_kv_retrieval_range_resolver(tokenizer, *, value_only: bool = False):
    """Locate the queried JSON key/value pair in the final tokenized prompt."""

    def resolve(input_ids: torch.Tensor) -> tuple[tuple[int, int], ...]:
        values = [int(value) for value in input_ids[0].detach().cpu().tolist()]
        text = tokenizer.decode(values, skip_special_tokens=False)
        uuids = UUID_EXPRESSION.findall(text)
        counts = Counter(value.lower() for value in uuids)
        repeated = [value for value, count in counts.items() if count >= 2]
        if len(repeated) != 1:
            raise RuntimeError(
                "Expected exactly one repeated UUID identifying the queried "
                f"KV key, found {len(repeated)} candidates"
            )
        queried_key = repeated[0]
        pair_expression = re.compile(
            rf'(?i)"{re.escape(queried_key)}"\s*:\s*"'
            rf'({UUID_PATTERN})"'
        )
        pair_matches = list(pair_expression.finditer(text))
        if len(pair_matches) != 1:
            raise RuntimeError(
                "Expected one context key/value pair for queried UUID "
                f"{queried_key!r}, found {len(pair_matches)}"
            )
        pair = pair_matches[0]

        answer_value = pair.group(1)
        key_ranges = _find_subsequence_ranges(
            values,
            tokenizer.encode(queried_key, add_special_tokens=False),
        )
        value_ranges = _find_subsequence_ranges(
            values,
            tokenizer.encode(answer_value, add_special_tokens=False),
        )
        if len(key_ranges) < 2 or not value_ranges:
            raise RuntimeError(
                "Could not map the queried key and answer UUIDs back to exact "
                f"input-token indices: keys={key_ranges}, values={value_ranges}"
            )
        context_key = min(key_ranges)
        following_values = [
            item
            for item in value_ranges
            if item[0] >= context_key[1] and item[0] - context_key[1] <= 16
        ]
        if len(following_values) != 1:
            raise RuntimeError(
                "Could not uniquely pair the context key with its adjacent "
                f"value: key={context_key}, values={value_ranges}"
            )
        context_value = following_values[0]
        if value_only:
            return (context_value,)
        return ((context_key[0], context_value[1]),)

    return resolve


def effective_context_length(config) -> int:
    """Return the validated context after an optional static YaRN extension."""

    base = int(config.max_position_embeddings)
    rope_scaling = getattr(config, "rope_scaling", None) or {}
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
    if rope_type != "yarn":
        return base
    original = int(
        rope_scaling.get("original_max_position_embeddings", base)
    )
    factor = float(rope_scaling.get("factor", 1.0))
    return int(original * factor)


def install_generation_last_logit_hook(model) -> None:
    """Avoid materializing prompt-length vocabulary logits during generation."""

    def keep_last_hidden_state(_module, inputs):
        hidden_states = inputs[0]
        return (hidden_states[:, -1:, :], *inputs[1:])

    model._dense_last_logit_hook = model.lm_head.register_forward_pre_hook(
        keep_last_hidden_state
    )


def parse_half_open_range(value: str) -> tuple[int, int]:
    try:
        start_text, end_text = value.split(":", 1)
        start, end = int(start_text), int(end_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Expected a half-open token range such as 48:53"
        ) from exc
    if start < 0 or end <= start:
        raise argparse.ArgumentTypeError(
            "Token ranges must satisfy 0 <= start < end"
        )
    return start, end


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="/home/ubuntu/work/model/Qwen2.5-7B"
    )
    parser.add_argument(
        "--method", choices=METHODS, default="shareprefill_ae3_full"
    )
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument(
        "--max_length",
        type=int,
        default=131072,
        help="Total model context, including prompt and generated tokens.",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chat", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--group_config")
    parser.add_argument(
        "--dense_layers",
        nargs="*",
        type=int,
        default=[],
        help="Zero-based layers that retain dense FlashAttention-2.",
    )
    parser.add_argument(
        "--calibration_scope",
        choices=["benchmark_specific", "longbench_fixed"],
        default="benchmark_specific",
    )
    parser.add_argument(
        "--flexprefill_root", default=str(DEFAULT_FLEXPREFILL_ROOT)
    )
    parser.add_argument(
        "--task_config_dir",
        help="Optional InfiniteBench YAML directory with filtered data_dir.",
    )
    parser.add_argument(
        "--output_dir",
        default=(
            "/home/ubuntu/work/experiments/outputs/"
            "infinitebench_benchmark_specific_comparison"
        ),
    )
    parser.add_argument("--reference_metrics")
    parser.add_argument(
        "--create_reference",
        action="store_true",
        help=(
            "Allow this run to create the model-specific input manifest. "
            "Use only for the first AutoBlock run of each model/task."
        ),
    )
    parser.add_argument(
        "--block_f_beta",
        type=float,
        default=2.0,
        help="F-beta used by the AutoBlock F-beta block selector.",
    )
    parser.add_argument(
        "--fixed_topk_budget",
        type=int,
        default=8192,
        help=(
            "Fixed TopK target-token budget and matching final whole-block "
            "kernel-key budget for fixed-TopK AutoBlock methods."
        ),
    )
    parser.add_argument(
        "--oracle_residual_tokens",
        type=int,
        default=0,
        help="Exact member-head residual tokens per non-representative row.",
    )
    parser.add_argument(
        "--oracle_member_topk_budget",
        type=int,
        default=8192,
        help="Member-head dense TopK reference budget used for recall metrics.",
    )
    parser.add_argument(
        "--oracle_residual_dump_dir",
        help="Write sampled shared, residual, member-TopK, and final masks.",
    )
    parser.add_argument(
        "--target_token_top_p",
        type=float,
        help=(
            "Override the token target top-p for probability AutoBlock methods. "
            "This changes only the probability prefix; block projection remains "
            "unchanged."
        ),
    )
    parser.add_argument(
        "--target_top_p_start_layer",
        type=int,
        help=(
            "Zero-based first Top-p layer. The configurable fixed/Top-p hybrid "
            "uses fixed top-8192 before it; the dense-prefix method uses full "
            "FlashAttention before it."
        ),
    )
    parser.add_argument(
        "--watched_key_range",
        dest="watched_key_ranges",
        action="append",
        type=parse_half_open_range,
        default=[],
        help=(
            "Half-open input-token interval whose selector probability mass and "
            "target/final-mask retention should be profiled. Repeat for multiple "
            "answer occurrences, for example 48:53 and 58:63."
        ),
    )
    parser.add_argument(
        "--record_sparsity",
        action="store_true",
        help=(
            "Record exact final-kernel causal token-pair sparsity without "
            "writing per-selector indices."
        ),
    )
    parser.add_argument(
        "--profile_one_token",
        action="store_true",
        help=(
            "Generate one token while preserving the task's standard input "
            "truncation budget and input hash."
        ),
    )
    parser.add_argument(
        "--dump_selector_details",
        action="store_true",
        help=(
            "Record reconstructible per-sample, per-layer, per-head selector "
            "indices and exact final-kernel token-pair sparsity."
        ),
    )
    parser.add_argument(
        "--profile_member_mask_fidelity",
        action="store_true",
        help=(
            "Recompute the fixed TopK selector for every member Q head and "
            "record overlap with the representative target and final kernel mask."
        ),
    )
    parser.add_argument(
        "--profile_kv_retrieval_ranges",
        action="store_true",
        help=(
            "Resolve and profile the queried context key/value token range for "
            "InfiniteBench KV Retrieval."
        ),
    )
    parser.add_argument(
        "--force_watched_key_range_blocks",
        action="store_true",
        help=(
            "Budget-preserving oracle: force blocks overlapping resolved watched "
            "ranges, while retaining the existing final whole-block budget."
        ),
    )
    parser.add_argument(
        "--ranked_probability_dump_dir",
        help=(
            "Write one full sorted key-token probability distribution per "
            "layer for the first formal sample."
        ),
    )
    parser.add_argument(
        "--dense_answer_score_dump_dir",
        help=(
            "Profile answer-value scores from full dense Llama attention. "
            "Requires shareprefill_ae3_full with every layer in --dense_layers."
        ),
    )
    args = parser.parse_args()
    args.group_config = str(
        resolve_group_config(
            "infinitebench", args.calibration_scope, args.group_config
        )
    )
    if args.task_config_dir is None and args.calibration_scope == "benchmark_specific":
        args.task_config_dir = (
            "/home/ubuntu/work/experiments/data/"
            "infinitebench_benchmark_specific_calibration/task_configs"
        )
    if args.batch_size != 1:
        raise ValueError("Formal sparse benchmark requires batch_size=1")
    if args.fixed_topk_budget <= 0:
        raise ValueError("--fixed_topk_budget must be positive")
    if args.dense_layers and args.method != "shareprefill_ae3_full":
        raise ValueError(
            "--dense_layers is only valid for SharePrefill-AE3 Full"
        )
    if args.dense_answer_score_dump_dir and not args.profile_kv_retrieval_ranges:
        raise ValueError(
            "--dense_answer_score_dump_dir requires "
            "--profile_kv_retrieval_ranges"
        )
    configurable_layer_switch_methods = {
        "shareprefill_ae3_token_block_auto_hybrid",
        "shareprefill_ae3_token_block_auto_dense_topp_mass",
    }
    if args.method in configurable_layer_switch_methods:
        if args.target_top_p_start_layer is None:
            raise ValueError(
                "The configurable hybrid method requires "
                "--target_top_p_start_layer"
            )
    elif args.target_top_p_start_layer is not None:
        raise ValueError(
            "--target_top_p_start_layer is only valid for a configurable "
            "layer-switch method"
        )
    if (
        args.profile_member_mask_fidelity
        and args.method != "shareprefill_ae3_token_block_auto"
    ):
        raise ValueError(
            "Member-mask fidelity profiling requires fixed-TopK AutoBlock"
        )
    if args.profile_kv_retrieval_ranges and args.task != "kv_retrieval":
        raise ValueError(
            "--profile_kv_retrieval_ranges is only valid for KV Retrieval"
        )
    if (
        args.force_watched_key_range_blocks
        and not args.profile_kv_retrieval_ranges
    ):
        raise ValueError(
            "--force_watched_key_range_blocks requires automatic KV range "
            "profiling"
        )
    if args.force_watched_key_range_blocks and args.method not in {
        "shareprefill_ae3_token_block_auto",
        "shareprefill_ae3_token_block_auto_fixed_mass_profile",
    }:
        raise ValueError(
            "The KV oracle requires the fixed-TopK whole-block AutoBlock path"
        )
    return args


def main() -> None:
    args = parse_args()
    ranked_probability_dump_dir = (
        Path(args.ranked_probability_dump_dir).resolve()
        if args.ranked_probability_dump_dir
        else None
    )
    oracle_residual_dump_dir = (
        Path(args.oracle_residual_dump_dir).resolve()
        if args.oracle_residual_dump_dir
        else None
    )
    ranked_input_saved = False
    standard_max_new_tokens = INFINITEBENCH_MAX_NEW_TOKENS[args.task]
    max_new_tokens = 1 if args.profile_one_token else standard_max_new_tokens
    max_input_length = args.max_length - standard_max_new_tokens
    if max_input_length <= 0:
        raise ValueError(
            f"Model context {args.max_length} must exceed the generation "
            f"budget {max_new_tokens} for {args.task}"
        )
    args.max_new_tokens = max_new_tokens
    args.standard_max_new_tokens = standard_max_new_tokens
    args.max_input_length = max_input_length
    if args.method in SHAREPREFILL_METHODS:
        validate_group_config_scope(
            args.group_config,
            benchmark="infinitebench",
            calibration_scope=args.calibration_scope,
        )
    benchmark_root = Path(args.flexprefill_root) / "experiments/benchmark"
    sys.path.insert(0, str(benchmark_root))
    from utils import (  # type: ignore
        convert_to_json_compatible,
        fixed_generate_until,
        seed_everything,
        tok_encode_middle_trunc,
    )

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    seed_everything(args.seed)

    HFLM.tok_encode = tok_encode_middle_trunc

    def budgeted_tok_batch_encode(
        model,
        strings,
        padding_side="left",
        left_truncate_len=None,
        truncation=False,
    ):
        """Middle-truncate prompts while reserving the decode budget."""

        nonlocal ranked_input_saved

        original_padding_side = model.tokenizer.padding_side
        model.tokenizer.padding_side = padding_side
        try:
            encoded = model.tokenizer(
                strings,
                add_special_tokens=bool(model.add_bos_token),
            )
            max_length = min(model.max_length, args.max_input_length)
            if left_truncate_len is not None:
                max_length = min(max_length, left_truncate_len)
            pad_length = min(
                max(len(ids) for ids in encoded["input_ids"]),
                max_length,
            )

            def truncate_and_pad(values, pad_value):
                if len(values) > max_length:
                    left = max_length // 2
                    right = max_length - left
                    values = values[:left] + values[-right:]
                tensor = torch.tensor(values, dtype=torch.long)
                if tensor.numel() < pad_length:
                    tensor = F.pad(
                        tensor,
                        (pad_length - tensor.numel(), 0),
                        value=pad_value,
                    )
                return tensor

            input_ids = torch.stack(
                [
                    truncate_and_pad(ids, model.tokenizer.pad_token_id)
                    for ids in encoded["input_ids"]
                ]
            )
            attention_mask = torch.stack(
                [
                    truncate_and_pad(mask, 0)
                    for mask in encoded["attention_mask"]
                ]
            )
            if (
                ranked_probability_dump_dir is not None
                and not ranked_input_saved
                and input_ids.shape[0] == 1
            ):
                ranked_probability_dump_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "input_ids": input_ids[0].to(torch.int32),
                        "attention_mask": attention_mask[0].to(torch.bool),
                        "task": args.task,
                        "watched_key_ranges": list(args.watched_key_ranges),
                    },
                    ranked_probability_dump_dir / "input_tokens.pt",
                )
                ranked_input_saved = True
            return input_ids, attention_mask
        finally:
            model.tokenizer.padding_side = original_padding_side

    HFLM.tok_batch_encode = budgeted_tok_batch_encode
    HFLM.generate_until = fixed_generate_until

    output_dir = (
        Path(args.output_dir) / args.method / args.task
    ).resolve()
    selector_dump_path = (
        output_dir / "selector_dump.jsonl"
        if args.dump_selector_details
        else None
    )
    reference_metrics = (
        Path(args.reference_metrics).resolve()
        if args.reference_metrics
        else (
            Path(args.output_dir)
            / "shareprefill_ae3_full"
            / args.task
            / "online_metrics.jsonl"
        ).resolve()
    )
    if args.create_reference and args.reference_metrics:
        raise ValueError(
            "--create_reference and --reference_metrics are mutually exclusive"
        )
    if (
        args.method != "shareprefill_ae3_full"
        and not args.create_reference
        and not reference_metrics.is_file()
    ):
        raise FileNotFoundError(
            "Baseline runs require the SharePrefill input manifest: "
            f"{reference_metrics}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    model_kwargs = {
        "pretrained": args.model,
        "backend": "causal",
        "max_length": args.max_length,
        "dtype": torch.bfloat16,
        "batch_size": 1,
        "trust_remote_code": True,
    }
    model_kwargs["_attn_implementation"] = "flash_attention_2"
    model = HFLM(
        **model_kwargs,
    )
    if args.method == "dense":
        install_generation_last_logit_hook(model._model)
    configured_context = effective_context_length(model._model.config)
    if args.max_length > configured_context:
        raise ValueError(
            f"Requested context {args.max_length} exceeds model limit "
            f"{configured_context}"
        )
    patched_model, patch, method_metadata = install_benchmark_method(
        model._model,
        args.method,
        model_name=args.model,
        group_config_path=args.group_config,
        flexprefill_root=args.flexprefill_root,
        dense_layers=args.dense_layers,
        selector_dump_path=selector_dump_path,
        record_sparsity=args.record_sparsity or args.dump_selector_details,
        block_f_beta=args.block_f_beta,
        fixed_topk_budget=args.fixed_topk_budget,
        target_token_top_p=args.target_token_top_p,
        target_top_p_start_layer=args.target_top_p_start_layer,
        watched_key_ranges=tuple(args.watched_key_ranges),
        force_watched_key_range_blocks=args.force_watched_key_range_blocks,
        profile_member_mask_fidelity=args.profile_member_mask_fidelity,
        oracle_residual_tokens=args.oracle_residual_tokens,
        oracle_member_topk_budget=args.oracle_member_topk_budget,
    )
    method_metadata["dense_generation_last_logit_only"] = (
        args.method == "dense"
    )
    model._model = patched_model
    dense_answer_profiler = None
    if args.dense_answer_score_dump_dir:
        num_layers = int(model._model.config.num_hidden_layers)
        if args.method != "shareprefill_ae3_full" or set(args.dense_layers) != set(
            range(num_layers)
        ):
            raise ValueError(
                "Dense answer profiling requires shareprefill_ae3_full and all "
                f"layers 0..{num_layers - 1} in --dense_layers"
            )
        if patch is None or not hasattr(patch, "selector"):
            raise RuntimeError("Dense answer profiling requires a selector carrier")
        install_generation_last_logit_hook(model._model)
        method_metadata["dense_generation_last_logit_only"] = True
        dense_answer_profiler = DenseFullQueryAnswerProfiler(
            model._model,
            args.group_config,
            patch.selector,
            args.dense_answer_score_dump_dir,
            topk_budget=args.fixed_topk_budget,
        )
        dense_answer_profiler.install()
        method_metadata["dense_answer_score_profile"] = {
            "enabled": True,
            "scope": "AE3 representatives under full dense attention",
            "query_score_mode": "full_query_causal_logit_mean",
            "answer_range": "queried_context_value_only",
        }

    # Compile the model and Triton path outside the measured benchmark calls.
    warmup = model.tokenizer(
        "Warmup " * 2048, return_tensors="pt", truncation=True, max_length=4096
    ).to(model._model.device)
    model._model.generate(
        **warmup,
        max_new_tokens=1,
        do_sample=False,
        pad_token_id=model.tokenizer.eos_token_id,
    )
    if patch is not None and hasattr(patch, "selector"):
        patch.selector.stats = type(patch.selector.stats)()
        if oracle_residual_dump_dir is not None:
            if not hasattr(patch.selector, "configure_oracle_residual_dump"):
                raise RuntimeError(
                    "The selected method does not support oracle residual dumps"
                )
            patch.selector.configure_oracle_residual_dump(
                oracle_residual_dump_dir, sample_limit=1
            )
        if ranked_probability_dump_dir is not None:
            if not hasattr(
                patch.selector, "configure_ranked_probability_dump"
            ):
                raise RuntimeError(
                    "The selected method does not support ranked probability dumps"
                )
            patch.selector.configure_ranked_probability_dump(
                ranked_probability_dump_dir
            )

    metrics_path = output_dir / "online_metrics.jsonl"
    if args.profile_kv_retrieval_ranges:
        watched_range_resolver = build_kv_retrieval_range_resolver(
            model.tokenizer,
            value_only=dense_answer_profiler is not None,
        )
    elif (
        args.profile_member_mask_fidelity
        and args.task in ANSWER_DIGIT_LENGTH
    ):
        watched_range_resolver = build_answer_range_resolver(
            model.tokenizer, args.task
        )
    else:
        watched_range_resolver = None
    recorder = GenerationMetricsRecorder(
        model._model,
        patch,
        metrics_path,
        method=args.method,
        watched_range_resolver=watched_range_resolver,
    )
    recorder.reset()
    recorder.set_context(benchmark="infinitebench", task=args.task)
    recorder.install()

    # Official task YAML files resolve dataset paths from the FlexPrefill root.
    os.chdir(args.flexprefill_root)
    task_manager = TaskManager(
        include_path=(
            args.task_config_dir
            or str(benchmark_root / "infinitebench")
        ),
        include_defaults=False,
    )
    results = lm_eval.simple_evaluate(
        model=model,
        tasks=[args.task],
        task_manager=task_manager,
        apply_chat_template=args.chat,
        gen_kwargs=(
            f"do_sample=False,max_gen_toks={max_new_tokens},"
            f"max_new_tokens={max_new_tokens}"
        ),
        log_samples=args.limit > 0 or args.task == "math_find",
        limit=args.limit if args.limit > 0 else None,
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
        fewshot_random_seed=args.seed,
    )
    recorder.uninstall()
    if dense_answer_profiler is not None:
        dense_answer_profiler.uninstall()
        dense_answer_profiler.finalize(recorder.rows)
    if args.create_reference:
        alignment = {"status": "reference_created", "count": len(recorder.rows)}
    elif args.method != "shareprefill_ae3_full" or args.reference_metrics:
        alignment = validate_input_alignment(reference_metrics, recorder.rows)
    else:
        alignment = {"status": "reference_created", "count": len(recorder.rows)}
    compatible = convert_to_json_compatible(results)
    if args.task == "math_find":
        samples = compatible.get("samples", {}).get(args.task, [])
        if len(samples) != len(recorder.rows):
            raise RuntimeError(
                "Math.Find prediction count does not match online metrics: "
                f"{len(samples)} != {len(recorder.rows)}"
            )
        predictions_path = output_dir / "math_find_predictions.jsonl"
        with predictions_path.open("w", encoding="utf-8") as stream:
            for sample, metrics in zip(samples, recorder.rows):
                filtered = sample.get("filtered_resps", [])
                prediction = filtered[0] if filtered else ""
                raw_responses = sample.get("resps", [])
                raw_prediction = (
                    raw_responses[0][0]
                    if raw_responses and raw_responses[0]
                    else prediction
                )
                row = {
                    "doc_id": sample.get("doc_id"),
                    "target": sample.get("target"),
                    "raw_prediction": raw_prediction,
                    "filtered_prediction": prediction,
                    "score": sample.get("get_score_one_math_find"),
                    "doc_hash": sample.get("doc_hash"),
                    "prompt_hash": sample.get("prompt_hash"),
                    "target_hash": sample.get("target_hash"),
                    "input_tokens": metrics.get("input_tokens"),
                    "input_ids_sha256": metrics.get("input_ids_sha256"),
                    "generated_tokens": metrics.get("generated_tokens"),
                    "prefill_latency_sec": metrics.get("prefill_latency_sec"),
                    "decode_latency_sec": metrics.get("decode_latency_sec"),
                    "total_latency_sec": metrics.get("total_latency_sec"),
                }
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        # Keep parser evidence without duplicating each long numeric context.
        compatible.pop("samples", None)
    (output_dir / "lm_eval_results.json").write_text(
        json.dumps(compatible, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_benchmark_summary(
        output_dir / "summary.json",
        benchmark="InfiniteBench",
        method=args.method,
        recorder=recorder,
        run_args=vars(args),
        method_metadata=method_metadata,
        extra={
            "official_lm_eval_results": compatible.get("results", {}),
            "input_alignment": alignment,
        },
    )


if __name__ == "__main__":
    main()
