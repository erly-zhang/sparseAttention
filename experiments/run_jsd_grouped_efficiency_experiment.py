#!/usr/bin/env python3
"""JSD-grouped representative-head sparse prefill on canonical LongBench-v2.

Run ``classify`` first with eager attention to create a fixed, offline grouping.
Run ``eval`` with the official FlexPrefill environment. The eval path uses the
project-owned representative-head block-top-p selector with FlexPrefill's
Triton sparse kernel, while standard FlashAttention handles decode and dense
fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.jsd_grouped_sparse import (  # noqa: E402
    JSDDistanceAccumulator,
    build_group_config,
    install_grouped_flexprefill,
)
from experiments.run_shared_layer_mask_experiment import (  # noqa: E402
    aggregate_global_representative_heads,
    build_prompt,
    collect_last_q_attentions,
    extract_mcq_answer,
    load_longbench_v2_samples,
    select_representative_head,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Token-JSD grouped representative-head sparse prefill"
    )
    parser.add_argument("stage", choices=["classify", "eval", "all"])
    parser.add_argument(
        "--model_name_or_path",
        default="/home/ubuntu/work/model/Qwen2.5-7B",
    )
    parser.add_argument(
        "--data_path",
        default=(
            "/home/ubuntu/work/experiments/data/"
            "longbench_v2_32k_full_7b.jsonl"
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=(
            "/home/ubuntu/work/experiments/outputs/"
            "stage2_jsd_single_cluster_block_top_p_0.85_500eval"
        ),
    )
    parser.add_argument(
        "--flexprefill_root", default="/home/ubuntu/work/FlexPrefill"
    )
    parser.add_argument("--head_selection_num_samples", type=int, default=3)
    parser.add_argument("--eval_num_samples", type=int, default=500)
    parser.add_argument(
        "--representative_metric",
        choices=["jsd", "coverage", "shareprefill_ae"],
        default="jsd",
        help="Offline criterion used to choose one representative per layer.",
    )
    parser.add_argument("--num_groups_per_layer", type=int, default=1)
    parser.add_argument(
        "--classification_group_counts",
        type=int,
        nargs="*",
        default=None,
        help=(
            "For JSD classification, export additional fixed-K configs from "
            "the same accumulated distance matrices."
        ),
    )
    parser.add_argument("--classification_last_q", type=int, default=32)
    parser.add_argument("--online_last_q", type=int, default=128)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--top_p", type=float, default=0.85)
    parser.add_argument(
        "--dense_layers",
        type=int,
        nargs="*",
        default=[],
        help=(
            "Zero-based layer indices that retain the model's original "
            "dense FlashAttention-2 prefill path. Other layers use grouped "
            "block-sparse prefill."
        ),
    )
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--min_budget", type=int, default=1024)
    parser.add_argument("--max_budget", type=int, default=None)
    parser.add_argument(
        "--force_sink_block",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--force_diagonal_block",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--token_top_p",
        type=float,
        default=None,
        help=(
            "Enable arbitrary key-token compaction inside ordinary selected "
            "blocks. Sink/diagonal behavior is controlled separately."
        ),
    )
    parser.add_argument(
        "--min_tokens_per_selected_block",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--token_chunk_size",
        type=int,
        choices=[16, 32],
        default=32,
        help="Hardware chunk width for compacted arbitrary token IDs.",
    )
    parser.add_argument("--max_input_length", type=int, default=32768)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--warmup_num_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval_start", type=int, default=0)
    parser.add_argument(
        "--baseline_results",
        nargs="*",
        default=[
            (
                "/home/ubuntu/work/experiments/outputs/"
                "official_sparse_baselines_500eval/minference/results.json"
            ),
            (
                "/home/ubuntu/work/experiments/outputs/"
                "official_sparse_baselines_500eval/flexprefill/results.json"
            ),
        ],
        help="Official result files whose sample ID order must match.",
    )
    parser.add_argument(
        "--require_baseline_fingerprints",
        action="store_true",
        help="Require prompt and input-ID SHA-256 matches for baseline rows.",
    )
    parser.add_argument("--log_every", type=int, default=5)
    args = parser.parse_args()

    # Compatibility fields consumed by the existing canonical loader/collector.
    # The canonical file starts with three reserved head-selection rows. Read
    # past all three even in tiny smoke runs so role-based task_eval selection
    # cannot accidentally fall back to another calibration row.
    args.num_samples = max(
        args.head_selection_num_samples + args.eval_num_samples,
        3 + args.eval_num_samples,
    )
    args.samples_per_domain = None
    args.domains = None
    args.sub_domain = None
    args.difficulty = None
    args.length = None
    args.start_line = 0
    args.last_q = args.classification_last_q
    args.chunk_size = 2048
    if args.online_last_q != args.block_size:
        raise ValueError(
            "Online query blocks use the configured block size; require "
            "online_last_q == block_size"
        )
    if args.token_top_p is not None and args.dense_layers:
        raise ValueError(
            "--dense_layers is currently supported only by Full selected "
            "blocks (omit --token_top_p)"
        )
    return args


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _input_ids_sha256(input_ids: torch.Tensor) -> str:
    values = ",".join(str(int(token)) for token in input_ids.view(-1).tolist())
    return _sha256_text(values)


def split_canonical_samples(
    samples: Sequence[Mapping[str, Any]],
    *,
    head_selection_num_samples: int,
    eval_num_samples: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    head_samples = [
        dict(sample)
        for sample in samples
        if sample.get("_sample_role") == "head_selection"
    ][:head_selection_num_samples]
    eval_samples = [
        dict(sample)
        for sample in samples
        if sample.get("_sample_role") == "task_eval"
    ][:eval_num_samples]

    if len(head_samples) < head_selection_num_samples:
        head_samples = [
            dict(sample) for sample in samples[:head_selection_num_samples]
        ]
    if len(eval_samples) < eval_num_samples:
        start = head_selection_num_samples
        eval_samples = [
            dict(sample) for sample in samples[start : start + eval_num_samples]
        ]
    if len(head_samples) != head_selection_num_samples:
        raise ValueError("Canonical data does not contain enough head samples")
    if len(eval_samples) != eval_num_samples:
        raise ValueError("Canonical data does not contain enough eval samples")

    head_ids = {str(sample.get("_id")) for sample in head_samples}
    eval_ids = {str(sample.get("_id")) for sample in eval_samples}
    overlap = head_ids & eval_ids
    if overlap:
        raise ValueError(f"Head-selection/eval samples overlap: {sorted(overlap)}")
    return head_samples, eval_samples


def tokenize_prompt(
    tokenizer,
    prompt: str,
    *,
    max_input_length: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get(
        "attention_mask", torch.ones_like(input_ids)
    ).to(device)
    return input_ids, attention_mask


def build_input_manifest(
    tokenizer,
    samples: Sequence[Mapping[str, Any]],
    *,
    max_input_length: int,
) -> Dict[str, Any]:
    entries = []
    for index, sample in enumerate(samples):
        prompt = build_prompt(dict(sample))
        input_ids, _ = tokenize_prompt(
            tokenizer,
            prompt,
            max_input_length=max_input_length,
            device="cpu",
        )
        entries.append(
            {
                "index": index + 1,
                "sample_id": str(sample.get("_id")),
                "source_line": int(sample.get("_line_index", -1)),
                "prompt_sha256": _sha256_text(prompt),
                "input_ids_sha256": _input_ids_sha256(input_ids),
                "input_tokens": int(input_ids.shape[1]),
            }
        )
    sequence_hash = _sha256_text(
        "\n".join(
            (
                f"{entry['sample_id']}:{entry['prompt_sha256']}:"
                f"{entry['input_ids_sha256']}:{entry['input_tokens']}"
            )
            for entry in entries
        )
    )
    return {
        "schema_version": 1,
        "max_input_length": max_input_length,
        "count": len(entries),
        "sequence_sha256": sequence_hash,
        "samples": entries,
    }


def validate_official_baseline_alignment(
    input_manifest: Mapping[str, Any],
    baseline_results_paths: Sequence[str],
    *,
    require_fingerprints: bool = False,
) -> Dict[str, Any]:
    expected = input_manifest["samples"]
    checks: Dict[str, Any] = {}
    for path_string in baseline_results_paths:
        path = Path(path_string)
        if not path.is_file():
            raise FileNotFoundError(f"Missing baseline result file: {path}")
        rows = json.loads(path.read_text(encoding="utf-8"))
        if len(rows) < len(expected):
            raise ValueError(
                f"{path} has {len(rows)} rows; expected at least "
                f"{len(expected)}"
            )
        aligned_rows = rows[: len(expected)]
        mismatches = []
        for index, (baseline_row, expected_row) in enumerate(
            zip(aligned_rows, expected), start=1
        ):
            baseline_id = str(
                baseline_row.get("id", baseline_row.get("sample_id"))
            )
            if baseline_id != expected_row["sample_id"]:
                mismatches.append(
                    {
                        "index": index,
                        "field": "sample_id",
                        "expected": expected_row["sample_id"],
                        "actual": baseline_id,
                    }
                )
            if int(baseline_row.get("input_tokens", -1)) != int(
                expected_row["input_tokens"]
            ):
                mismatches.append(
                    {
                        "index": index,
                        "field": "input_tokens",
                        "expected": expected_row["input_tokens"],
                        "actual": baseline_row.get("input_tokens"),
                    }
                )
            for field in ("prompt_sha256", "input_ids_sha256"):
                actual = baseline_row.get(field)
                if (
                    require_fingerprints
                    and actual != expected_row[field]
                ):
                    mismatches.append(
                        {
                            "index": index,
                            "field": field,
                            "expected": expected_row[field],
                            "actual": actual,
                        }
                    )
        if mismatches:
            raise ValueError(
                f"Input alignment failed for {path}: {mismatches[:5]}"
            )
        checks[str(path)] = {
            "source_count": len(rows),
            "aligned_count": len(aligned_rows),
            "alignment_scope": (
                "full" if len(rows) == len(expected) else "ordered_prefix"
            ),
            "sample_id_order_match": True,
            "input_token_count_match": True,
            "prompt_sha256_match": (
                True if require_fingerprints else None
            ),
            "input_ids_sha256_match": (
                True if require_fingerprints else None
            ),
        }
    return checks


def load_model(
    args: argparse.Namespace, *, attention_implementation: str
):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map={"": args.device},
        trust_remote_code=True,
        attn_implementation=attention_implementation,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


def _attention_overlap_coverage_similarity(
    layer_attention: torch.Tensor,
    eps: float = 1e-12,
    token_chunk_size: int = 512,
) -> torch.Tensor:
    """Parameter-free coverage similarity over complete attention rows.

    Args:
        layer_attention: ``[heads, last_q, seq_len]``.

    Returns:
        Pairwise attention-overlap coefficient, averaged over query rows.
        For normalized rows this is equivalent to ``1 - total_variation``.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probabilities = layer_attention.to(
        device=device, dtype=torch.float32
    ).clamp_min(0)
    probabilities = probabilities / probabilities.sum(
        dim=-1, keepdim=True
    ).clamp_min(eps)
    num_heads, num_queries, seq_len = probabilities.shape
    l1_distance = torch.zeros(
        (num_heads, num_heads, num_queries),
        dtype=torch.float32,
        device=device,
    )
    for start in range(0, seq_len, token_chunk_size):
        chunk = probabilities[..., start : start + token_chunk_size]
        l1_distance.add_(
            torch.abs(chunk[:, None, :, :] - chunk[None, :, :, :]).sum(
                dim=-1
            )
        )
    overlap = 1.0 - 0.5 * l1_distance
    overlap.clamp_(min=0.0, max=1.0)
    diagonal = torch.arange(num_heads, device=device)
    overlap[diagonal, diagonal, :] = 1.0
    return overlap.mean(dim=-1).to(device="cpu")


def run_classification(
    args: argparse.Namespace,
    head_samples: Sequence[Mapping[str, Any]],
) -> Path:
    if args.classification_last_q != 32:
        raise ValueError("Formal representative selection requires last_q=32")
    if args.representative_metric == "coverage" and args.num_groups_per_layer != 1:
        raise ValueError("Coverage representative selection currently requires K=1")
    if args.representative_metric == "shareprefill_ae":
        raise ValueError(
            "Build SharePrefill autoencoder groups with "
            "run_shareprefill_autoencoder_clustering.py, then run stage=eval"
        )
    model, tokenizer = load_model(args, attention_implementation="eager")
    sample_ids: List[str] = []
    accumulator = JSDDistanceAccumulator()
    coverage_results: List[Dict[str, Any]] = []

    for index, sample in enumerate(head_samples):
        prompt = build_prompt(dict(sample))
        sample_id = str(sample.get("_id"))
        sample_ids.append(sample_id)
        attentions, seq_len, _, _ = collect_last_q_attentions(
            model, tokenizer, prompt, args
        )
        if attentions.shape[2] != 32:
            raise ValueError(
                "Expected 32 representative-selection query rows, got "
                f"{attentions.shape}"
            )
        if args.representative_metric == "jsd":
            accumulator.add(attentions)
        else:
            representative_heads: Dict[int, int] = {}
            coverage_scores: Dict[int, Dict[str, Any]] = {}
            for layer_idx in range(attentions.shape[0]):
                similarity = _attention_overlap_coverage_similarity(
                    attentions[layer_idx]
                )
                representative, score_per_head = select_representative_head(
                    similarity
                )
                representative_heads[layer_idx] = representative
                coverage_scores[layer_idx] = {
                    "coverage_score_per_head": score_per_head.tolist(),
                    "representative_head": representative,
                    "representative_score": float(
                        score_per_head[representative].item()
                    ),
                    "mean_coverage": float(similarity.mean().item()),
                    "min_coverage": float(similarity.min().item()),
                    "std_coverage": float(
                        similarity.std(unbiased=False).item()
                    ),
                }
            coverage_results.append(
                {
                    "sample_id": sample_id,
                    "input_tokens": seq_len,
                    "representative_heads": representative_heads,
                    "coverage_scores": coverage_scores,
                }
            )
        del attentions
        torch.cuda.empty_cache()
        logger.info(
            "%s calibration %d/%d complete",
            args.representative_metric.upper(),
            index + 1,
            len(head_samples),
        )

    jsd_configs: Dict[int, Dict[str, Any]] = {}
    if args.representative_metric == "jsd":
        mean_distances = accumulator.mean()
        group_counts = set(args.classification_group_counts or [])
        group_counts.add(args.num_groups_per_layer)
        for group_count in sorted(group_counts):
            jsd_configs[group_count] = build_group_config(
                mean_distances,
                num_groups=group_count,
                calibration_sample_ids=sample_ids,
                classification_last_q=args.classification_last_q,
            )
        config = jsd_configs[args.num_groups_per_layer]
    else:
        num_layers = int(model.config.num_hidden_layers)
        num_heads = int(model.config.num_attention_heads)
        aggregate = aggregate_global_representative_heads(
            coverage_results,
            num_layers,
            representative_selection="coverage",
        )
        config = {
            "schema_version": 1,
            "classification_metric": "token_attention_overlap_coverage",
            "classification_query_scope": "last_32",
            "classification_last_q": args.classification_last_q,
            "classification_formula": (
                "mean_q(sum_k(min(normalized_head_i, normalized_head_j)))"
            ),
            "classification_top_p": None,
            "classification_aggregation": (
                "per_sample_argmax_then_majority_vote_score_tiebreak"
            ),
            "num_groups_per_layer": 1,
            "num_calibration_samples": len(sample_ids),
            "calibration_sample_ids": sample_ids,
            "per_sample_representative_heads": {
                result["sample_id"]: {
                    str(layer_idx): int(head)
                    for layer_idx, head in result[
                        "representative_heads"
                    ].items()
                }
                for result in coverage_results
            },
            "per_layer_vote_detail": aggregate["per_layer_vote_detail"],
            "layers": {
                str(layer_idx): [
                    {
                        "representative": int(
                            aggregate["representative_heads"][layer_idx]
                        ),
                        "members": list(range(num_heads)),
                        "vote_detail": aggregate["per_layer_vote_detail"][
                            str(layer_idx)
                        ],
                    }
                ]
                for layer_idx in range(num_layers)
            },
        }
    online_metadata = {
        "model_name_or_path": args.model_name_or_path,
        "data_path": args.data_path,
        "online_last_q": args.online_last_q,
        "online_block_size": args.block_size,
        "online_top_p": args.top_p,
    }
    config.update(online_metadata)
    output_path = Path(args.output_dir) / "jsd_head_groups.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for group_count, additional_config in jsd_configs.items():
        additional_config.update(online_metadata)
        additional_path = (
            Path(args.output_dir)
            / f"jsd_head_groups_k{group_count}.json"
        )
        additional_path.write_text(
            json.dumps(additional_config, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(
            "Saved K=%d representative head groups to %s",
            group_count,
            additional_path,
        )
    logger.info("Saved representative head groups to %s", output_path)
    del model
    torch.cuda.empty_cache()
    return output_path


@torch.inference_mode()
def generate_with_timing(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    max_new_tokens: int,
) -> Tuple[str, List[int], float, float]:
    """Run the same HF greedy generation call used by official baselines."""

    prefill_start = torch.cuda.Event(enable_timing=True)
    prefill_end = torch.cuda.Event(enable_timing=True)
    timing_state = {"started": False, "finished": False}

    def before_forward(_module, _args, kwargs):
        ids = kwargs.get("input_ids")
        if (
            not timing_state["started"]
            and ids is not None
            and ids.ndim == 2
            and ids.shape[1] > 1
        ):
            prefill_start.record()
            timing_state["started"] = True

    def after_forward(_module, _args, _kwargs, _output):
        if timing_state["started"] and not timing_state["finished"]:
            prefill_end.record()
            timing_state["finished"] = True

    pre_handle = model.register_forward_pre_hook(
        before_forward, with_kwargs=True
    )
    post_handle = model.register_forward_hook(after_forward, with_kwargs=True)
    torch.cuda.synchronize()
    total_start = time.perf_counter()
    try:
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    finally:
        pre_handle.remove()
        post_handle.remove()
    torch.cuda.synchronize()
    total_seconds = time.perf_counter() - total_start

    if not timing_state["finished"]:
        raise RuntimeError("Could not identify the prefill forward during generate")
    prefill_seconds = prefill_start.elapsed_time(prefill_end) / 1000.0
    decode_seconds = max(total_seconds - prefill_seconds, 0.0)
    generated_tensor = output_ids[0, input_ids.shape[1] :]
    generated = [int(token) for token in generated_tensor.tolist()]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text, generated, prefill_seconds, decode_seconds


def run_eval(
    args: argparse.Namespace,
    eval_samples: Sequence[Mapping[str, Any]],
) -> None:
    flex_root = Path(args.flexprefill_root)
    if not flex_root.is_dir():
        raise FileNotFoundError(f"FlexPrefill root not found: {flex_root}")
    sys.path.insert(0, str(flex_root))

    group_path = Path(args.output_dir) / "jsd_head_groups.json"
    if not group_path.is_file():
        raise FileNotFoundError(
            f"Run classify first; missing group config: {group_path}"
        )
    group_config = json.loads(group_path.read_text(encoding="utf-8"))
    classification_metric = str(
        group_config.get("classification_metric", "")
    )
    configured_groups = int(group_config.get("num_groups_per_layer", -1))
    if configured_groups != args.num_groups_per_layer:
        raise ValueError(
            "Group config K does not match --num_groups_per_layer: "
            f"{configured_groups} != {args.num_groups_per_layer}"
        )
    if (
        classification_metric == "token_sqrt_jsd"
        and int(group_config.get("classification_last_q", -1)) != 32
    ):
        raise ValueError("JSD group config was not produced with last_q=32")
    if (
        classification_metric == "token_attention_overlap_coverage"
        and (
            group_config.get("classification_query_scope") != "last_32"
            or int(group_config.get("classification_last_q", -1)) != 32
            or group_config.get("classification_top_p") is not None
        )
    ):
        raise ValueError(
            "Coverage group config must use parameter-free last-32 overlap"
        )
    if classification_metric == "shareprefill_attention_map_autoencoder":
        autoencoder = group_config.get("autoencoder", {})
        if (
            int(autoencoder.get("latent_dim", -1)) != 64
            or int(autoencoder.get("map_size", -1)) != 1936
            or autoencoder.get("latent_normalization") != "l2"
        ):
            raise ValueError(
                "SharePrefill group config must use a 1936-map, 64-D, "
                "L2-normalized autoencoder representation"
            )
    elif classification_metric not in {
        "token_sqrt_jsd",
        "token_attention_overlap_coverage",
    }:
        raise ValueError(
            f"Unsupported group classification metric: {classification_metric}"
        )

    model, tokenizer = load_model(
        args, attention_implementation="flash_attention_2"
    )
    input_manifest = build_input_manifest(
        tokenizer,
        eval_samples,
        max_input_length=args.max_input_length,
    )
    baseline_checks = validate_official_baseline_alignment(
        input_manifest,
        args.baseline_results,
        require_fingerprints=args.require_baseline_fingerprints,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "canonical_input_manifest.json").write_text(
        json.dumps(input_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "baseline_alignment.json").write_text(
        json.dumps(baseline_checks, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if args.token_top_p is None:
        patch = install_grouped_flexprefill(
            model,
            group_config,
            block_size=args.block_size,
            gamma=args.top_p,
            tau=args.tau,
            min_budget=args.min_budget,
            max_budget=args.max_budget,
            force_sink_block=args.force_sink_block,
            force_diagonal_block=args.force_diagonal_block,
            dense_layers=args.dense_layers,
        )
    else:
        from experiments.token_compacted_sparse import (
            install_grouped_token_compacted_flexprefill,
        )

        patch = install_grouped_token_compacted_flexprefill(
            model,
            group_config,
            block_size=args.block_size,
            gamma=args.top_p,
            tau=args.tau,
            min_budget=args.min_budget,
            max_budget=args.max_budget,
            token_top_p=args.token_top_p,
            min_tokens_per_selected_block=(
                args.min_tokens_per_selected_block
            ),
            token_chunk_size=args.token_chunk_size,
            force_sink_block=args.force_sink_block,
            force_diagonal_block=args.force_diagonal_block,
        )

    for warmup_index, sample in enumerate(
        eval_samples[: args.warmup_num_samples], start=1
    ):
        warmup_prompt = build_prompt(dict(sample))
        warmup_input_ids, warmup_attention_mask = tokenize_prompt(
            tokenizer,
            warmup_prompt,
            max_input_length=args.max_input_length,
            device=args.device,
        )
        generate_with_timing(
            model,
            tokenizer,
            warmup_input_ids,
            warmup_attention_mask,
            max_new_tokens=args.max_new_tokens,
        )
        del warmup_input_ids, warmup_attention_mask
        torch.cuda.empty_cache()
        logger.info(
            "Completed unmeasured warmup %d/%d",
            warmup_index,
            args.warmup_num_samples,
        )
    if args.warmup_num_samples:
        patch.selector.stats = type(patch.selector.stats)()

    result_path = output_dir / "results.json"
    existing: List[Dict[str, Any]] = []
    if result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
    completed_ids = {str(row["id"]) for row in existing}
    rows = list(existing)

    for eval_index, sample in enumerate(eval_samples):
        if eval_index < args.eval_start:
            continue
        sample_id = str(sample.get("_id"))
        if sample_id in completed_ids:
            continue
        prompt = build_prompt(dict(sample))
        input_ids, attention_mask = tokenize_prompt(
            tokenizer,
            prompt,
            max_input_length=args.max_input_length,
            device=args.device,
        )
        expected = input_manifest["samples"][eval_index]
        if (
            _sha256_text(prompt) != expected["prompt_sha256"]
            or _input_ids_sha256(input_ids.cpu())
            != expected["input_ids_sha256"]
        ):
            raise RuntimeError(f"Input fingerprint changed for {sample_id}")

        before_stats = patch.selector.stats.snapshot()
        torch.cuda.reset_peak_memory_stats()
        text, token_ids, prefill_seconds, decode_seconds = generate_with_timing(
            model,
            tokenizer,
            input_ids,
            attention_mask,
            max_new_tokens=args.max_new_tokens,
        )
        after_stats = patch.selector.stats.snapshot()
        selected_blocks = (
            after_stats["selected_blocks"] - before_stats["selected_blocks"]
        )
        causal_blocks = (
            after_stats["causal_blocks"] - before_stats["causal_blocks"]
        )
        block_keep_ratio = (
            selected_blocks / causal_blocks if causal_blocks else float("nan")
        )
        selected_token_pairs = (
            after_stats.get("selected_token_pairs", 0)
            - before_stats.get("selected_token_pairs", 0)
        )
        causal_token_pairs = (
            after_stats.get("causal_token_pairs", 0)
            - before_stats.get("causal_token_pairs", 0)
        )
        token_pair_keep_ratio = (
            selected_token_pairs / causal_token_pairs
            if causal_token_pairs
            else float("nan")
        )
        compacted_key_tokens = (
            after_stats.get("compacted_key_tokens", 0)
            - before_stats.get("compacted_key_tokens", 0)
        )
        candidate_key_tokens = (
            after_stats.get("candidate_key_tokens", 0)
            - before_stats.get("candidate_key_tokens", 0)
        )
        intra_block_token_keep_ratio = (
            compacted_key_tokens / candidate_key_tokens
            if candidate_key_tokens
            else float("nan")
        )
        selector_latency_sec = (
            after_stats.get("selection_latency_sec", 0.0)
            - before_stats.get("selection_latency_sec", 0.0)
        )
        sparse_kernel_latency_sec = (
            after_stats.get("kernel_latency_sec", 0.0)
            - before_stats.get("kernel_latency_sec", 0.0)
        )
        prediction = extract_mcq_answer(text)
        gold = str(sample.get("answer", "")).strip().upper()
        row = {
            "index": eval_index + 1,
            "id": sample_id,
            "domain": sample.get("domain"),
            "sub_domain": sample.get("sub_domain"),
            "gold_answer": gold,
            "pred_answer": prediction,
            "correct": prediction == gold,
            "generation": text,
            "new_token_ids": token_ids,
            "input_tokens": int(input_ids.shape[1]),
            "prompt_sha256": expected["prompt_sha256"],
            "input_ids_sha256": expected["input_ids_sha256"],
            "prefill_latency_sec": prefill_seconds,
            "decode_latency_sec": decode_seconds,
            "latency_sec": prefill_seconds + decode_seconds,
            "selected_blocks": selected_blocks,
            "causal_blocks": causal_blocks,
            "block_keep_ratio": block_keep_ratio,
            "block_sparsity": 1.0 - block_keep_ratio,
            "selected_token_pairs": selected_token_pairs,
            "causal_token_pairs": causal_token_pairs,
            "token_pair_keep_ratio": token_pair_keep_ratio,
            "token_pair_sparsity": 1.0 - token_pair_keep_ratio,
            "compacted_key_tokens": compacted_key_tokens,
            "candidate_key_tokens": candidate_key_tokens,
            "intra_block_token_keep_ratio": (
                intra_block_token_keep_ratio
            ),
            "selector_latency_sec": selector_latency_sec,
            "sparse_kernel_latency_sec": sparse_kernel_latency_sec,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        }
        rows.append(row)
        result_path.write_text(
            json.dumps(rows, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if (eval_index + 1) % args.log_every == 0:
            logger.info(
                "eval %d/%d | acc=%.4f | prefill=%.3fs | keep=%.4f",
                eval_index + 1,
                len(eval_samples),
                sum(bool(item["correct"]) for item in rows) / len(rows),
                prefill_seconds,
                block_keep_ratio,
            )
        del input_ids, attention_mask
        torch.cuda.empty_cache()

    ordered_ids = [row["id"] for row in rows]
    expected_ids = [
        entry["sample_id"] for entry in input_manifest["samples"][: len(rows)]
    ]
    if ordered_ids != expected_ids:
        raise RuntimeError("Output rows are not in canonical eval order")

    summary = {
        "method": (
            f"{group_config.get('classification_metric', 'unknown')}_"
            f"k{configured_groups}_representative_head_block_top_p"
        ),
        "timing": "synchronized_wall_clock_with_cuda_event_prefill",
        "hardware": {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_total_memory_bytes": torch.cuda.get_device_properties(
                0
            ).total_memory,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
        "implementation": (
            (
                "project_representative_block_top_p_selector_"
                "token_compacted_triton_kernel"
            )
            if args.token_top_p is not None
            else (
                "project_representative_block_top_p_selector_"
                "flexprefill_triton_kernel"
            )
        ),
        "layer_attention_modes": {
            "indexing": "zero_based",
            "dense_layers": list(patch.dense_layers),
            "sparse_layers": [
                layer_idx
                for layer_idx in range(int(model.config.num_hidden_layers))
                if layer_idx not in set(patch.dense_layers)
            ],
            "dense_implementation": "original_flash_attention_2_forward",
            "block_keep_ratio_scope": "sparse_layers_only",
        },
        "mask_constraints": {
            "force_sink_block": args.force_sink_block,
            "force_diagonal_block": args.force_diagonal_block,
            "token_top_p": args.token_top_p,
            "min_tokens_per_selected_block": (
                args.min_tokens_per_selected_block
            ),
            "token_chunk_size": args.token_chunk_size,
        },
        "count": len(rows),
        "correct": sum(bool(row["correct"]) for row in rows),
        "accuracy": (
            sum(bool(row["correct"]) for row in rows) / len(rows)
            if rows
            else float("nan")
        ),
        "avg_prefill_latency_sec": (
            sum(row["prefill_latency_sec"] for row in rows) / len(rows)
            if rows
            else float("nan")
        ),
        "avg_block_keep_ratio": (
            sum(row["block_keep_ratio"] for row in rows) / len(rows)
            if rows
            else float("nan")
        ),
        "avg_token_pair_keep_ratio": (
            sum(row["token_pair_keep_ratio"] for row in rows) / len(rows)
            if rows and args.token_top_p is not None
            else float("nan")
        ),
        "avg_intra_block_token_keep_ratio": (
            sum(
                row["intra_block_token_keep_ratio"] for row in rows
            )
            / len(rows)
            if rows and args.token_top_p is not None
            else float("nan")
        ),
        "avg_selector_latency_sec": (
            sum(row["selector_latency_sec"] for row in rows) / len(rows)
            if rows and args.token_top_p is not None
            else float("nan")
        ),
        "avg_sparse_kernel_latency_sec": (
            sum(row["sparse_kernel_latency_sec"] for row in rows)
            / len(rows)
            if rows and args.token_top_p is not None
            else float("nan")
        ),
        "input_sequence_sha256": input_manifest["sequence_sha256"],
        "baseline_alignment": baseline_checks,
        "selector_stats": patch.selector.stats.snapshot(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Saved evaluation summary to %s", output_dir / "summary.json")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load with the same canonical loader used by the aligned Stage2 runner.
    loader_tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    del loader_tokenizer
    samples = load_longbench_v2_samples(args)
    head_samples, eval_samples = split_canonical_samples(
        samples,
        head_selection_num_samples=args.head_selection_num_samples,
        eval_num_samples=args.eval_num_samples,
    )
    logger.info(
        "Canonical split: head_selection=%d eval=%d first_eval_id=%s",
        len(head_samples),
        len(eval_samples),
        eval_samples[0].get("_id") if eval_samples else None,
    )

    if args.stage in {"classify", "all"}:
        run_classification(args, head_samples)
    if args.stage in {"eval", "all"}:
        run_eval(args, eval_samples)


if __name__ == "__main__":
    main()
