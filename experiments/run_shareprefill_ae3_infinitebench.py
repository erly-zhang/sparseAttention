#!/usr/bin/env python3
"""Run sparse methods on the official InfiniteBench adapter."""

from __future__ import annotations

import argparse
import json
import os
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
        "--record_sparsity",
        action="store_true",
        help=(
            "Record exact final-kernel causal token-pair sparsity without "
            "writing per-selector indices."
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
    if args.dense_layers and args.method != "shareprefill_ae3_full":
        raise ValueError(
            "--dense_layers is only valid for SharePrefill-AE3 Full"
        )
    return args


def main() -> None:
    args = parse_args()
    max_new_tokens = INFINITEBENCH_MAX_NEW_TOKENS[args.task]
    max_input_length = args.max_length - max_new_tokens
    if max_input_length <= 0:
        raise ValueError(
            f"Model context {args.max_length} must exceed the generation "
            f"budget {max_new_tokens} for {args.task}"
        )
    args.max_new_tokens = max_new_tokens
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

        original_padding_side = model.tokenizer.padding_side
        model.tokenizer.padding_side = padding_side
        try:
            encoded = model.tokenizer(
                strings,
                add_special_tokens=bool(model.add_bos_token),
            )
            max_length = model.max_length
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
    if (
        args.method != "shareprefill_ae3_full"
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
    configured_context = int(model._model.config.max_position_embeddings)
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
    )
    model._model = patched_model

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

    metrics_path = output_dir / "online_metrics.jsonl"
    recorder = GenerationMetricsRecorder(
        model._model, patch, metrics_path, method=args.method
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
    alignment = (
        validate_input_alignment(reference_metrics, recorder.rows)
        if args.method != "shareprefill_ae3_full" or args.reference_metrics
        else {"status": "reference_created", "count": len(recorder.rows)}
    )
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
