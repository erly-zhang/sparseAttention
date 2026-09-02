#!/usr/bin/env python3
"""Run dense or sparse methods on the official RULER adapter."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

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


LENGTHS = [4096, 8192, 16384, 32768, 65536, 131072]
TASKS = [
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
]
MAX_NEW_TOKENS = 256


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="/home/ubuntu/work/model/Qwen2.5-7B"
    )
    parser.add_argument(
        "--method", choices=METHODS, default="shareprefill_ae3_full"
    )
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    parser.add_argument(
        "--lengths", nargs="+", type=int, choices=LENGTHS, default=LENGTHS
    )
    parser.add_argument("--limit_per_task", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--chat",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Apply the tokenizer chat template. Disabled by default because the "
            "configured Qwen2.5-7B checkpoint is the base model."
        ),
    )
    parser.add_argument(
        "--data_root",
        default=(
            "/home/ubuntu/work/FlexPrefill/experiments/benchmark/"
            "ruler/data/qwen2_5"
        ),
    )
    parser.add_argument("--group_config")
    parser.add_argument(
        "--calibration_scope",
        choices=["benchmark_specific", "longbench_fixed"],
        default="benchmark_specific",
    )
    parser.add_argument(
        "--flexprefill_root", default=str(DEFAULT_FLEXPREFILL_ROOT)
    )
    parser.add_argument(
        "--output_dir",
        default=(
            "/home/ubuntu/work/experiments/outputs/"
            "ruler_benchmark_specific_comparison"
        ),
    )
    parser.add_argument(
        "--evaluator_python",
        default="/home/ubuntu/miniconda3/envs/official_flex/bin/python",
        help="Shared Python environment used only for the official evaluator.",
    )
    parser.add_argument("--reference_metrics")
    parser.add_argument(
        "--reference_run",
        action="store_true",
        help="Treat this method as the aligned input reference for the run.",
    )
    parser.add_argument(
        "--record_sparsity",
        action="store_true",
        help="Record final-kernel causal token-pair keep and sparsity.",
    )
    args = parser.parse_args()
    args.group_config = str(
        resolve_group_config(
            "ruler", args.calibration_scope, args.group_config
        )
    )
    return args


def read_jsonl(path: Path) -> list[Dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", buffering=1) as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_existing_metrics(path: Path) -> list[Dict[str, Any]]:
    if not path.is_file():
        return []
    return read_jsonl(path)


def tokenize_input(tokenizer, text: str, *, chat: bool, device: str):
    if chat:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)
    return tokenizer(
        text, return_tensors="pt", return_attention_mask=False
    ).input_ids.to(device)


def middle_truncate_input_ids(
    input_ids: torch.Tensor, max_input_length: int
) -> torch.Tensor:
    """Reserve decode positions after the requested prompt formatting."""

    if input_ids.shape[1] <= max_input_length:
        return input_ids
    left = max_input_length // 2
    right = max_input_length - left
    return torch.cat((input_ids[:, :left], input_ids[:, -right:]), dim=1)


def evaluate_outputs(
    flexprefill_root: Path,
    output_dir: Path,
    lengths: Iterable[int],
    evaluator_python: str,
) -> None:
    evaluator = (
        flexprefill_root
        / "experiments/benchmark/ruler/eval/evaluate.py"
    )
    for length in lengths:
        data_dir = output_dir / str(length)
        evaluator_env = os.environ.copy()
        evaluator_env["PYTHONPATH"] = str(
            _REPO_ROOT / "experiments/ruler_compat"
        )
        evaluator_env["PYTHONSAFEPATH"] = "1"
        subprocess.run(
            [
                evaluator_python,
                str(evaluator),
                "--data_dir",
                str(data_dir),
                "--benchmark",
                "synthetic",
            ],
            check=True,
            cwd="/tmp",
            env=evaluator_env,
        )


def main() -> None:
    args = parse_args()
    if args.method.startswith("shareprefill_ae3"):
        validate_group_config_scope(
            args.group_config,
            benchmark="ruler",
            calibration_scope=args.calibration_scope,
        )
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    compat_root = _REPO_ROOT / "experiments/ruler_compat"
    if str(compat_root) not in sys.path:
        sys.path.insert(0, str(compat_root))
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = (
        f"{compat_root}:{existing_pythonpath}"
        if existing_pythonpath
        else str(compat_root)
    )

    flex_root = Path(args.flexprefill_root)
    benchmark_root = flex_root / "experiments/benchmark"
    ruler_root = benchmark_root / "ruler"
    sys.path.insert(0, str(ruler_root))
    constants = importlib.import_module("data.synthetic.constants")
    with (ruler_root / "synthetic.yaml").open(encoding="utf-8") as stream:
        task_configs = yaml.safe_load(stream)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )
    model_label = Path(args.model).name.lower()
    if args.chat and "qwen2.5-7b" in model_label and "instruct" not in model_label:
        raise ValueError(
            "Qwen2.5-7B is a base checkpoint. Use --no-chat so RULER is "
            "evaluated as raw completion, or explicitly select an Instruct "
            "checkpoint before enabling --chat."
        )
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": {"": "cuda"},
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    model_kwargs["attn_implementation"] = "flash_attention_2"
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.eval()
    if args.method == "dense":
        install_generation_last_logit_hook(model)
    model_context_length = effective_context_length(model.config)
    max_input_length = model_context_length - MAX_NEW_TOKENS
    if max_input_length <= 0:
        raise ValueError(
            f"Model context {model_context_length} must exceed the "
            f"generation budget {MAX_NEW_TOKENS}"
        )
    args.model_context_length = model_context_length
    args.max_new_tokens = MAX_NEW_TOKENS
    args.max_input_length = max_input_length
    model, patch, method_metadata = install_benchmark_method(
        model,
        args.method,
        model_name=args.model,
        group_config_path=args.group_config,
        flexprefill_root=args.flexprefill_root,
        record_sparsity=args.record_sparsity,
        flexprefill_min_budget=1024,
    )
    method_metadata["dense_generation_last_logit_only"] = (
        args.method == "dense"
    )
    method_metadata["prompt_protocol"] = {
        "checkpoint_family": Path(args.model).name,
        "chat_template_applied": bool(args.chat),
        "format": "chat_template" if args.chat else "raw_completion",
    }

    warmup_ids = tokenizer(
        "Warmup " * 2048,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    ).input_ids.to(model.device)
    model.generate(
        warmup_ids,
        attention_mask=torch.ones_like(warmup_ids),
        max_new_tokens=1,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    if patch is not None and hasattr(patch, "selector"):
        patch.selector.stats = type(patch.selector.stats)()

    output_dir = Path(args.output_dir) / args.method
    reference_metrics = (
        Path(args.reference_metrics)
        if args.reference_metrics
        else Path(args.output_dir)
        / "shareprefill_ae3_full"
        / "online_metrics.jsonl"
    )
    if (
        not args.reference_run
        and args.method != "shareprefill_ae3_full"
        and not reference_metrics.is_file()
    ):
        raise FileNotFoundError(
            "Baseline runs require the SharePrefill input manifest: "
            f"{reference_metrics}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "online_metrics.jsonl"
    recorder = GenerationMetricsRecorder(
        model, patch, metrics_path, method=args.method
    )
    recorder.rows = load_existing_metrics(metrics_path)
    recorder.install()

    completed = 0
    for length in args.lengths:
        for task in args.tasks:
            if task not in task_configs:
                raise ValueError(f"Unknown RULER task: {task}")
            config = dict(task_configs[task])
            config.update(constants.TASKS[config["task"]])
            data_path = (
                Path(args.data_root)
                / str(length)
                / task
                / "validation.jsonl"
            )
            if not data_path.is_file():
                raise FileNotFoundError(f"Missing RULER data: {data_path}")
            samples = read_jsonl(data_path)
            if args.limit_per_task > 0:
                samples = samples[: args.limit_per_task]

            prediction_path = output_dir / str(length) / f"{task}.jsonl"
            existing = (
                read_jsonl(prediction_path)
                if prediction_path.is_file()
                else []
            )
            completed_indices = {int(row["index"]) for row in existing}
            for sample in samples:
                index = int(sample["index"])
                if index in completed_indices:
                    continue
                input_ids = tokenize_input(
                    tokenizer,
                    str(sample["input"]),
                    chat=args.chat,
                    device=str(model.device),
                )
                input_ids = middle_truncate_input_ids(
                    input_ids, max_input_length
                )
                recorder.set_context(
                    benchmark="ruler",
                    task=task,
                    sequence_length=length,
                    sample_index=index,
                )
                output_ids = model.generate(
                    input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    do_sample=False,
                    max_new_tokens=MAX_NEW_TOKENS,
                    pad_token_id=tokenizer.eos_token_id,
                )
                prediction = tokenizer.decode(
                    output_ids[0, input_ids.shape[1] :],
                    skip_special_tokens=True,
                )
                row = {
                    "index": index,
                    "pred": prediction,
                    "input": sample["input"],
                    "outputs": sample["outputs"],
                    "others": sample.get("others", {}),
                    "truncation": sample.get("truncation", -1),
                    "length": sample.get("length", length),
                    "online_metrics": recorder.rows[-1],
                }
                append_jsonl(prediction_path, row)
                completed += 1
                if completed % 10 == 0:
                    print(
                        f"completed={completed} task={task} length={length} "
                        f"index={index}",
                        flush=True,
                    )
                del input_ids, output_ids
                torch.cuda.empty_cache()

    recorder.uninstall()
    alignment = (
        validate_input_alignment(reference_metrics, recorder.rows)
        if not args.reference_run and args.method != "shareprefill_ae3_full"
        else {"status": "reference_created", "count": len(recorder.rows)}
    )
    evaluate_outputs(
        flex_root, output_dir, args.lengths, args.evaluator_python
    )
    write_benchmark_summary(
        output_dir / "summary.json",
        benchmark="RULER",
        method=args.method,
        recorder=recorder,
        run_args=vars(args),
        method_metadata=method_metadata,
        extra={
            "tasks": args.tasks,
            "sequence_lengths": args.lengths,
            "official_evaluation_directory": str(output_dir),
            "input_alignment": alignment,
        },
    )


if __name__ == "__main__":
    main()
