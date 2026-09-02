#!/usr/bin/env python3
"""Run Qwen2-7B-Instruct AutoBlock Top-p=0.95 on three weak tasks."""

from __future__ import annotations

import concurrent.futures
import json
import os
import subprocess
import time
from pathlib import Path


WORK = Path("/home/ubuntu/work")
PYTHON = Path("/home/ubuntu/miniconda3/envs/official_flex/bin/python")
MODEL = WORK / "model/Qwen2-7B-Instruct-128K-YaRN"
RUNNER = WORK / "experiments/run_shareprefill_ae3_infinitebench.py"
TASK_CONFIG_DIR = (
    WORK
    / "experiments/data/infinitebench_benchmark_specific_calibration/task_configs"
)
TOPK_ROOT = (
    WORK / "experiments/outputs/infinitebench_multimodel_topk8192_20260818"
)
GROUP_CONFIG = (
    TOPK_ROOT
    / "calibration/qwen2_7b_instruct/shareprefill_ae_k3_head_groups.json"
)
OUTPUT_ROOT = (
    WORK
    / "experiments/outputs/infinitebench_qwen2_instruct_autoblock_topp95_worst3_20260820"
)
SMOKE_ROOT = OUTPUT_ROOT / "smoke"
FORMAL_ROOT = OUTPUT_ROOT / "formal"
LOG_ROOT = OUTPUT_ROOT / "logs"
STATUS_PATH = OUTPUT_ROOT / "scheduler_status.jsonl"
METHOD = "shareprefill_ae3_token_block_auto_topp"
TARGET_TOP_P = 0.95
TASK_COUNTS = {
    "kv_retrieval": 497,
    "longbook_choice_eng": 229,
    "math_find": 350,
}
TASK_GPUS = {
    "kv_retrieval": 4,
    "longbook_choice_eng": 5,
    "math_find": 6,
}


def append_status(event: str, **payload: object) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    record = {"time": time.time(), "event": event, **payload}
    with STATUS_PATH.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def count_jsonl(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def reference_metrics(task: str) -> Path:
    return (
        TOPK_ROOT
        / "qwen2/shareprefill_ae3_token_block_auto"
        / task
        / "online_metrics.jsonl"
    )


def result_dir(root: Path, task: str) -> Path:
    return root / METHOD / task


def validate_run(root: Path, task: str, expected: int, *, smoke: bool) -> bool:
    directory = result_dir(root, task)
    metrics_path = directory / "online_metrics.jsonl"
    summary_path = directory / "summary.json"
    if count_jsonl(metrics_path) != expected or not summary_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    alignment = summary.get("input_alignment", {})
    metadata = summary.get("method_metadata", {})
    online = metadata.get("online_config", {})
    expected_scope = "ordered_subsequence" if smoke else "full"
    return (
        alignment.get("status") == "passed"
        and alignment.get("alignment_scope") == expected_scope
        and int(alignment.get("count", -1)) == expected
        and int(alignment.get("reference_count", -1)) == TASK_COUNTS[task]
        and metadata.get("implementation")
        == "shareprefill_ae_k3_token_first_probability_block_cover"
        and online.get("target_selection") == "minimum_probability_prefix"
        and abs(float(online.get("target_token_top_p", -1.0)) - TARGET_TOP_P)
        < 1e-9
        and online.get("target_token_budget") is None
        and int(online.get("final_whole_block_token_budget", -1)) == 8192
        and abs(float(online.get("minimum_block_coverage_ratio", -1.0)) - 0.125)
        < 1e-9
        and online.get("logical_key_block_sizes") == [32, 64, 128]
        and online.get("force_sink_block") is True
        and online.get("force_diagonal_block") is True
    )


def run_task(task: str, *, smoke: bool) -> None:
    gpu = TASK_GPUS[task]
    root = SMOKE_ROOT if smoke else FORMAL_ROOT
    expected = 1 if smoke else TASK_COUNTS[task]
    stage = "smoke" if smoke else "formal"
    if validate_run(root, task, expected, smoke=smoke):
        append_status("skip_complete", stage=stage, task=task, gpu=gpu)
        return

    command = [
        str(PYTHON),
        str(RUNNER),
        "--model",
        str(MODEL),
        "--method",
        METHOD,
        "--task",
        task,
        "--max_length",
        "131072",
        "--batch_size",
        "1",
        "--seed",
        "42",
        "--chat",
        "--output_dir",
        str(root),
        "--group_config",
        str(GROUP_CONFIG),
        "--calibration_scope",
        "benchmark_specific",
        "--task_config_dir",
        str(TASK_CONFIG_DIR),
        "--reference_metrics",
        str(reference_metrics(task)),
        "--target_token_top_p",
        str(TARGET_TOP_P),
        "--record_sparsity",
    ]
    if smoke:
        command.extend(["--limit", "1"])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = str(WORK)
    log_path = LOG_ROOT / f"gpu{gpu}_{stage}_{task}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    append_status("start", stage=stage, task=task, gpu=gpu, log=str(log_path))
    with log_path.open("a", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=WORK,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    append_status(
        "finish" if result.returncode == 0 else "failed",
        stage=stage,
        task=task,
        gpu=gpu,
        returncode=result.returncode,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{stage} failed for {task}")
    if not validate_run(root, task, expected, smoke=smoke):
        raise RuntimeError(f"validation failed for {stage} {task}")
    append_status("validated", stage=stage, task=task, gpu=gpu, count=expected)


def run_parallel(*, smoke: bool) -> None:
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(run_task, task, smoke=smoke): task
            for task in TASK_COUNTS
        }
        failures = []
        for future, task in futures.items():
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                failures.append((task, repr(exc)))
        if failures:
            raise RuntimeError(f"task failures: {failures}")


def main() -> None:
    append_status(
        "scheduler_start",
        model=str(MODEL),
        method=METHOD,
        target_token_top_p=TARGET_TOP_P,
        final_whole_block_token_budget=8192,
        tasks=TASK_COUNTS,
        task_gpus=TASK_GPUS,
    )
    if not MODEL.joinpath("config.json").is_file():
        raise FileNotFoundError(MODEL / "config.json")
    if not GROUP_CONFIG.is_file():
        raise FileNotFoundError(GROUP_CONFIG)
    for task in TASK_COUNTS:
        if count_jsonl(reference_metrics(task)) != TASK_COUNTS[task]:
            raise RuntimeError(f"incomplete TopK reference for {task}")
    run_parallel(smoke=True)
    append_status("smoke_complete")
    run_parallel(smoke=False)
    append_status("scheduler_complete")


if __name__ == "__main__":
    main()
