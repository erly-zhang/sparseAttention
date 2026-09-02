#!/usr/bin/env python3
"""Run the aligned 12-sample Llama AutoBlock oracle-residual matrix."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path


WORK = Path("/home/ubuntu/work")
ROOT = WORK / "experiments/outputs/infinitebench_llama31_oracle_residual_small12_20260824"
LOG_DIR = ROOT / "logs"
STATUS_PATH = ROOT / "scheduler_status.jsonl"
PYTHON = "/home/ubuntu/miniconda3/envs/official_flex/bin/python"
RUNNER = str(WORK / "experiments/run_kv_oracle_infinitebench.py")
METHOD = "shareprefill_ae3_token_block_auto_oracle_residual"
MODEL = str(WORK / "model/Llama-3.1-8B-Instruct")
GROUPS = str(
    WORK
    / "experiments/outputs/infinitebench_multimodel_topk8192_20260818/"
    "calibration/llama31_8b_instruct/shareprefill_ae_k3_head_groups.json"
)
REFERENCE_ROOT = (
    WORK
    / "experiments/outputs/infinitebench_multimodel_topk8192_20260818/"
    "llama31/shareprefill_ae3_token_block_auto"
)
KV_TASK_CONFIG = str(
    WORK / "experiments/data/kv_oracle_stratified12_20260823/task_configs"
)

CONFIGS = (
    ("a_shared8192_residual0", 8192, 0),
    ("b_shared7936_residual256", 7936, 256),
    ("c_shared7680_residual512", 7680, 512),
    ("d_shared7168_residual1024", 7168, 1024),
    ("e_shared6144_residual2048", 6144, 2048),
    ("upper_shared8192_residual1024", 8192, 1024),
)
TASKS = ("kv_retrieval", "passkey")


def emit(event: str, **fields: object) -> None:
    row = {"time": time.time(), "event": event, **fields}
    with STATUS_PATH.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def completed(config: str, task: str) -> bool:
    summary_path = ROOT / config / METHOD / task / "summary.json"
    metrics_path = ROOT / config / METHOD / task / "online_metrics.jsonl"
    if not summary_path.is_file() or not metrics_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text())
        alignment = summary["input_alignment"]
        return (
            summary["runtime"]["count"] == 12
            and sum(1 for _ in metrics_path.open()) == 12
            and alignment["status"] == "passed"
            and alignment["alignment_scope"] == "ordered_subsequence"
            and summary["method_metadata"]["online_config"]["residual_mode"]
            == "oracle"
        )
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return False


def command(config: str, shared: int, residual: int, task: str) -> list[str]:
    output_root = ROOT / config
    dump_root = output_root / "oracle_mask_dump" / task
    args = [
        PYTHON,
        RUNNER,
        "--model",
        MODEL,
        "--method",
        METHOD,
        "--task",
        task,
        "--limit",
        "12",
        "--chat",
        "--calibration_scope",
        "benchmark_specific",
        "--group_config",
        GROUPS,
        "--output_dir",
        str(output_root),
        "--reference_metrics",
        str(REFERENCE_ROOT / task / "online_metrics.jsonl"),
        "--fixed_topk_budget",
        str(shared),
        "--oracle_residual_tokens",
        str(residual),
        "--oracle_member_topk_budget",
        "8192",
        "--oracle_residual_dump_dir",
        str(dump_root),
        "--record_sparsity",
    ]
    if task == "kv_retrieval":
        args.extend(
            [
                "--task_config_dir",
                KV_TASK_CONFIG,
                "--profile_kv_retrieval_ranges",
            ]
        )
    return args


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    queue = [
        (config, shared, residual, task)
        for config, shared, residual in CONFIGS
        for task in TASKS
        if not completed(config, task)
    ]
    emit("scheduler_start", pending=len(queue), total=12)
    running: dict[int, tuple[subprocess.Popen[bytes], object, tuple[str, int, int, str]]] = {}
    failures: list[dict[str, object]] = []
    while queue or running:
        for gpu in range(8):
            if not queue or gpu in running:
                continue
            config, shared, residual, task = queue.pop(0)
            log_path = LOG_DIR / f"gpu{gpu}_{config}_{task}.log"
            log_stream = log_path.open("ab", buffering=0)
            env = os.environ.copy()
            env.update(
                {
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "PYTHONPATH": "/home/ubuntu/work/FlexPrefill:/home/ubuntu/work",
                    "TOKENIZERS_PARALLELISM": "false",
                }
            )
            process = subprocess.Popen(
                command(config, shared, residual, task),
                cwd=WORK,
                env=env,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
            )
            running[gpu] = (process, log_stream, (config, shared, residual, task))
            emit(
                "started",
                gpu=gpu,
                pid=process.pid,
                config=config,
                task=task,
                shared=shared,
                residual=residual,
                log=str(log_path),
            )
        time.sleep(5)
        for gpu, (process, log_stream, job) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            log_stream.close()
            config, shared, residual, task = job
            ok = code == 0 and completed(config, task)
            emit(
                "finished" if ok else "failed",
                gpu=gpu,
                pid=process.pid,
                returncode=code,
                config=config,
                task=task,
                complete=ok,
            )
            if not ok:
                failures.append(
                    {
                        "gpu": gpu,
                        "config": config,
                        "task": task,
                        "returncode": code,
                    }
                )
            del running[gpu]
    emit("scheduler_complete", failures=failures)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
