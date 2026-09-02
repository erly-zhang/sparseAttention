#!/usr/bin/env python3
"""Run the focused E1-E3 AutoBlock diagnosis on isolated GPUs."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path


PYTHON = "/home/ubuntu/miniconda3/envs/official_flex/bin/python"
WORK = Path("/home/ubuntu/work")
RUNNER = WORK / "experiments/run_dense_multimodel_infinitebench.py"
TASK_CONFIGS = (
    WORK / "experiments/data/infinitebench_benchmark_specific_calibration/task_configs"
)
OUTPUT_ROOT = (
    WORK / "experiments/outputs/autoblock_e1_e4_diagnostic_20260820/formal"
)
GPUS = (0, 4, 5, 7)
MAX_ATTEMPTS = 3

TASKS = (
    ("passkey", 590),
    ("kv_retrieval", 497),
    ("longbook_choice_eng", 229),
    ("math_find", 350),
)

METHODS = (
    "shareprefill_per_head_token_topk",
    "shareprefill_ae3_representative_token_topk",
    "shareprefill_per_head_token_block_auto",
)


@dataclass(frozen=True)
class ModelSpec:
    label: str
    checkpoint: str
    chat: bool
    group_config: str
    reference_root: str


@dataclass(frozen=True)
class Job:
    model: ModelSpec
    method: str
    task: str
    expected_count: int
    attempt: int = 1


MODELS = (
    ModelSpec(
        "qwen25_base",
        "Qwen2.5-7B",
        True,
        str(
            WORK
            / "experiments/outputs/infinitebench_shareprefill_ae_calibration_k3"
            / "shareprefill_ae_k3_head_groups.json"
        ),
        str(
            WORK
            / "experiments/outputs/infinitebench_benchmark_specific_comparison"
            / "shareprefill_ae3_full"
        ),
    ),
    ModelSpec(
        "qwen2_instruct",
        "Qwen2-7B-Instruct-128K-YaRN",
        True,
        str(
            WORK
            / "experiments/outputs/infinitebench_multimodel_topk8192_20260818"
            / "calibration/qwen2_7b_instruct/shareprefill_ae_k3_head_groups.json"
        ),
        str(
            WORK
            / "experiments/outputs/infinitebench_multimodel_topk8192_20260818"
            / "qwen2/shareprefill_ae3_token_block_auto"
        ),
    ),
    ModelSpec(
        "llama31_instruct",
        "Llama-3.1-8B-Instruct",
        True,
        str(
            WORK
            / "experiments/outputs/infinitebench_multimodel_topk8192_20260818"
            / "calibration/llama31_8b_instruct/shareprefill_ae_k3_head_groups.json"
        ),
        str(
            WORK
            / "experiments/outputs/infinitebench_multimodel_topk8192_20260818"
            / "llama31/shareprefill_ae3_token_block_auto"
        ),
    ),
)

STATUS_PATH = OUTPUT_ROOT / "scheduler_status.jsonl"
STATUS_LOCK = threading.Lock()
JOBS: queue.Queue[Job] = queue.Queue()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_status(event: dict) -> None:
    event = {"time": now(), **event}
    with STATUS_LOCK:
        STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with STATUS_PATH.open("a", encoding="utf-8", buffering=1) as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")


def line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def paths_for(job: Job) -> tuple[Path, Path, Path]:
    output = OUTPUT_ROOT / job.model.label
    method_dir = output / job.method / job.task
    log = OUTPUT_ROOT / "logs" / f"{job.model.label}_{job.method}_{job.task}.log"
    return output, method_dir, log


def expected_online_config(method: str) -> tuple[str, bool, bool]:
    if method == "shareprefill_per_head_token_topk":
        return "each_query_head", False, False
    if method == "shareprefill_ae3_representative_token_topk":
        return "ae_group_representative", True, False
    if method == "shareprefill_per_head_token_block_auto":
        return "each_query_head", False, True
    raise ValueError(method)


def is_complete(job: Job) -> bool:
    _, method_dir, _ = paths_for(job)
    metrics = method_dir / "online_metrics.jsonl"
    summary_path = method_dir / "summary.json"
    if line_count(metrics) != job.expected_count or not summary_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    alignment = summary.get("input_alignment", {})
    if not (
        alignment.get("status") == "passed"
        and alignment.get("alignment_scope") == "full"
        and alignment.get("count") == job.expected_count
        and alignment.get("reference_count") == job.expected_count
    ):
        return False
    if summary.get("method") != job.method:
        return False
    metadata = summary.get("method_metadata", {})
    if metadata.get("num_groups_per_layer") != 3:
        return False
    online = metadata.get("online_config", {})
    source, broadcast, projection = expected_online_config(job.method)
    return (
        online.get("mask_source") == source
        and online.get("group_mask_broadcast") is broadcast
        and online.get("whole_block_projection") is projection
    )


def command_for(job: Job) -> list[str]:
    output, _, _ = paths_for(job)
    return [
        PYTHON,
        str(RUNNER),
        "--model",
        str(WORK / "model" / job.model.checkpoint),
        "--method",
        job.method,
        "--task",
        job.task,
        "--max_length",
        "131072",
        "--batch_size",
        "1",
        "--seed",
        "42",
        "--chat" if job.model.chat else "--no-chat",
        "--group_config",
        job.model.group_config,
        "--calibration_scope",
        "benchmark_specific",
        "--output_dir",
        str(output),
        "--task_config_dir",
        str(TASK_CONFIGS),
        "--reference_metrics",
        str(Path(job.model.reference_root) / job.task / "online_metrics.jsonl"),
        "--record_sparsity",
    ]


def wait_for_free_gpu(gpu: int) -> None:
    while True:
        result = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(gpu),
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and not result.stdout.strip():
            return
        time.sleep(30)


def run_job(job: Job, gpu: int) -> int:
    output, _, log_path = paths_for(job)
    output.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = command_for(job)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONPATH"] = f"{WORK}:{env.get('PYTHONPATH', '')}"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    append_status(
        {
            "event": "started",
            "model": job.model.label,
            "method": job.method,
            "task": job.task,
            "attempt": job.attempt,
            "gpu": gpu,
            "command": command,
        }
    )
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(f"\n===== attempt={job.attempt} gpu={gpu} time={now()} =====\n")
        result = subprocess.run(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return result.returncode


def worker(gpu: int) -> None:
    while True:
        try:
            job = JOBS.get(timeout=2)
        except queue.Empty:
            return
        try:
            if is_complete(job):
                append_status(
                    {
                        "event": "skipped_complete",
                        "model": job.model.label,
                        "method": job.method,
                        "task": job.task,
                        "gpu": gpu,
                    }
                )
                continue
            wait_for_free_gpu(gpu)
            returncode = run_job(job, gpu)
            complete = is_complete(job)
            append_status(
                {
                    "event": "finished" if complete else "failed",
                    "model": job.model.label,
                    "method": job.method,
                    "task": job.task,
                    "attempt": job.attempt,
                    "gpu": gpu,
                    "returncode": returncode,
                    "complete": complete,
                }
            )
            if not complete and job.attempt < MAX_ATTEMPTS:
                JOBS.put(replace(job, attempt=job.attempt + 1))
        finally:
            JOBS.task_done()


def enqueue_jobs() -> None:
    for task, count in TASKS:
        for model in MODELS:
            for method in METHODS:
                JOBS.put(Job(model, method, task, count))


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    enqueue_jobs()
    append_status(
        {
            "event": "scheduler_started",
            "gpus": GPUS,
            "models": [model.label for model in MODELS],
            "methods": METHODS,
            "tasks": [task for task, _ in TASKS],
            "queued_jobs": JOBS.qsize(),
            "queued_samples": sum(count for _, count in TASKS)
            * len(MODELS)
            * len(METHODS),
        }
    )
    threads = [
        threading.Thread(target=worker, args=(gpu,), name=f"gpu-{gpu}")
        for gpu in GPUS
    ]
    for thread in threads:
        thread.start()
    JOBS.join()
    for thread in threads:
        thread.join()
    append_status({"event": "scheduler_complete"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
