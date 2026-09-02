#!/usr/bin/env python3
"""Rerun aligned Qwen2.5-7B Base dense InfiniteBench without a chat template."""

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
OUTPUT_ROOT = (
    WORK
    / "experiments/outputs/dense_qwen25_base_nochat_aligned_20260822"
)
REFERENCE_ROOT = Path(
    "/local/experiments/outputs/dense_multimodel_20260820/"
    "infinitebench/qwen25_base/dense"
)
RUNNER = WORK / "experiments/run_dense_multimodel_infinitebench.py"
TASK_CONFIGS = (
    WORK
    / "experiments/data/infinitebench_benchmark_specific_calibration/task_configs"
)
MODEL = WORK / "model/Qwen2.5-7B"
GPUS = (1, 2, 3, 6)
MAX_ATTEMPTS = 3

TASK_COUNTS = {
    "passkey": 590,
    "number_string": 590,
    "kv_retrieval": 497,
    "math_find": 350,
    "code_debug": 394,
    "longbook_choice_eng": 229,
    "longbook_qa_eng": 351,
    "longdialogue_qa_eng": 200,
    "longbook_qa_chn": 189,
    "longbook_sum_eng": 103,
}

STATUS_PATH = OUTPUT_ROOT / "scheduler_status.jsonl"
STATUS_LOCK = threading.Lock()
JOBS: queue.Queue["Job"] = queue.Queue()


@dataclass(frozen=True)
class Job:
    task: str
    expected_count: int
    attempt: int = 1


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_status(event: dict) -> None:
    event = {"time": now(), **event}
    with STATUS_LOCK:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        with STATUS_PATH.open("a", encoding="utf-8", buffering=1) as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")


def line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def method_dir(job: Job) -> Path:
    return OUTPUT_ROOT / "dense" / job.task


def is_complete(job: Job) -> bool:
    output = method_dir(job)
    metrics = output / "online_metrics.jsonl"
    summary_path = output / "summary.json"
    if line_count(metrics) != job.expected_count or not summary_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    alignment = summary.get("input_alignment", {})
    run_args = summary.get("run_args", {})
    return (
        summary.get("method") == "dense"
        and run_args.get("chat") is False
        and alignment.get("status") == "passed"
        and alignment.get("alignment_scope") == "full"
        and alignment.get("count") == job.expected_count
        and alignment.get("reference_count") == job.expected_count
    )


def command_for(job: Job) -> list[str]:
    return [
        PYTHON,
        str(RUNNER),
        "--model",
        str(MODEL),
        "--method",
        "dense",
        "--task",
        job.task,
        "--max_length",
        "131072",
        "--batch_size",
        "1",
        "--seed",
        "42",
        "--no-chat",
        "--output_dir",
        str(OUTPUT_ROOT),
        "--task_config_dir",
        str(TASK_CONFIGS),
        "--reference_metrics",
        str(REFERENCE_ROOT / job.task / "online_metrics.jsonl"),
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
    log_path = OUTPUT_ROOT / "logs" / f"{job.task}.log"
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
                append_status({"event": "skipped_complete", "task": job.task, "gpu": gpu})
                continue
            wait_for_free_gpu(gpu)
            returncode = run_job(job, gpu)
            complete = is_complete(job)
            append_status(
                {
                    "event": "finished" if complete else "failed",
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


def main() -> int:
    missing_references = [
        task
        for task, count in TASK_COUNTS.items()
        if line_count(REFERENCE_ROOT / task / "online_metrics.jsonl") != count
    ]
    if missing_references:
        raise RuntimeError(f"Incomplete no-chat references: {missing_references}")

    for task, count in TASK_COUNTS.items():
        JOBS.put(Job(task, count))
    append_status(
        {
            "event": "scheduler_started",
            "model": str(MODEL),
            "chat": False,
            "gpus": GPUS,
            "queued_jobs": JOBS.qsize(),
            "reference_root": str(REFERENCE_ROOT),
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
    incomplete = [
        task for task, count in TASK_COUNTS.items() if not is_complete(Job(task, count))
    ]
    append_status({"event": "scheduler_complete", "incomplete_tasks": incomplete})
    return 1 if incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
