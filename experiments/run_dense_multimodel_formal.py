#!/usr/bin/env python3
"""Schedule aligned dense InfiniteBench and RULER runs on reserved GPUs."""

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
OUTPUT_ROOT = Path(
    "/local/experiments/outputs/dense_multimodel_20260820"
)
INFINITE_RUNNER = WORK / "experiments/run_dense_multimodel_infinitebench.py"
RULER_RUNNER = WORK / "experiments/run_dense_multimodel_ruler.py"
TASK_CONFIGS = WORK / "experiments/data/infinitebench_benchmark_specific_calibration/task_configs"
RULER_DATA = (
    WORK
    / "FlexPrefill/experiments/benchmark/ruler/data/qwen2_5_formal_200"
)
GPUS = (1, 2, 3, 5, 6, 7)
MAX_ATTEMPTS = 3

INFINITE_TASK_COUNTS = {
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
RULER_LENGTHS = (4096, 8192, 16384, 32768, 65536, 131072)


@dataclass(frozen=True)
class ModelSpec:
    label: str
    checkpoint: str
    infinite_chat: bool
    infinite_reference_root: str | None


@dataclass(frozen=True)
class Job:
    benchmark: str
    model: ModelSpec
    target: str
    expected_count: int
    attempt: int = 1


MODELS = (
    ModelSpec(
        "qwen25_base",
        "Qwen2.5-7B",
        False,
        str(
            WORK
            / "experiments/outputs/infinitebench_benchmark_specific_comparison/shareprefill_ae3_full"
        ),
    ),
    ModelSpec(
        "qwen2_instruct",
        "Qwen2-7B-Instruct-128K-YaRN",
        True,
        str(
            WORK
            / "experiments/outputs/infinitebench_multimodel_topk8192_20260818/qwen2/shareprefill_ae3_token_block_auto"
        ),
    ),
    ModelSpec(
        "llama31_instruct",
        "Llama-3.1-8B-Instruct",
        True,
        str(
            WORK
            / "experiments/outputs/infinitebench_multimodel_topk8192_20260818/llama31/shareprefill_ae3_token_block_auto"
        ),
    ),
    ModelSpec("llama31_base", "Llama-3.1-8B", False, None),
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
    if job.benchmark == "infinitebench":
        output = OUTPUT_ROOT / "infinitebench" / job.model.label
        method_dir = output / "dense" / job.target
        log = OUTPUT_ROOT / "logs" / f"infinite_{job.model.label}_{job.target}.log"
    else:
        output = (
            OUTPUT_ROOT
            / "ruler"
            / job.model.label
            / f"length_{job.target}"
        )
        method_dir = output / "dense"
        log = OUTPUT_ROOT / "logs" / f"ruler_{job.model.label}_{job.target}.log"
    return output, method_dir, log


def is_complete(job: Job) -> bool:
    _, method_dir, _ = paths_for(job)
    metrics = method_dir / "online_metrics.jsonl"
    summary = method_dir / "summary.json"
    if line_count(metrics) != job.expected_count or not summary.is_file():
        return False
    if job.benchmark == "ruler":
        official = method_dir / job.target / "summary.csv"
        if not official.is_file():
            return False
    return True


def command_for(job: Job) -> list[str]:
    output, _, _ = paths_for(job)
    model = str(WORK / "model" / job.model.checkpoint)
    if job.benchmark == "infinitebench":
        command = [
            PYTHON,
            str(INFINITE_RUNNER),
            "--model",
            model,
            "--method",
            "dense",
            "--task",
            job.target,
            "--max_length",
            "131072",
            "--batch_size",
            "1",
            "--seed",
            "42",
            "--chat" if job.model.infinite_chat else "--no-chat",
            "--output_dir",
            str(output),
            "--task_config_dir",
            str(TASK_CONFIGS),
        ]
        if job.model.infinite_reference_root:
            command.extend(
                [
                    "--reference_metrics",
                    str(
                        Path(job.model.infinite_reference_root)
                        / job.target
                        / "online_metrics.jsonl"
                    ),
                ]
            )
        else:
            command.append("--create_reference")
        return command
    return [
        PYTHON,
        str(RULER_RUNNER),
        "--model",
        model,
        "--method",
        "dense",
        "--lengths",
        job.target,
        "--seed",
        "42",
        "--no-chat",
        "--data_root",
        str(RULER_DATA),
        "--output_dir",
        str(output),
        "--reference_run",
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
            "benchmark": job.benchmark,
            "model": job.model.label,
            "target": job.target,
            "attempt": job.attempt,
            "gpu": gpu,
            "command": command,
        }
    )
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(
            f"\n===== attempt={job.attempt} gpu={gpu} time={now()} =====\n"
        )
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
                        "benchmark": job.benchmark,
                        "model": job.model.label,
                        "target": job.target,
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
                    "benchmark": job.benchmark,
                    "model": job.model.label,
                    "target": job.target,
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
    for task, count in INFINITE_TASK_COUNTS.items():
        for model in MODELS:
            JOBS.put(Job("infinitebench", model, task, count))
    for length in RULER_LENGTHS:
        for model in MODELS:
            JOBS.put(Job("ruler", model, str(length), 2600))


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    enqueue_jobs()
    append_status(
        {
            "event": "scheduler_started",
            "gpus": GPUS,
            "models": [model.label for model in MODELS],
            "queued_jobs": JOBS.qsize(),
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
