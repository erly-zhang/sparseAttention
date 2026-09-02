#!/usr/bin/env python3
"""Run Llama-3.1-8B-Instruct AutoBlock with AE K=8 on eight GPUs."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path


WORK = Path("/home/ubuntu/work")
RUNNER = WORK / "experiments/run_shareprefill_ae3_infinitebench.py"
BASELINE_ROOT = WORK / "experiments/outputs/infinitebench_multimodel_topk8192_20260818"
OUTPUT_ROOT = WORK / "experiments/outputs/infinitebench_llama31_autoblock_ae8_20260819"
LOG_ROOT = OUTPUT_ROOT / "logs"
STATUS_PATH = OUTPUT_ROOT / "scheduler_status.jsonl"
DONE_PATH = OUTPUT_ROOT / "scheduler_done.json"
PYTHON = Path("/home/ubuntu/miniconda3/envs/official_flex/bin/python")
MODEL = WORK / "model/Llama-3.1-8B-Instruct"
GROUP_CONFIG = (
    BASELINE_ROOT
    / "calibration/llama31_8b_instruct/shareprefill_ae_k8_head_groups.json"
)
METHOD = "shareprefill_ae8_token_block_auto"

TASK_COUNTS = {
    "longbook_sum_eng": 103,
    "longbook_qa_eng": 351,
    "longbook_choice_eng": 229,
    "longdialogue_qa_eng": 200,
    "longbook_qa_chn": 189,
    "code_debug": 394,
    "math_find": 350,
    "passkey": 590,
    "number_string": 590,
    "kv_retrieval": 497,
}
TASK_ORDER = (
    "passkey",
    "number_string",
    "kv_retrieval",
    "longbook_sum_eng",
    "code_debug",
    "longbook_qa_eng",
    "longbook_choice_eng",
    "longdialogue_qa_eng",
    "longbook_qa_chn",
    "math_find",
)

ready: queue.Queue[str | None] = queue.Queue()
status_lock = threading.Lock()


def result_dir(task: str) -> Path:
    return OUTPUT_ROOT / METHOD / task


def count_jsonl(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def is_complete(task: str) -> bool:
    directory = result_dir(task)
    summary_path = directory / "summary.json"
    if count_jsonl(directory / "online_metrics.jsonl") != TASK_COUNTS[task]:
        return False
    if not summary_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    alignment = summary.get("input_alignment", {})
    metadata = summary.get("method_metadata", {})
    return (
        alignment.get("status") == "passed"
        and alignment.get("alignment_scope") == "full"
        and int(alignment.get("count", -1)) == TASK_COUNTS[task]
        and int(alignment.get("reference_count", -1)) == TASK_COUNTS[task]
        and int(metadata.get("num_groups_per_layer", -1)) == 8
    )


def append_status(payload: dict) -> None:
    with status_lock:
        STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with STATUS_PATH.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"time": time.time(), **payload}) + "\n")


def command_for(task: str) -> list[str]:
    reference = (
        BASELINE_ROOT
        / "llama31/shareprefill_ae3_token_block_auto"
        / task
        / "online_metrics.jsonl"
    )
    return [
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
        "--output_dir",
        str(OUTPUT_ROOT),
        "--group_config",
        str(GROUP_CONFIG),
        "--calibration_scope",
        "benchmark_specific",
        "--reference_metrics",
        str(reference),
        "--record_sparsity",
    ]


def run_one(gpu: int, task: str) -> bool:
    if is_complete(task):
        append_status({"event": "skip_complete", "gpu": gpu, "task": task})
        return True
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = str(WORK)
    log_path = LOG_ROOT / f"gpu{gpu}_{task}.log"
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    for attempt in (1, 2):
        append_status(
            {"event": "start", "attempt": attempt, "gpu": gpu, "task": task}
        )
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n=== attempt {attempt} gpu={gpu} task={task} ===\n")
            log.flush()
            result = subprocess.run(
                command_for(task),
                cwd=WORK,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        success = result.returncode == 0 and is_complete(task)
        append_status(
            {
                "event": "finish" if success else "failed",
                "attempt": attempt,
                "returncode": result.returncode,
                "gpu": gpu,
                "task": task,
            }
        )
        if success:
            return True
        if attempt == 1:
            time.sleep(30)
    return False


def worker(gpu: int) -> None:
    while True:
        task = ready.get()
        try:
            if task is None:
                return
            run_one(gpu, task)
        finally:
            ready.task_done()


def main() -> None:
    if not GROUP_CONFIG.is_file():
        raise FileNotFoundError(GROUP_CONFIG)
    for task in TASK_ORDER:
        ready.put(task)
    workers = [
        threading.Thread(target=worker, args=(gpu,), daemon=True)
        for gpu in range(8)
    ]
    for thread in workers:
        thread.start()
    ready.join()
    for _ in workers:
        ready.put(None)
    ready.join()
    for thread in workers:
        thread.join()
    complete = {task: is_complete(task) for task in TASK_COUNTS}
    DONE_PATH.write_text(json.dumps(complete, indent=2), encoding="utf-8")
    if not all(complete.values()):
        raise RuntimeError(f"Incomplete tasks: {complete}")


if __name__ == "__main__":
    main()
