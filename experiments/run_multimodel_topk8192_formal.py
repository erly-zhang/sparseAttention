#!/usr/bin/env python3
"""Run the approved two-model InfiniteBench matrix on eight GPUs."""

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
OUTPUT_ROOT = (
    WORK / "experiments/outputs/infinitebench_multimodel_topk8192_20260818"
)
LOG_ROOT = OUTPUT_ROOT / "logs"
STATUS_PATH = OUTPUT_ROOT / "scheduler_status.jsonl"
FLEX_PYTHON = Path("/home/ubuntu/miniconda3/envs/official_flex/bin/python")
MI_PYTHON = Path("/home/ubuntu/miniconda3/envs/official_mi/bin/python")

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

# Start the expensive jobs first; the dynamic queue balances later completions.
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

MODELS = {
    "qwen2": {
        "path": WORK / "model/Qwen2-7B-Instruct-128K-YaRN",
        "group_config": OUTPUT_ROOT
        / "calibration/qwen2_7b_instruct/shareprefill_ae_k3_head_groups.json",
    },
    "llama31": {
        "path": WORK / "model/Llama-3.1-8B-Instruct",
        "group_config": OUTPUT_ROOT
        / "calibration/llama31_8b_instruct/shareprefill_ae_k3_head_groups.json",
    },
}

METHODS = (
    "shareprefill_ae3_token_block_auto",
    "flexprefill",
    "minference",
)

ready: queue.Queue[tuple[str, str, str] | None] = queue.Queue()
status_lock = threading.Lock()


def output_dir(model_id: str) -> Path:
    return OUTPUT_ROOT / model_id


def result_dir(model_id: str, method: str, task: str) -> Path:
    return output_dir(model_id) / method / task


def metrics_path(model_id: str, method: str, task: str) -> Path:
    return result_dir(model_id, method, task) / "online_metrics.jsonl"


def append_status(payload: dict) -> None:
    payload = {"time": time.time(), **payload}
    with status_lock:
        STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with STATUS_PATH.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


def count_jsonl(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def is_complete(model_id: str, method: str, task: str) -> bool:
    directory = result_dir(model_id, method, task)
    summary_path = directory / "summary.json"
    metrics = directory / "online_metrics.jsonl"
    expected = TASK_COUNTS[task]
    if not summary_path.is_file() or count_jsonl(metrics) != expected:
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    alignment = summary.get("input_alignment", {})
    if method == "shareprefill_ae3_token_block_auto":
        return (
            alignment.get("status") == "reference_created"
            and int(alignment.get("count", -1)) == expected
        )
    return (
        alignment.get("status") == "passed"
        and alignment.get("alignment_scope") == "full"
        and int(alignment.get("count", -1)) == expected
        and int(alignment.get("reference_count", -1)) == expected
    )


def build_command(model_id: str, method: str, task: str) -> tuple[list[str], dict]:
    model = MODELS[model_id]
    python = MI_PYTHON if method == "minference" else FLEX_PYTHON
    command = [
        str(python),
        str(RUNNER),
        "--model",
        str(model["path"]),
        "--method",
        method,
        "--task",
        task,
        "--max_length",
        "131072",
        "--batch_size",
        "1",
        "--seed",
        "42",
        "--output_dir",
        str(output_dir(model_id)),
        "--record_sparsity",
    ]
    if method == "shareprefill_ae3_token_block_auto":
        command.extend(
            [
                "--group_config",
                str(model["group_config"]),
                "--calibration_scope",
                "benchmark_specific",
                "--create_reference",
            ]
        )
    else:
        command.extend(
            [
                "--reference_metrics",
                str(
                    metrics_path(
                        model_id,
                        "shareprefill_ae3_token_block_auto",
                        task,
                    )
                ),
            ]
        )
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{WORK / 'MInference'}:{WORK}"
        if method == "minference"
        else str(WORK)
    )
    return command, env


def run_one(gpu: int, model_id: str, method: str, task: str) -> bool:
    if is_complete(model_id, method, task):
        append_status(
            {
                "event": "skip_complete",
                "gpu": gpu,
                "model": model_id,
                "method": method,
                "task": task,
            }
        )
        return True
    command, env = build_command(model_id, method, task)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"gpu{gpu}_{model_id}_{method}_{task}.log"
    for attempt in (1, 2):
        append_status(
            {
                "event": "start",
                "attempt": attempt,
                "gpu": gpu,
                "model": model_id,
                "method": method,
                "task": task,
                "log": str(log_path),
            }
        )
        with log_path.open("a", encoding="utf-8") as log:
            log.write(
                f"\n=== attempt {attempt} gpu={gpu} "
                f"model={model_id} method={method} task={task} ===\n"
            )
            log.flush()
            result = subprocess.run(
                command,
                cwd=WORK,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        success = result.returncode == 0 and is_complete(
            model_id, method, task
        )
        append_status(
            {
                "event": "finish" if success else "failed",
                "attempt": attempt,
                "returncode": result.returncode,
                "gpu": gpu,
                "model": model_id,
                "method": method,
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
        item = ready.get()
        try:
            if item is None:
                return
            model_id, method, task = item
            success = run_one(gpu, model_id, method, task)
            if success and method == "shareprefill_ae3_token_block_auto":
                ready.put((model_id, "flexprefill", task))
                ready.put((model_id, "minference", task))
        finally:
            ready.task_done()


def main() -> None:
    for model in MODELS.values():
        if not Path(model["group_config"]).is_file():
            raise FileNotFoundError(model["group_config"])
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for task in TASK_ORDER:
        for model_id in MODELS:
            ready.put((model_id, "shareprefill_ae3_token_block_auto", task))
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
    complete = {
        model_id: {
            method: {
                task: is_complete(model_id, method, task)
                for task in TASK_COUNTS
            }
            for method in METHODS
        }
        for model_id in MODELS
    }
    (OUTPUT_ROOT / "scheduler_done.json").write_text(
        json.dumps(complete, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
