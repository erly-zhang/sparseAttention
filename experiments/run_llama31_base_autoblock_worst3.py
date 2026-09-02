#!/usr/bin/env python3
"""Calibrate Llama-3.1-8B Base and run the three weakest AutoBlock tasks."""

from __future__ import annotations

import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


WORK = Path("/home/ubuntu/work")
PYTHON = Path("/home/ubuntu/miniconda3/envs/official_flex/bin/python")
MODEL = WORK / "model/Llama-3.1-8B"
CALIBRATOR = WORK / "experiments/run_shareprefill_autoencoder_clustering.py"
RUNNER = WORK / "experiments/run_shareprefill_ae3_infinitebench.py"
CALIBRATION_DATA = (
    WORK
    / "experiments/data/infinitebench_benchmark_specific_calibration/calibration.jsonl"
)
TASK_CONFIG_DIR = (
    WORK
    / "experiments/data/infinitebench_benchmark_specific_calibration/task_configs"
)
OUTPUT_ROOT = (
    WORK
    / "experiments/outputs/infinitebench_llama31_base_autoblock_worst3_20260820"
)
LOCAL_ROOT = Path(
    "/local/results/infinitebench_llama31_base_autoblock_worst3_20260820"
)
LOCAL_CALIBRATION = LOCAL_ROOT / "calibration/llama31_8b_base"
PERSISTENT_CALIBRATION = OUTPUT_ROOT / "calibration/llama31_8b_base"
GROUP_CONFIG = PERSISTENT_CALIBRATION / "shareprefill_ae_k3_head_groups.json"
LOG_ROOT = OUTPUT_ROOT / "logs"
STATUS_PATH = OUTPUT_ROOT / "scheduler_status.jsonl"
METHOD = "shareprefill_ae3_token_block_auto"
TASK_COUNTS = {
    "kv_retrieval": 497,
    "longbook_choice_eng": 229,
    "math_find": 350,
}
TASK_GPUS = {
    "kv_retrieval": 0,
    "longbook_choice_eng": 1,
    "math_find": 2,
}
PERSISTED_CALIBRATION_FILES = (
    "attention_map_manifest.json",
    "autoencoder_best.pt",
    "autoencoder_training.json",
    "normalized_latents.npy",
    "shareprefill_ae_k3_head_groups.json",
)


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


def run_logged(
    command: list[str],
    *,
    gpu: int,
    log_path: Path,
    stage: str,
    task: str | None = None,
) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = str(WORK)
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
        raise RuntimeError(f"{stage} failed for {task or 'calibration'}")


def validate_group_config() -> bool:
    if not GROUP_CONFIG.is_file():
        return False
    config = json.loads(GROUP_CONFIG.read_text(encoding="utf-8"))
    layers = config.get("layers", {})
    return (
        config.get("classification_metric")
        == "shareprefill_attention_map_autoencoder"
        and int(config.get("num_groups_per_layer", -1)) == 3
        and len(layers) == 32
    )


def run_calibration() -> None:
    if validate_group_config():
        append_status("skip_complete", stage="calibration")
        return
    command = [
        str(PYTHON),
        str(CALIBRATOR),
        "all",
        "--model_name_or_path",
        str(MODEL),
        "--data_path",
        str(CALIBRATION_DATA),
        "--calibration_format",
        "prompt_jsonl",
        "--artifact_dir",
        str(LOCAL_CALIBRATION),
        "--head_selection_num_samples",
        "3",
        "--map_size",
        "1936",
        "--latent_dim",
        "64",
        "--epochs",
        "1000",
        "--patience",
        "30",
        "--min_delta",
        "1e-7",
        "--learning_rate",
        "1e-3",
        "--batch_size",
        "1",
        "--groups",
        "3",
        "--seed",
        "42",
        "--device",
        "cuda",
    ]
    run_logged(
        command,
        gpu=0,
        log_path=LOG_ROOT / "calibration.log",
        stage="calibration",
    )
    PERSISTENT_CALIBRATION.mkdir(parents=True, exist_ok=True)
    for name in PERSISTED_CALIBRATION_FILES:
        source = LOCAL_CALIBRATION / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, PERSISTENT_CALIBRATION / name)
    if not validate_group_config():
        raise RuntimeError("Persisted K=3 calibration failed validation")
    append_status("validated", stage="calibration", group_config=str(GROUP_CONFIG))


def result_dir(root: Path, task: str) -> Path:
    return root / METHOD / task


def validate_run(root: Path, task: str, expected: int) -> bool:
    directory = result_dir(root, task)
    metrics = directory / "online_metrics.jsonl"
    summary_path = directory / "summary.json"
    if count_jsonl(metrics) != expected or not summary_path.is_file():
        return False
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    alignment = summary.get("input_alignment", {})
    metadata = summary.get("method_metadata", {})
    online = metadata.get("online_config", {})
    return (
        alignment.get("status") == "reference_created"
        and int(alignment.get("count", -1)) == expected
        and metadata.get("implementation")
        == "shareprefill_ae_k3_token_first_block_cover"
        and online.get("target_selection") == "fixed_top_k"
        and int(online.get("target_token_budget", -1)) == 8192
        and int(online.get("final_whole_block_token_budget", -1)) == 8192
    )


def run_task(task: str, *, smoke: bool) -> None:
    gpu = TASK_GPUS[task]
    root = OUTPUT_ROOT / ("smoke" if smoke else "formal")
    expected = 1 if smoke else TASK_COUNTS[task]
    if validate_run(root, task, expected):
        append_status(
            "skip_complete",
            stage="smoke" if smoke else "formal",
            task=task,
            gpu=gpu,
        )
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
        "--no-chat",
        "--output_dir",
        str(root),
        "--group_config",
        str(GROUP_CONFIG),
        "--calibration_scope",
        "benchmark_specific",
        "--task_config_dir",
        str(TASK_CONFIG_DIR),
        "--create_reference",
        "--record_sparsity",
    ]
    if smoke:
        command.extend(["--limit", "1"])
    stage = "smoke" if smoke else "formal"
    run_logged(
        command,
        gpu=gpu,
        log_path=LOG_ROOT / f"gpu{gpu}_{stage}_{task}.log",
        stage=stage,
        task=task,
    )
    if not validate_run(root, task, expected):
        raise RuntimeError(f"Validation failed for {stage} {task}")
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
            raise RuntimeError(f"Task failures: {failures}")


def main() -> None:
    append_status(
        "scheduler_start",
        model=str(MODEL),
        tasks=TASK_COUNTS,
        chat=False,
        method=METHOD,
    )
    run_calibration()
    run_parallel(smoke=True)
    run_parallel(smoke=False)
    append_status("scheduler_complete", tasks=TASK_COUNTS)
    (OUTPUT_ROOT / "scheduler_done.json").write_text(
        json.dumps(
            {
                "complete": True,
                "model": str(MODEL),
                "method": METHOD,
                "chat": False,
                "tasks": TASK_COUNTS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        append_status("scheduler_fatal", error=repr(exc))
        raise
