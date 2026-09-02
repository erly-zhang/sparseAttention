#!/usr/bin/env python3
"""Gate AutoBlock smoke and formal runs on completed model calibrations."""

from __future__ import annotations

import concurrent.futures
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


WORK = Path("/home/ubuntu/work")
LOCAL_ROOT = Path("/local/results/infinitebench_multimodel_topk8192_20260818")
OUTPUT_ROOT = (
    WORK / "experiments/outputs/infinitebench_multimodel_topk8192_20260818"
)
RUNNER = WORK / "experiments/run_shareprefill_ae3_infinitebench.py"
FORMAL_SCHEDULER = WORK / "experiments/run_multimodel_topk8192_formal.py"
FLEX_PYTHON = Path("/home/ubuntu/miniconda3/envs/official_flex/bin/python")
CALIBRATION_PIDS = (103002, 103003)

MODELS = {
    "qwen2": {
        "path": WORK / "model/Qwen2-7B-Instruct-128K-YaRN",
        "artifact": LOCAL_ROOT / "calibration/qwen2_7b_instruct",
        "expected_layers": 28,
    },
    "llama31": {
        "path": WORK / "model/Llama-3.1-8B-Instruct",
        "artifact": LOCAL_ROOT / "calibration/llama31_8b_instruct",
        "expected_layers": 32,
    },
}
TASKS = ("passkey", "number_string", "longbook_choice_eng")


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for_calibration() -> None:
    while any(process_alive(pid) for pid in CALIBRATION_PIDS):
        print("waiting for calibration", flush=True)
        time.sleep(30)


def validate_and_persist_calibration(model_id: str, spec: dict) -> Path:
    artifact = Path(spec["artifact"])
    group_path = artifact / "shareprefill_ae_k3_head_groups.json"
    if not group_path.is_file():
        raise FileNotFoundError(group_path)
    config = json.loads(group_path.read_text(encoding="utf-8"))
    if config.get("classification_metric") != (
        "shareprefill_attention_map_autoencoder"
    ):
        raise RuntimeError(f"Unexpected metric in {group_path}")
    if int(config.get("num_groups_per_layer", -1)) != 3:
        raise RuntimeError(f"Unexpected K in {group_path}")
    layers = config.get("layers", {})
    if len(layers) != int(spec["expected_layers"]):
        raise RuntimeError(
            f"Layer count mismatch for {model_id}: "
            f"{len(layers)} != {spec['expected_layers']}"
        )
    persistent = OUTPUT_ROOT / "calibration" / artifact.name
    persistent.mkdir(parents=True, exist_ok=True)
    for source in artifact.iterdir():
        if source.is_file():
            shutil.copy2(source, persistent / source.name)
    persisted_group = persistent / group_path.name
    if not persisted_group.is_file():
        raise FileNotFoundError(persisted_group)
    return persisted_group


def metrics_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def validate_smoke(model_id: str, method: str, task: str) -> bool:
    directory = LOCAL_ROOT / "smoke" / model_id / method / task
    summary_path = directory / "summary.json"
    if not summary_path.is_file():
        return False
    if metrics_count(directory / "online_metrics.jsonl") != 3:
        return False
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    alignment = summary.get("input_alignment", {})
    if method == "flexprefill":
        return (
            alignment.get("status") == "reference_created"
            and int(alignment.get("count", -1)) == 3
        )
    return (
        alignment.get("status") == "passed"
        and alignment.get("alignment_scope") == "full"
        and int(alignment.get("count", -1)) == 3
        and int(alignment.get("reference_count", -1)) == 3
    )


def run_autoblock_smoke(
    gpu: int, model_id: str, task: str, group_config: Path
) -> bool:
    spec = MODELS[model_id]
    reference = (
        LOCAL_ROOT
        / "smoke"
        / model_id
        / "flexprefill"
        / task
        / "online_metrics.jsonl"
    )
    log_path = LOCAL_ROOT / "logs/smoke" / f"{model_id}_autoblock_{task}.log"
    command = [
        str(FLEX_PYTHON),
        str(RUNNER),
        "--model",
        str(spec["path"]),
        "--method",
        "shareprefill_ae3_token_block_auto",
        "--task",
        task,
        "--limit",
        "3",
        "--max_length",
        "131072",
        "--batch_size",
        "1",
        "--seed",
        "42",
        "--output_dir",
        str(LOCAL_ROOT / "smoke" / model_id),
        "--group_config",
        str(group_config),
        "--calibration_scope",
        "benchmark_specific",
        "--reference_metrics",
        str(reference),
        "--record_sparsity",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = str(WORK)
    for attempt in (1, 2):
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n=== attempt {attempt} gpu={gpu} ===\n")
            result = subprocess.run(
                command,
                cwd=WORK,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode == 0 and validate_smoke(
            model_id, "shareprefill_ae3_token_block_auto", task
        ):
            return True
        if attempt == 1:
            time.sleep(30)
    return False


def main() -> None:
    wait_for_calibration()
    fatal_terms = ("Traceback", "CUDA out of memory", "Killed")
    for log_path in (
        LOCAL_ROOT / "logs/calibration_qwen2.log",
        LOCAL_ROOT / "logs/calibration_llama.log",
    ):
        text = log_path.read_text(encoding="utf-8", errors="replace")
        hits = [term for term in fatal_terms if term in text]
        if hits:
            raise RuntimeError(f"Calibration failure in {log_path}: {hits}")
    group_configs = {
        model_id: validate_and_persist_calibration(model_id, spec)
        for model_id, spec in MODELS.items()
    }
    jobs = []
    gpu = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        for model_id in MODELS:
            for task in TASKS:
                jobs.append(
                    (
                        model_id,
                        task,
                        executor.submit(
                            run_autoblock_smoke,
                            gpu,
                            model_id,
                            task,
                            group_configs[model_id],
                        ),
                    )
                )
                gpu += 1
        failures = [
            (model_id, task)
            for model_id, task, future in jobs
            if not future.result()
        ]
    if failures:
        raise RuntimeError(f"AutoBlock smoke failures: {failures}")
    for model_id in MODELS:
        for task in TASKS:
            for method in (
                "flexprefill",
                "minference",
                "shareprefill_ae3_token_block_auto",
            ):
                if not validate_smoke(model_id, method, task):
                    raise RuntimeError(
                        f"Smoke validation failed: {model_id}/{method}/{task}"
                    )
    subprocess.run(
        [str(FLEX_PYTHON), str(FORMAL_SCHEDULER)],
        cwd=WORK,
        env={**os.environ, "PYTHONPATH": str(WORK)},
        check=True,
    )


if __name__ == "__main__":
    main()
