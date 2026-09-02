#!/usr/bin/env python3
"""Launch one aligned 128K dense smoke per model on separate GPUs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


PYTHON = "/home/ubuntu/miniconda3/envs/official_flex/bin/python"
WORK = Path("/home/ubuntu/work")
INFINITE_RUNNER = WORK / "experiments/run_dense_multimodel_infinitebench.py"
RULER_RUNNER = WORK / "experiments/run_dense_multimodel_ruler.py"
TASK_CONFIGS = WORK / "experiments/data/infinitebench_benchmark_specific_calibration/task_configs"
RULER_DATA = (
    WORK
    / "FlexPrefill/experiments/benchmark/ruler/data/qwen2_5_formal_200"
)
OUTPUT_ROOT = WORK / "experiments/outputs/dense_multimodel_smoke_20260820"
MODELS = [
    ("qwen25_base", "Qwen2.5-7B", False, 1),
    ("qwen2_instruct", "Qwen2-7B-Instruct-128K-YaRN", True, 2),
    ("llama31_instruct", "Llama-3.1-8B-Instruct", True, 3),
    ("llama31_base", "Llama-3.1-8B", False, 5),
]


def command_for(label: str, model_name: str, chat: bool) -> list[str]:
    model = str(WORK / "model" / model_name)
    chat_flag = "--chat" if chat else "--no-chat"
    infinite_output = OUTPUT_ROOT / "infinitebench" / label
    ruler_output = OUTPUT_ROOT / "ruler" / label
    infinite = [
        PYTHON,
        str(INFINITE_RUNNER),
        "--model",
        model,
        "--method",
        "dense",
        "--task",
        "passkey",
        "--max_length",
        "131072",
        "--batch_size",
        "1",
        "--limit",
        "1",
        "--seed",
        "42",
        chat_flag,
        "--output_dir",
        str(infinite_output),
        "--task_config_dir",
        str(TASK_CONFIGS),
        "--create_reference",
    ]
    ruler = [
        PYTHON,
        str(RULER_RUNNER),
        "--model",
        model,
        "--method",
        "dense",
        "--tasks",
        "niah_single_1",
        "--lengths",
        "131072",
        "--limit_per_task",
        "1",
        "--seed",
        "42",
        "--no-chat",
        "--data_root",
        str(RULER_DATA),
        "--output_dir",
        str(ruler_output),
        "--reference_run",
    ]
    return infinite + ["__THEN_RULER__"] + ruler


def run_one(label: str, model_name: str, chat: bool, gpu: int) -> int:
    log_dir = OUTPUT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    combined = command_for(label, model_name, chat)
    marker = combined.index("__THEN_RULER__")
    commands = [combined[:marker], combined[marker + 1 :]]
    completion_markers = [
        OUTPUT_ROOT
        / "infinitebench"
        / label
        / "dense/passkey/summary.json",
        OUTPUT_ROOT / "ruler" / label / "dense/summary.json",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONPATH"] = f"{WORK}:{env.get('PYTHONPATH', '')}"
    status_path = OUTPUT_ROOT / f"{label}_status.json"
    with (log_dir / f"{label}.log").open("a", buffering=1) as log:
        for phase, command, completion_marker in zip(
            ("infinitebench", "ruler"), commands, completion_markers
        ):
            if completion_marker.is_file():
                continue
            status_path.write_text(
                json.dumps(
                    {
                        "label": label,
                        "model": model_name,
                        "gpu": gpu,
                        "phase": phase,
                        "command": command,
                    },
                    indent=2,
                )
            )
            result = subprocess.run(
                command,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
            if result.returncode != 0:
                status_path.write_text(
                    json.dumps(
                        {
                            "label": label,
                            "model": model_name,
                            "gpu": gpu,
                            "phase": phase,
                            "returncode": result.returncode,
                        },
                        indent=2,
                    )
                )
                return result.returncode
    status_path.write_text(
        json.dumps(
            {
                "label": label,
                "model": model_name,
                "gpu": gpu,
                "phase": "complete",
                "returncode": 0,
            },
            indent=2,
        )
    )
    return 0


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    children = []
    for label, model_name, chat, gpu in MODELS:
        pid = os.fork()
        if pid == 0:
            raise SystemExit(run_one(label, model_name, chat, gpu))
        children.append((label, pid))
    failed = []
    for label, pid in children:
        _, status = os.waitpid(pid, 0)
        code = os.waitstatus_to_exitcode(status)
        if code != 0:
            failed.append((label, code))
    if failed:
        print(json.dumps({"status": "failed", "failures": failed}))
        return 1
    print(json.dumps({"status": "complete", "models": len(children)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
