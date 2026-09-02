#!/usr/bin/env python3
"""Shard the three worst Llama AE3 tasks across eight GPUs and merge them."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import threading
import time
from collections import defaultdict, deque
from pathlib import Path


WORK = Path("/home/ubuntu/work")
PYTHON = Path("/home/ubuntu/miniconda3/envs/official_flex/bin/python")
RUNNER = WORK / "experiments/run_shareprefill_ae3_infinitebench.py"
MODEL = WORK / "model/Llama-3.1-8B-Instruct"
METHOD = "shareprefill_ae8_token_block_auto"
SOURCE_DATA = (
    WORK
    / "experiments/data/infinitebench_benchmark_specific_calibration/filtered_data"
)
SOURCE_CONFIG = (
    WORK
    / "experiments/data/infinitebench_benchmark_specific_calibration/task_configs"
)
REFERENCE_ROOT = (
    WORK
    / "experiments/outputs/infinitebench_multimodel_topk8192_20260818"
    / "llama31/shareprefill_ae3_token_block_auto"
)
GROUP_CONFIG = (
    WORK
    / "experiments/outputs/infinitebench_multimodel_topk8192_20260818"
    / "calibration/llama31_8b_instruct/shareprefill_ae_k8_head_groups.json"
)
OUTPUT_ROOT = (
    WORK
    / "experiments/outputs/infinitebench_llama31_autoblock_ae8_worst3_8gpu_20260819"
)
SHARD_ROOT = OUTPUT_ROOT / "shards"
DATA_ROOT = OUTPUT_ROOT / "shard_data"
LOG_ROOT = OUTPUT_ROOT / "logs"
MERGED_ROOT = OUTPUT_ROOT / "merged"
STATUS_PATH = OUTPUT_ROOT / "scheduler_status.jsonl"

TASK_COUNTS = {
    "kv_retrieval": 497,
    "longbook_choice_eng": 229,
    "math_find": 350,
}
JOBS = (
    (0, "kv_retrieval", 0, 166),
    (1, "kv_retrieval", 166, 332),
    (2, "kv_retrieval", 332, 497),
    (3, "longbook_choice_eng", 0, 115),
    (4, "longbook_choice_eng", 115, 229),
    (5, "math_find", 0, 117),
    (6, "math_find", 117, 234),
    (7, "math_find", 234, 350),
)

status_lock = threading.Lock()


def shard_id(gpu: int, task: str, start: int, end: int) -> str:
    return f"gpu{gpu}_{task}_{start}_{end}"


def shard_output(gpu: int, task: str, start: int, end: int) -> Path:
    return SHARD_ROOT / shard_id(gpu, task, start, end)


def result_dir(gpu: int, task: str, start: int, end: int) -> Path:
    return shard_output(gpu, task, start, end) / METHOD / task


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_status(payload: dict) -> None:
    with status_lock:
        STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with STATUS_PATH.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"time": time.time(), **payload}) + "\n")


def prepare_shards() -> None:
    manifest = []
    source_cache: dict[str, list[str]] = {}
    for task, count in TASK_COUNTS.items():
        lines = (SOURCE_DATA / f"{task}.jsonl").read_text(
            encoding="utf-8"
        ).splitlines(keepends=True)
        if len(lines) != count:
            raise RuntimeError(f"{task}: expected {count} source rows, got {len(lines)}")
        source_cache[task] = lines

    for gpu, task, start, end in JOBS:
        sid = shard_id(gpu, task, start, end)
        task_root = DATA_ROOT / sid
        data_dir = task_root / "filtered_data"
        config_dir = task_root / "task_configs"
        data_dir.mkdir(parents=True, exist_ok=True)
        config_dir.mkdir(parents=True, exist_ok=True)
        rows = source_cache[task][start:end]
        data_path = data_dir / f"{task}.jsonl"
        data_path.write_text("".join(rows), encoding="utf-8")

        source_yaml = (SOURCE_CONFIG / f"{task}.yaml").read_text(encoding="utf-8")
        old_data_dir = str(SOURCE_DATA)
        if source_yaml.count(old_data_dir) != 1:
            raise RuntimeError(f"Unexpected data_dir declaration in {task}.yaml")
        shard_yaml = source_yaml.replace(old_data_dir, str(data_dir))
        (config_dir / f"{task}.yaml").write_text(shard_yaml, encoding="utf-8")
        for helper in SOURCE_CONFIG.glob("*.py"):
            shutil.copy2(helper, config_dir / helper.name)

        manifest.append(
            {
                "gpu": gpu,
                "task": task,
                "source_start_inclusive": start,
                "source_end_exclusive": end,
                "count": end - start,
                "data_path": str(data_path),
                "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
                "config_dir": str(config_dir),
                "output_root": str(shard_output(gpu, task, start, end)),
            }
        )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "shard_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def expected_count(start: int, end: int) -> int:
    return end - start


def is_complete(gpu: int, task: str, start: int, end: int) -> bool:
    directory = result_dir(gpu, task, start, end)
    metrics = directory / "online_metrics.jsonl"
    summary_path = directory / "summary.json"
    if not metrics.is_file() or not summary_path.is_file():
        return False
    if len(read_jsonl(metrics)) != expected_count(start, end):
        return False
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    alignment = summary.get("input_alignment", {})
    metadata = summary.get("method_metadata", {})
    return (
        alignment.get("status") == "passed"
        and alignment.get("alignment_scope") == "ordered_subsequence"
        and int(alignment.get("count", -1)) == expected_count(start, end)
        and int(alignment.get("reference_count", -1)) == TASK_COUNTS[task]
        and int(metadata.get("num_groups_per_layer", -1)) == 8
    )


def command_for(gpu: int, task: str, start: int, end: int) -> list[str]:
    sid = shard_id(gpu, task, start, end)
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
        str(shard_output(gpu, task, start, end)),
        "--group_config",
        str(GROUP_CONFIG),
        "--calibration_scope",
        "benchmark_specific",
        "--task_config_dir",
        str(DATA_ROOT / sid / "task_configs"),
        "--reference_metrics",
        str(REFERENCE_ROOT / task / "online_metrics.jsonl"),
        "--record_sparsity",
    ]


def run_job(gpu: int, task: str, start: int, end: int) -> None:
    if is_complete(gpu, task, start, end):
        append_status({"event": "skip_complete", "gpu": gpu, "task": task})
        return
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = str(WORK)
    log_path = LOG_ROOT / f"{shard_id(gpu, task, start, end)}.log"
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    for attempt in (1, 2):
        append_status(
            {
                "event": "start",
                "attempt": attempt,
                "gpu": gpu,
                "task": task,
                "start": start,
                "end": end,
            }
        )
        with log_path.open("a", encoding="utf-8") as log:
            result = subprocess.run(
                command_for(gpu, task, start, end),
                cwd=WORK,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode == 0 and is_complete(gpu, task, start, end):
            append_status(
                {"event": "finish", "gpu": gpu, "task": task, "start": start, "end": end}
            )
            return
        append_status(
            {
                "event": "failed",
                "attempt": attempt,
                "returncode": result.returncode,
                "gpu": gpu,
                "task": task,
                "start": start,
                "end": end,
            }
        )
        if attempt == 1:
            time.sleep(30)
    raise RuntimeError(f"Shard failed: {gpu=} {task=} {start=} {end=}")


def identity(row: dict) -> tuple:
    return (
        row.get("benchmark"),
        row.get("task"),
        row.get("sequence_length"),
        row.get("sample_index"),
        row.get("input_tokens"),
        row.get("input_ids_sha256"),
    )


def mean(rows: list[dict], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else math.nan


def ratio_of_sums(rows: list[dict], numerator: str, denominator: str) -> float:
    num = sum(float(row.get(numerator, 0)) for row in rows)
    den = sum(float(row.get(denominator, 0)) for row in rows)
    return num / den if den else math.nan


def merge_task(task: str) -> None:
    task_jobs = [job for job in JOBS if job[1] == task]
    actual_rows = []
    summaries = []
    predictions = []
    for gpu, _, start, end in task_jobs:
        directory = result_dir(gpu, task, start, end)
        actual_rows.extend(read_jsonl(directory / "online_metrics.jsonl"))
        summaries.append(json.loads((directory / "summary.json").read_text()))
        prediction_path = directory / "math_find_predictions.jsonl"
        if prediction_path.is_file():
            predictions.extend(read_jsonl(prediction_path))

    reference = read_jsonl(REFERENCE_ROOT / task / "online_metrics.jsonl")
    buckets: dict[tuple, deque[dict]] = defaultdict(deque)
    for row in actual_rows:
        buckets[identity(row)].append(row)
    merged = []
    for call_index, reference_row in enumerate(reference):
        key = identity(reference_row)
        if not buckets[key]:
            raise RuntimeError(f"Missing aligned row for {task} call {call_index}")
        row = buckets[key].popleft()
        row["call_index"] = call_index
        merged.append(row)
    leftovers = sum(len(values) for values in buckets.values())
    if leftovers:
        raise RuntimeError(f"{task}: {leftovers} unmatched shard rows")
    if len(merged) != TASK_COUNTS[task]:
        raise RuntimeError(f"{task}: merged count mismatch")

    output = MERGED_ROOT / METHOD / task
    write_jsonl(output / "online_metrics.jsonl", merged)

    if predictions:
        prediction_buckets: dict[str, deque[dict]] = defaultdict(deque)
        for row in predictions:
            prediction_buckets[str(row["input_ids_sha256"])].append(row)
        ordered_predictions = []
        for metrics in merged:
            key = str(metrics["input_ids_sha256"])
            if not prediction_buckets[key]:
                raise RuntimeError(f"Missing Math.Find prediction for {key}")
            ordered_predictions.append(prediction_buckets[key].popleft())
        write_jsonl(output / "math_find_predictions.jsonl", ordered_predictions)

    score_key = next(
        key
        for key in summaries[0]["official_lm_eval_results"][task]
        if "get_score" in key and "stderr" not in key
    )
    weighted_score = sum(
        summary["official_lm_eval_results"][task][score_key]
        * summary["runtime"]["count"]
        for summary in summaries
    ) / TASK_COUNTS[task]
    chosen_sizes: dict[str, int] = defaultdict(int)
    for row in merged:
        for size, count in row.get("chosen_block_size_rows", {}).items():
            chosen_sizes[str(size)] += int(count)
    selected_pairs = sum(int(row.get("selected_token_pairs", 0)) for row in merged)
    causal_pairs = sum(int(row.get("causal_token_pairs", 0)) for row in merged)
    target_tokens = sum(int(row.get("target_mask_tokens", 0)) for row in merged)
    covered_tokens = sum(int(row.get("covered_target_tokens", 0)) for row in merged)
    global_keep = selected_pairs / causal_pairs
    runtime = {
        "count": len(merged),
        "avg_input_tokens": mean(merged, "input_tokens"),
        "avg_generated_tokens": mean(merged, "generated_tokens"),
        "avg_prefill_latency_sec": mean(merged, "prefill_latency_sec"),
        "avg_decode_latency_sec": mean(merged, "decode_latency_sec"),
        "avg_total_latency_sec": mean(merged, "total_latency_sec"),
        "avg_block_keep_ratio": mean(merged, "block_keep_ratio"),
        "avg_global_token_keep_ratio": mean(merged, "global_token_keep_ratio"),
        "avg_global_token_sparsity": mean(merged, "global_token_sparsity"),
        "avg_target_mask_coverage_ratio": mean(merged, "target_mask_coverage_ratio"),
        "chosen_block_size_rows": dict(chosen_sizes),
        "max_peak_memory_bytes": max(int(row["peak_memory_bytes"]) for row in merged),
        "selected_token_pairs": selected_pairs,
        "causal_token_pairs": causal_pairs,
        "global_token_keep_ratio": global_keep,
        "global_token_sparsity": 1.0 - global_keep,
        "target_mask_tokens": target_tokens,
        "covered_target_tokens": covered_tokens,
        "target_mask_coverage_ratio": covered_tokens / target_tokens,
    }
    first = summaries[0]
    summary = {
        "benchmark": "InfiniteBench",
        "method": METHOD,
        "method_metadata": first["method_metadata"],
        "hardware": first["hardware"],
        "timing_scope": first["timing_scope"],
        "runtime": runtime,
        "official_lm_eval_results": {
            task: {"alias": task, score_key: weighted_score}
        },
        "input_alignment": {
            "status": "passed",
            "reference_metrics": str(REFERENCE_ROOT / task / "online_metrics.jsonl"),
            "count": len(merged),
            "reference_count": len(reference),
            "alignment_scope": "full",
            "compared_fields": [
                "benchmark",
                "task",
                "sequence_length",
                "sample_index",
                "input_tokens",
                "input_ids_sha256",
            ],
        },
        "merge": {
            "source_shards": [shard_id(*job) for job in task_jobs],
            "order": "full K=3 reference identity order",
            "score_aggregation": "sample-count-weighted shard mean",
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def main() -> None:
    prepare_shards()
    threads = [threading.Thread(target=run_job, args=job) for job in JOBS]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    incomplete = [job for job in JOBS if not is_complete(*job)]
    if incomplete:
        raise RuntimeError(f"Incomplete shards: {incomplete}")
    for task in TASK_COUNTS:
        merge_task(task)
    (OUTPUT_ROOT / "scheduler_done.json").write_text(
        json.dumps({"complete": True, "tasks": TASK_COUNTS}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
