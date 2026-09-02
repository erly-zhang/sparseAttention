#!/usr/bin/env python3
"""Run and merge the matched-budget token-only KV Retrieval control."""

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
RUNNER = WORK / "experiments/run_dense_multimodel_infinitebench.py"
MODEL = WORK / "model/Llama-3.1-8B-Instruct"
METHOD = "shareprefill_ae3_representative_token_topk_protected"
TASK = "kv_retrieval"
TOTAL = 497
SOURCE_DATA = (
    WORK
    / "experiments/data/infinitebench_benchmark_specific_calibration/filtered_data"
)
SOURCE_CONFIG = (
    WORK
    / "experiments/data/infinitebench_benchmark_specific_calibration/task_configs"
)
REFERENCE = (
    WORK
    / "experiments/outputs/infinitebench_multimodel_topk8192_20260818"
    / "llama31/shareprefill_ae3_token_block_auto/kv_retrieval/online_metrics.jsonl"
)
GROUP_CONFIG = (
    WORK
    / "experiments/outputs/infinitebench_multimodel_topk8192_20260818"
    / "calibration/llama31_8b_instruct/shareprefill_ae_k3_head_groups.json"
)
ROOT = (
    WORK
    / "experiments/outputs"
    / "infinitebench_llama31_tokenonly_protected8192_fullkv_8gpu_20260825"
)
SHARD_ROOT = ROOT / "shards"
DATA_ROOT = ROOT / "shard_data"
LOG_ROOT = ROOT / "logs"
MERGED_ROOT = ROOT / "merged"
STATUS_PATH = ROOT / "scheduler_status.jsonl"
DONE_PATH = ROOT / "scheduler_done.json"

JOBS = (
    (0, 0, 63),
    (1, 63, 125),
    (2, 125, 187),
    (3, 187, 249),
    (4, 249, 311),
    (5, 311, 373),
    (6, 373, 435),
    (7, 435, 497),
)

status_lock = threading.Lock()


def sid(gpu: int, start: int, end: int) -> str:
    return f"gpu{gpu}_{TASK}_{start}_{end}"


def shard_output(gpu: int, start: int, end: int) -> Path:
    return SHARD_ROOT / sid(gpu, start, end)


def result_dir(gpu: int, start: int, end: int) -> Path:
    return shard_output(gpu, start, end) / METHOD / TASK


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def emit(event: str, **fields: object) -> None:
    with status_lock:
        STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with STATUS_PATH.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps({"time": time.time(), "event": event, **fields}) + "\n"
            )


def prepare_shards() -> None:
    source_path = SOURCE_DATA / f"{TASK}.jsonl"
    lines = source_path.read_text(encoding="utf-8").splitlines(keepends=True)
    if len(lines) != TOTAL:
        raise RuntimeError(f"Expected {TOTAL} source rows, got {len(lines)}")
    source_yaml = (SOURCE_CONFIG / f"{TASK}.yaml").read_text(encoding="utf-8")
    source_dir_text = str(SOURCE_DATA)
    if source_yaml.count(source_dir_text) != 1:
        raise RuntimeError("Unexpected data_dir declaration in task YAML")

    manifest = []
    for gpu, start, end in JOBS:
        shard = sid(gpu, start, end)
        data_dir = DATA_ROOT / shard / "filtered_data"
        config_dir = DATA_ROOT / shard / "task_configs"
        data_dir.mkdir(parents=True, exist_ok=True)
        config_dir.mkdir(parents=True, exist_ok=True)
        data_path = data_dir / f"{TASK}.jsonl"
        data_path.write_text("".join(lines[start:end]), encoding="utf-8")
        (config_dir / f"{TASK}.yaml").write_text(
            source_yaml.replace(source_dir_text, str(data_dir)), encoding="utf-8"
        )
        for helper in SOURCE_CONFIG.glob("*.py"):
            shutil.copy2(helper, config_dir / helper.name)
        manifest.append(
            {
                "gpu": gpu,
                "start_inclusive": start,
                "end_exclusive": end,
                "count": end - start,
                "data_path": str(data_path),
                "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
                "config_dir": str(config_dir),
                "output_root": str(shard_output(gpu, start, end)),
            }
        )
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "shard_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def is_complete(gpu: int, start: int, end: int) -> bool:
    directory = result_dir(gpu, start, end)
    metrics_path = directory / "online_metrics.jsonl"
    summary_path = directory / "summary.json"
    if not metrics_path.is_file() or not summary_path.is_file():
        return False
    try:
        rows = read_jsonl(metrics_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        alignment = summary["input_alignment"]
        metadata = summary["method_metadata"]
        online = metadata["online_config"]
        return (
            len(rows) == end - start
            and summary["runtime"]["count"] == end - start
            and alignment["status"] == "passed"
            and alignment["alignment_scope"] == "ordered_subsequence"
            and alignment["reference_count"] == TOTAL
            and metadata["num_groups_per_layer"] == 3
            and online["target_token_budget"] == 8192
            and online["whole_block_projection"] is False
            and online["final_token_budget"] == 8192
            and online["protected_token_span"] == 128
            and online["force_sink_block"] is True
            and online["force_diagonal_block"] is True
            and online["protected_tokens_count_within_budget"] is True
        )
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return False


def command(gpu: int, start: int, end: int) -> list[str]:
    config_dir = DATA_ROOT / sid(gpu, start, end) / "task_configs"
    return [
        str(PYTHON),
        str(RUNNER),
        "--model",
        str(MODEL),
        "--method",
        METHOD,
        "--task",
        TASK,
        "--max_length",
        "131072",
        "--batch_size",
        "1",
        "--seed",
        "42",
        "--chat",
        "--calibration_scope",
        "benchmark_specific",
        "--group_config",
        str(GROUP_CONFIG),
        "--task_config_dir",
        str(config_dir),
        "--output_dir",
        str(shard_output(gpu, start, end)),
        "--reference_metrics",
        str(REFERENCE),
        "--record_sparsity",
    ]


def run_job(gpu: int, start: int, end: int) -> None:
    if is_complete(gpu, start, end):
        emit("skip_complete", gpu=gpu, start=start, end=end)
        return
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "PYTHONPATH": "/home/ubuntu/work/FlexPrefill:/home/ubuntu/work",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"{sid(gpu, start, end)}.log"
    for attempt in (1, 2, 3):
        emit("started", gpu=gpu, start=start, end=end, attempt=attempt)
        with log_path.open("a", encoding="utf-8") as log:
            result = subprocess.run(
                command(gpu, start, end),
                cwd=WORK,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode == 0 and is_complete(gpu, start, end):
            emit("finished", gpu=gpu, start=start, end=end)
            return
        emit(
            "failed",
            gpu=gpu,
            start=start,
            end=end,
            attempt=attempt,
            returncode=result.returncode,
        )
        if attempt < 3:
            time.sleep(30)
    raise RuntimeError(f"Shard failed: {gpu=} {start=} {end=}")


def identity(row: dict) -> tuple:
    return (
        row.get("benchmark"),
        row.get("task"),
        row.get("sequence_length"),
        row.get("sample_index"),
        row.get("input_tokens"),
        row.get("input_ids_sha256"),
    )


def mean(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def sum_key(rows: list[dict], key: str) -> int:
    return sum(int(row.get(key, 0)) for row in rows)


def merge() -> None:
    actual_rows: list[dict] = []
    summaries: list[dict] = []
    for gpu, start, end in JOBS:
        directory = result_dir(gpu, start, end)
        actual_rows.extend(read_jsonl(directory / "online_metrics.jsonl"))
        summaries.append(json.loads((directory / "summary.json").read_text()))

    reference_rows = read_jsonl(REFERENCE)
    buckets: dict[tuple, deque[dict]] = defaultdict(deque)
    for row in actual_rows:
        buckets[identity(row)].append(row)
    merged: list[dict] = []
    for call_index, reference_row in enumerate(reference_rows):
        key = identity(reference_row)
        if not buckets[key]:
            raise RuntimeError(f"Missing aligned row {call_index}: {key}")
        row = buckets[key].popleft()
        row["call_index"] = call_index
        merged.append(row)
    leftovers = sum(len(rows) for rows in buckets.values())
    if leftovers or len(merged) != TOTAL:
        raise RuntimeError(f"Merge mismatch: rows={len(merged)} leftovers={leftovers}")

    output = MERGED_ROOT / METHOD / TASK
    write_jsonl(output / "online_metrics.jsonl", merged)

    score_key = next(
        key
        for key in summaries[0]["official_lm_eval_results"][TASK]
        if "get_score" in key and "stderr" not in key
    )
    score = sum(
        summary["official_lm_eval_results"][TASK][score_key]
        * summary["runtime"]["count"]
        for summary in summaries
    ) / TOTAL

    chosen: dict[str, int] = defaultdict(int)
    for row in merged:
        for size, count in row.get("chosen_block_size_rows", {}).items():
            chosen[str(size)] += int(count)
    selected_pairs = sum_key(merged, "selected_token_pairs")
    causal_pairs = sum_key(merged, "causal_token_pairs")
    compacted = sum_key(merged, "compacted_key_tokens")
    candidate_keys = sum_key(merged, "candidate_key_tokens")
    target = sum_key(merged, "target_mask_tokens")
    covered = sum_key(merged, "covered_target_tokens")
    keep = selected_pairs / causal_pairs

    watched: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    additive_watched = {
        "probability_mass",
        "selector_rows",
        "valid_token_slots",
        "target_token_slots",
        "selected_token_slots",
        "sink_token_slots",
    }
    for row in merged:
        for key, stats in row.get("watched_key_range_stats", {}).items():
            for name in additive_watched:
                watched[key][name] += float(stats.get(name, 0.0))
    watched_summary = {}
    for key, stats in watched.items():
        selector_rows = stats["selector_rows"]
        valid = stats["valid_token_slots"]
        watched_summary[key] = {
            **stats,
            "mean_probability_mass_per_selector_row": (
                stats["probability_mass"] / selector_rows if selector_rows else 0.0
            ),
            "target_keep_ratio": stats["target_token_slots"] / valid if valid else 0.0,
            "final_kernel_keep_ratio": (
                stats["selected_token_slots"] / valid if valid else 0.0
            ),
            "sink_protection_ratio": stats["sink_token_slots"] / valid if valid else 0.0,
        }

    runtime = {
        "count": TOTAL,
        "avg_input_tokens": mean(merged, "input_tokens"),
        "avg_generated_tokens": mean(merged, "generated_tokens"),
        "avg_prefill_latency_sec": mean(merged, "prefill_latency_sec"),
        "avg_decode_latency_sec": mean(merged, "decode_latency_sec"),
        "avg_total_latency_sec": mean(merged, "total_latency_sec"),
        "avg_block_keep_ratio": mean(merged, "block_keep_ratio"),
        "avg_global_token_keep_ratio": mean(merged, "global_token_keep_ratio"),
        "avg_global_token_sparsity": mean(merged, "global_token_sparsity"),
        "avg_within_selected_block_token_keep_ratio": mean(
            merged, "within_selected_block_token_keep_ratio"
        ),
        "avg_target_mask_coverage_ratio": mean(merged, "target_mask_coverage_ratio"),
        "chosen_block_size_rows": dict(chosen),
        "watched_key_range_stats": watched_summary,
        "max_peak_memory_bytes": max(int(row["peak_memory_bytes"]) for row in merged),
        "selected_token_pairs": selected_pairs,
        "causal_token_pairs": causal_pairs,
        "global_token_keep_ratio": keep,
        "global_token_sparsity": 1.0 - keep,
        "compacted_key_tokens": compacted,
        "candidate_key_tokens": candidate_keys,
        "within_selected_block_token_keep_ratio": compacted / candidate_keys,
        "within_selected_block_token_sparsity": 1.0 - compacted / candidate_keys,
        "target_mask_tokens": target,
        "covered_target_tokens": covered,
        "target_mask_coverage_ratio": covered / target,
    }
    first = summaries[0]
    summary = {
        "benchmark": "InfiniteBench",
        "method": METHOD,
        "method_metadata": first["method_metadata"],
        "hardware": first["hardware"],
        "timing_scope": first["timing_scope"],
        "runtime": runtime,
        "run_args": {
            **first.get("run_args", {}),
            "limit": -1,
            "output_dir": str(MERGED_ROOT),
            "task_config_dir": str(SOURCE_CONFIG),
        },
        "official_lm_eval_results": {TASK: {"alias": TASK, score_key: score}},
        "input_alignment": {
            "status": "passed",
            "reference_metrics": str(REFERENCE),
            "count": TOTAL,
            "reference_count": len(reference_rows),
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
            "source_shards": [sid(*job) for job in JOBS],
            "order": "full Llama K=3 AutoBlock reference identity order",
            "score_aggregation": "sample-count-weighted shard mean",
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output / "full_evaluation.json").write_text(
        json.dumps(
            {
                "task": TASK,
                "count": TOTAL,
                "accuracy": score,
                "score_key": score_key,
                "input_alignment": "passed/full",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    prepare_shards()
    emit("scheduler_start", jobs=len(JOBS), total=TOTAL)
    errors: list[str] = []

    def guarded(job: tuple[int, int, int]) -> None:
        try:
            run_job(*job)
        except Exception as exc:  # Keep other independent shards running.
            errors.append(repr(exc))
            emit("thread_error", job=job, error=repr(exc))

    threads = [threading.Thread(target=guarded, args=(job,)) for job in JOBS]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    incomplete = [job for job in JOBS if not is_complete(*job)]
    if errors or incomplete:
        emit("scheduler_failed", errors=errors, incomplete=incomplete)
        raise RuntimeError(f"errors={errors}; incomplete={incomplete}")
    merge()
    DONE_PATH.write_text(
        json.dumps(
            {
                "complete": True,
                "task": TASK,
                "count": TOTAL,
        "configuration": {
            "target_topk": 8192,
            "final_token_budget": 8192,
            "sink_local_span": 128,
            "whole_block_projection": False,
        },
                "merged_output": str(MERGED_ROOT / METHOD / TASK),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    emit("scheduler_complete", count=TOTAL)


if __name__ == "__main__":
    main()
