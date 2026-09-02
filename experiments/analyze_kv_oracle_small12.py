#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean


ROOT = Path(
    "/home/ubuntu/work/experiments/outputs/"
    "infinitebench_llama31_kv_oracle_diagnostic_20260823"
)
METHOD = "shareprefill_ae3_token_block_auto_fixed_mass_profile"
TASK = "kv_retrieval"
UUID = re.compile(
    r"(?i)(?<![0-9a-f])"
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}"
    r"(?![0-9a-f])"
)


def load(tag: str):
    path = ROOT / tag / METHOD / TASK
    metrics = [
        json.loads(line)
        for line in (path / "online_metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    evaluation = json.loads((path / "lm_eval_results.json").read_text())
    summary = json.loads((path / "summary.json").read_text())
    samples = evaluation["samples"][TASK]
    return path, metrics, samples, summary


def range_stats(row: dict) -> dict[str, float]:
    per_layer = row["per_layer_selector_stats"]
    start, end = row["resolved_watched_key_ranges"][0]
    range_name = f"{start}:{end}"
    values = []
    for _, layer in sorted(per_layer.items(), key=lambda item: int(item[0])):
        values.append(layer["watched_key_ranges"][range_name])
    return {
        "target_keep": mean(item["target_keep_ratio"] for item in values),
        "kernel_keep": mean(
            item["final_kernel_keep_ratio"] for item in values
        ),
        "probability_mass": mean(
            item["mean_probability_mass_per_selector_row"] for item in values
        ),
    }


def layer_profiles(rows: list[dict]) -> list[dict[str, float]]:
    profiles = []
    for layer_index in range(32):
        values = []
        for row in rows:
            start, end = row["resolved_watched_key_ranges"][0]
            values.append(
                row["per_layer_selector_stats"][str(layer_index)][
                    "watched_key_ranges"
                ][f"{start}:{end}"]
            )
        profiles.append(
            {
                "layer": layer_index,
                "target_keep": mean(
                    item["target_keep_ratio"] for item in values
                ),
                "kernel_keep": mean(
                    item["final_kernel_keep_ratio"] for item in values
                ),
                "probability_mass": mean(
                    item["mean_probability_mass_per_selector_row"]
                    for item in values
                ),
            }
        )
    return profiles


def bucket_profiles(profiles: list[dict[str, float]]) -> list[dict]:
    buckets = []
    for start in range(0, 32, 8):
        values = profiles[start : start + 8]
        buckets.append(
            {
                "layers": f"{start}-{start + 7}",
                "target_keep": mean(item["target_keep"] for item in values),
                "kernel_keep": mean(item["kernel_keep"] for item in values),
                "probability_mass": mean(
                    item["probability_mass"] for item in values
                ),
            }
        )
    return buckets


def classify(response: str, target: str, score: int) -> str:
    if score:
        return "correct"
    response_uuids = UUID.findall(response)
    if response_uuids:
        return "wrong_uuid"
    lowered = response.lower()
    if any(
        phrase in lowered
        for phrase in (
            "provide the key",
            "need to know the exact key",
            "cannot",
            "unable",
        )
    ):
        return "refusal_or_instruction"
    return "other_wrong"


baseline_path, baseline_rows, baseline_samples, baseline_summary = load(
    "stratified12_baseline_a3"
)
oracle_path, oracle_rows, oracle_samples, oracle_summary = load(
    "stratified12_oracle_a3"
)
assert len(baseline_rows) == len(oracle_rows) == 12
assert all(
    before["input_ids_sha256"] == after["input_ids_sha256"]
    for before, after in zip(baseline_rows, oracle_rows)
)

baseline_samples = sorted(baseline_samples, key=lambda item: item["doc_id"])
oracle_samples = sorted(oracle_samples, key=lambda item: item["doc_id"])
paired_samples = []
for before_sample, after_sample in zip(baseline_samples, oracle_samples):
    assert before_sample["doc_id"] == after_sample["doc_id"]
    before_score = int(before_sample["get_score_one_kv_retrieval"])
    after_score = int(after_sample["get_score_one_kv_retrieval"])
    target = str(before_sample["target"])
    before_response = str(before_sample["filtered_resps"][0])
    after_response = str(after_sample["filtered_resps"][0])
    paired_samples.append(
        {
            "doc_id": int(before_sample["doc_id"]),
            "target": target,
            "baseline_score": before_score,
            "oracle_score": after_score,
            "baseline_class": classify(
                before_response, target, before_score
            ),
            "oracle_class": classify(after_response, target, after_score),
            "baseline_response": before_response,
            "oracle_response": after_response,
        }
    )

paired_metrics = []
for before_row, after_row in zip(baseline_rows, oracle_rows):
    assert before_row["input_ids_sha256"] == after_row["input_ids_sha256"]
    before_range = range_stats(before_row)
    after_range = range_stats(after_row)
    start, end = before_row["resolved_watched_key_ranges"][0]
    paired_metrics.append(
        {
            "input_ids_sha256": before_row["input_ids_sha256"],
            "target_position_ratio": start / before_row["input_tokens"],
            "target_range_tokens": end - start,
            "baseline_generated_tokens": before_row["generated_tokens"],
            "oracle_generated_tokens": after_row["generated_tokens"],
            "baseline_target_keep": before_range["target_keep"],
            "baseline_kernel_keep": before_range["kernel_keep"],
            "oracle_kernel_keep": after_range["kernel_keep"],
            "target_probability_mass": before_range["probability_mass"],
        }
    )


def aggregate(rows: list[dict], samples: list[dict]) -> dict:
    watched = [range_stats(row) for row in rows]
    return {
        "count": len(rows),
        "accuracy": mean(
            int(sample["get_score_one_kv_retrieval"])
            for sample in samples
        ),
        "prefill_latency_sec": mean(row["prefill_latency_sec"] for row in rows),
        "decode_latency_sec": mean(row["decode_latency_sec"] for row in rows),
        "total_latency_sec": mean(row["total_latency_sec"] for row in rows),
        "generated_tokens": mean(row["generated_tokens"] for row in rows),
        "pair_keep": mean(row["global_token_keep_ratio"] for row in rows),
        "pair_sparsity": mean(row["global_token_sparsity"] for row in rows),
        "target_keep": mean(item["target_keep"] for item in watched),
        "kernel_keep": mean(item["kernel_keep"] for item in watched),
        "target_probability_mass": mean(
            item["probability_mass"] for item in watched
        ),
        "target_position_ratio_min": min(
            row["resolved_watched_key_ranges"][0][0] / row["input_tokens"]
            for row in rows
        ),
        "target_position_ratio_max": max(
            row["resolved_watched_key_ranges"][0][0] / row["input_tokens"]
            for row in rows
        ),
        "kernel_keep_min": min(item["kernel_keep"] for item in watched),
        "input_alignment": (
            baseline_summary["input_alignment"]
            if rows is baseline_rows
            else oracle_summary["input_alignment"]
        ),
    }


baseline_layers = layer_profiles(baseline_rows)
oracle_layers = layer_profiles(oracle_rows)
report = {
    "experiment": "Llama-3.1-8B-Instruct KV Retrieval budget-preserving oracle",
    "baseline": aggregate(baseline_rows, baseline_samples),
    "oracle": aggregate(oracle_rows, oracle_samples),
    "baseline_layer_profiles": baseline_layers,
    "oracle_layer_profiles": oracle_layers,
    "baseline_layer_buckets": bucket_profiles(baseline_layers),
    "oracle_layer_buckets": bucket_profiles(oracle_layers),
    "baseline_worst_layers_by_kernel_keep": sorted(
        baseline_layers, key=lambda item: item["kernel_keep"]
    )[:8],
    "paired_samples": paired_samples,
    "paired_metrics": paired_metrics,
    "paired_outcome_counts": {
        f"{before}->{after}": sum(
            item["baseline_class"] == before and item["oracle_class"] == after
            for item in paired_samples
        )
        for before in {
            item["baseline_class"] for item in paired_samples
        }
        for after in {item["oracle_class"] for item in paired_samples}
    },
}
(ROOT / "stratified12_paired_analysis.json").write_text(
    json.dumps(report, indent=2, ensure_ascii=True) + "\n"
)
print(json.dumps({
    key: value
    for key, value in report.items()
    if key not in {
        "paired_samples",
        "paired_metrics",
        "baseline_layer_profiles",
        "oracle_layer_profiles",
    }
}, indent=2))
for item in paired_metrics:
    print(
        f"pos={item['target_position_ratio']:.3f}",
        f"mass={item['target_probability_mass']:.6g}",
        f"target={item['baseline_target_keep']:.3f}",
        f"kernel={item['baseline_kernel_keep']:.3f}->"
        f"{item['oracle_kernel_keep']:.3f}",
    )
