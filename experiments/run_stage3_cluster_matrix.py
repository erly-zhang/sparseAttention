#!/usr/bin/env python3
"""Run and validate the approved Stage 3 clustering experiment matrix.

The matrix compares token-JSD K-medoids and SharePrefill-style autoencoder
grouping at K=2, 3, and 4. Each grouping is evaluated with either complete
selected macro blocks or token compaction inside ordinary selected blocks.
Smoke runs must all pass before the formal 500-example phase can begin.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


WORK_ROOT = Path("/home/ubuntu/work")
EXPERIMENTS = WORK_ROOT / "experiments"
OUTPUTS = EXPERIMENTS / "outputs"
DATA_PATH = EXPERIMENTS / "data/longbench_v2_32k_full_7b.jsonl"
RUNNER = EXPERIMENTS / "run_jsd_grouped_efficiency_experiment.py"
JSD_CONFIG_ROOT = OUTPUTS / "stage3_jsd_multi_k_top_p_0.90_classification"
AE_CONFIG_ROOT = OUTPUTS / "shareprefill_autoencoder_calibration_v2"
MATRIX_ROOT = OUTPUTS / "stage3_cluster_matrix_top_p_0.90"

BASELINES = {
    "smoke": [
        OUTPUTS / "smoke_official_baseline_prefix10/minference/results.json",
        OUTPUTS / "smoke_official_baseline_prefix10/flexprefill/results.json",
    ],
    "formal": [
        OUTPUTS / "official_sparse_baselines_500eval/minference/results.json",
        OUTPUTS / "official_sparse_baselines_500eval/flexprefill/results.json",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=["smoke", "formal", "all"],
        default="all",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_eval_ids(count: int) -> list[str]:
    ids: list[str] = []
    with DATA_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("_sample_role") == "task_eval":
                ids.append(str(row.get("_id")))
                if len(ids) == count:
                    break
    if len(ids) != count:
        raise RuntimeError(f"Canonical data has {len(ids)} eval IDs, need {count}")
    return ids


def config_source(method: str, groups: int) -> Path:
    if method == "jsd":
        return JSD_CONFIG_ROOT / f"jsd_head_groups_k{groups}.json"
    return AE_CONFIG_ROOT / f"shareprefill_ae_k{groups}_head_groups.json"


def output_dir(
    phase: str, method: str, groups: int, strategy: str
) -> Path:
    count = 10 if phase == "smoke" else 500
    prefix = "smoke_stage3" if phase == "smoke" else "stage3"
    return OUTPUTS / (
        f"{prefix}_{method}_k{groups}_{strategy}_"
        f"top_p_0.90_{count}eval"
    )


def install_config(source: Path, destination_dir: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / "jsd_head_groups.json"
    if destination.is_file() and sha256(destination) != sha256(source):
        if (destination_dir / "results.json").is_file():
            raise RuntimeError(
                f"Refusing to change config for existing results: {destination_dir}"
            )
    shutil.copy2(source, destination)


def validate_run(run_dir: Path, expected_ids: list[str]) -> dict[str, Any]:
    results_path = run_dir / "results.json"
    summary_path = run_dir / "summary.json"
    alignment_path = run_dir / "baseline_alignment.json"
    for path in (results_path, summary_path, alignment_path):
        if not path.is_file():
            raise RuntimeError(f"Missing required output: {path}")

    rows = json.loads(results_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    ids = [str(row["id"]) for row in rows]
    if ids != expected_ids:
        raise RuntimeError(f"Canonical ID order mismatch in {run_dir}")
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"Duplicate eval IDs in {run_dir}")
    if int(summary["count"]) != len(expected_ids):
        raise RuntimeError(f"Summary count mismatch in {run_dir}")
    for baseline, check in alignment.items():
        if (
            int(check["count"]) != len(expected_ids)
            or check["sample_id_order_match"] is not True
            or check["input_token_count_match"] is not True
        ):
            raise RuntimeError(
                f"Official baseline alignment failed: {baseline}: {check}"
            )
    return summary


def run_one(
    phase: str, method: str, groups: int, strategy: str
) -> dict[str, Any]:
    count = 10 if phase == "smoke" else 500
    expected_ids = canonical_eval_ids(count)
    run_dir = output_dir(phase, method, groups, strategy)
    install_config(config_source(method, groups), run_dir)

    try:
        return validate_run(run_dir, expected_ids)
    except RuntimeError:
        pass

    command = [
        sys.executable,
        str(RUNNER),
        "eval",
        "--output_dir",
        str(run_dir),
        "--head_selection_num_samples",
        "3",
        "--eval_num_samples",
        str(count),
        "--representative_metric",
        "jsd" if method == "jsd" else "shareprefill_ae",
        "--num_groups_per_layer",
        str(groups),
        "--classification_last_q",
        "32",
        "--online_last_q",
        "128",
        "--block_size",
        "128",
        "--top_p",
        "0.90",
        "--tau",
        "0.1",
        "--min_budget",
        "1024",
        "--force_sink_block",
        "--force_diagonal_block",
        "--max_input_length",
        "32768",
        "--max_new_tokens",
        "8",
        "--seed",
        "42",
        "--log_every",
        "1" if phase == "smoke" else "10",
        "--baseline_results",
        *(str(path) for path in BASELINES[phase]),
    ]
    if strategy == "token_compacted":
        command.extend(
            [
                "--token_top_p",
                "0.90",
                "--min_tokens_per_selected_block",
                "16",
                "--token_chunk_size",
                "32",
            ]
        )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    log_path = run_dir / "run.log"
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\nCOMMAND: " + " ".join(command) + "\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=WORK_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{phase} failed for {method} K={groups} {strategy}; "
            f"see {log_path}"
        )
    return validate_run(run_dir, expected_ids)


def write_matrix_summary(records: list[dict[str, Any]]) -> None:
    MATRIX_ROOT.mkdir(parents=True, exist_ok=True)
    json_path = MATRIX_ROOT / "matrix_summary.json"
    csv_path = MATRIX_ROOT / "matrix_summary.csv"
    json_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    columns = [
        "phase",
        "method",
        "groups",
        "strategy",
        "count",
        "accuracy",
        "avg_prefill_latency_sec",
        "avg_block_keep_ratio",
        "avg_token_pair_keep_ratio",
        "avg_intra_block_token_keep_ratio",
        "output_dir",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in columns} for row in records)


def run_phase(phase: str, records: list[dict[str, Any]]) -> None:
    for method in ("jsd", "shareprefill_ae"):
        for groups in (2, 3, 4):
            for strategy in ("full_blocks", "token_compacted"):
                print(
                    f"[matrix] {phase}: {method} K={groups} {strategy}",
                    flush=True,
                )
                summary = run_one(phase, method, groups, strategy)
                record = {
                    "phase": phase,
                    "method": method,
                    "groups": groups,
                    "strategy": strategy,
                    "output_dir": str(
                        output_dir(phase, method, groups, strategy)
                    ),
                    **{
                        key: summary.get(key)
                        for key in (
                            "count",
                            "accuracy",
                            "avg_prefill_latency_sec",
                            "avg_block_keep_ratio",
                            "avg_token_pair_keep_ratio",
                            "avg_intra_block_token_keep_ratio",
                        )
                    },
                }
                records.append(record)
                write_matrix_summary(records)
                print(
                    f"[matrix] passed: accuracy={record['accuracy']:.4f} "
                    f"prefill={record['avg_prefill_latency_sec']:.3f}s",
                    flush=True,
                )


def main() -> None:
    args = parse_args()
    records: list[dict[str, Any]] = []
    if args.phase in {"smoke", "all"}:
        run_phase("smoke", records)
    if args.phase in {"formal", "all"}:
        if args.phase == "all":
            smoke_records = [
                row for row in records if row["phase"] == "smoke"
            ]
            if len(smoke_records) != 12:
                raise RuntimeError("Formal phase requires all 12 smoke runs")
        run_phase("formal", records)
    write_matrix_summary(records)


if __name__ == "__main__":
    main()
