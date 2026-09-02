#!/usr/bin/env python3
"""Merge independently evaluated RULER task shards without changing sources."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

from benchmark_shareprefill_ae3 import (
    GenerationMetricsRecorder,
    validate_input_alignment,
)


TASKS = [
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def row_identity(row: dict) -> tuple:
    return (
        row.get("benchmark"),
        row.get("task"),
        row.get("sequence_length"),
        row.get("sample_index"),
        row.get("input_tokens"),
        row.get("input_ids_sha256"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_method_root", type=Path, required=True)
    parser.add_argument("--qa1_method_root", type=Path, required=True)
    parser.add_argument("--qa2_method_root", type=Path, required=True)
    parser.add_argument("--output_method_root", type=Path, required=True)
    parser.add_argument("--reference_metrics", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--evaluator_python", type=Path, required=True)
    parser.add_argument("--length", type=int, default=131072)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_method_root.exists() and any(args.output_method_root.iterdir()):
        raise RuntimeError(
            f"Refusing to overwrite non-empty output: {args.output_method_root}"
        )

    source_for_task = {
        **{task: args.main_method_root for task in TASKS[:-2]},
        "qa_1": args.qa1_method_root,
        "qa_2": args.qa2_method_root,
    }
    source_metric_rows = {
        root: read_jsonl(root / "online_metrics.jsonl")
        for root in set(source_for_task.values())
    }
    source_metric_ids = {
        root: {row_identity(row) for row in rows}
        for root, rows in source_metric_rows.items()
    }

    merged_metrics: list[dict] = []
    task_counts: dict[str, int] = {}
    for task in TASKS:
        source_root = source_for_task[task]
        prediction_path = source_root / str(args.length) / f"{task}.jsonl"
        predictions = sorted(
            read_jsonl(prediction_path), key=lambda row: int(row["index"])
        )
        if len(predictions) != 200:
            raise RuntimeError(f"{task}: expected 200 rows, got {len(predictions)}")
        indices = [int(row["index"]) for row in predictions]
        if len(set(indices)) != 200:
            raise RuntimeError(f"{task}: duplicate sample indices")

        output_predictions: list[dict] = []
        for prediction in predictions:
            metric = dict(prediction["online_metrics"])
            if row_identity(metric) not in source_metric_ids[source_root]:
                raise RuntimeError(
                    f"{task} index={prediction['index']}: embedded metric mismatch"
                )
            metric["call_index"] = len(merged_metrics)
            prediction = dict(prediction)
            prediction["online_metrics"] = metric
            output_predictions.append(prediction)
            merged_metrics.append(metric)

        write_jsonl(
            args.output_method_root / str(args.length) / f"{task}.jsonl",
            output_predictions,
        )
        task_counts[task] = len(output_predictions)

    if len(merged_metrics) != 2600:
        raise RuntimeError(f"Expected 2600 merged metrics, got {len(merged_metrics)}")
    alignment = validate_input_alignment(args.reference_metrics, merged_metrics)
    if alignment.get("alignment_scope") != "full":
        raise RuntimeError(f"Expected full alignment, got {alignment}")
    write_jsonl(args.output_method_root / "online_metrics.jsonl", merged_metrics)

    subprocess.run(
        [
            str(args.evaluator_python),
            str(args.evaluator),
            "--data_dir",
            str(args.output_method_root / str(args.length)),
            "--benchmark",
            "synthetic",
        ],
        check=True,
    )

    main_summary = json.loads(
        (args.main_method_root / "summary.json").read_text(encoding="utf-8")
    )
    fake_model = SimpleNamespace(generate=lambda *unused_args, **unused_kwargs: None)
    recorder = GenerationMetricsRecorder(
        fake_model,
        None,
        args.output_method_root / "online_metrics.jsonl",
        method="minference",
    )
    recorder.rows = merged_metrics
    summary = {
        "benchmark": "RULER",
        "method": "minference",
        "method_metadata": main_summary.get("method_metadata", {}),
        "hardware": main_summary.get("hardware", {}),
        "timing_scope": main_summary.get(
            "timing_scope", "synchronized model.generate; prefill via CUDA events"
        ),
        "runtime": recorder.summary(),
        "run_args": {
            "merge_only": True,
            "lengths": [args.length],
            "tasks": TASKS,
            "record_sparsity": True,
            "source_method_roots": [
                str(args.main_method_root),
                str(args.qa1_method_root),
                str(args.qa2_method_root),
            ],
        },
        "tasks": TASKS,
        "sequence_lengths": [args.length],
        "official_evaluation_directory": str(args.output_method_root),
        "input_alignment": alignment,
        "merge_manifest": {
            "task_counts": task_counts,
            "canonical_order": TASKS,
            "sources_preserved": True,
        },
    }
    (args.output_method_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    shutil.copy2(
        args.output_method_root / str(args.length) / "summary.csv",
        args.output_method_root / "official_ruler_summary.csv",
    )
    print(
        json.dumps(
            {
                "output": str(args.output_method_root),
                "count": len(merged_metrics),
                "alignment": alignment,
                "task_counts": task_counts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
