#!/usr/bin/env python3
"""Export ranked key-token probability dumps as lazy-loaded browser assets."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer


def parse_mapping(value: str) -> tuple[str, Path]:
    try:
        name, path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected NAME=PATH") from exc
    if not name or not path:
        raise argparse.ArgumentTypeError("Expected NAME=PATH")
    return name, Path(path).resolve()


def encode_float32(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values, dtype="<f4")
    return base64.b64encode(contiguous.tobytes()).decode("ascii")


def load_metric_row(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as stream:
        line = stream.readline()
    return json.loads(line) if line else {}


def answer_token_metadata(
    input_ids: np.ndarray,
    watched_ranges: list[list[int]],
    tokenizer,
) -> list[dict[str, Any]]:
    tokens = []
    for range_index, (start, end) in enumerate(watched_ranges):
        for position in range(start, min(end, input_ids.shape[0])):
            token_id = int(input_ids[position])
            tokens.append(
                {
                    "range_index": range_index,
                    "position": position,
                    "token_id": token_id,
                    "token_text": tokenizer.convert_ids_to_tokens(token_id),
                    "decoded": tokenizer.decode(
                        [token_id], skip_special_tokens=False
                    ),
                }
            )
    return tokens


def export_task(
    task: str,
    dump_dir: Path,
    output_dir: Path,
    tokenizer,
    metric_path: Path | None,
) -> dict[str, Any]:
    input_payload = torch.load(
        dump_dir / "input_tokens.pt", map_location="cpu", weights_only=False
    )
    input_ids = input_payload["input_ids"].numpy().astype(np.int32, copy=False)
    watched_ranges = [
        [int(start), int(end)]
        for start, end in input_payload.get("watched_key_ranges", [])
    ]
    answer_tokens = answer_token_metadata(
        input_ids, watched_ranges, tokenizer
    )
    layer_files = sorted(dump_dir.glob("layer_*.pt"))
    if len(layer_files) != 28:
        raise RuntimeError(f"{task}: expected 28 layer files, found {len(layer_files)}")

    task_dir = output_dir / task
    task_dir.mkdir(parents=True, exist_ok=True)
    layer_manifest = []
    for layer_file in layer_files:
        payload = torch.load(layer_file, map_location="cpu", weights_only=False)
        layer = int(payload["layer"])
        probabilities = payload["sorted_probabilities"].numpy().astype(
            np.float32, copy=False
        )
        indices = payload["sorted_key_indices"].numpy().astype(
            np.int32, copy=False
        )
        groups = []
        for group_index in range(probabilities.shape[0]):
            group_probabilities = probabilities[group_index]
            group_indices = indices[group_index]
            cumulative = np.cumsum(group_probabilities, dtype=np.float64)
            k95 = int(np.searchsorted(cumulative, 0.95, side="left") + 1)
            inverse_ranks = np.empty(group_indices.shape[0], dtype=np.int32)
            inverse_ranks[group_indices] = np.arange(
                group_indices.shape[0], dtype=np.int32
            )
            answer_points = []
            for token in answer_tokens:
                position = token["position"]
                rank_zero = int(inverse_ranks[position])
                answer_points.append(
                    {
                        **token,
                        "rank": rank_zero + 1,
                        "probability": float(group_probabilities[rank_zero]),
                        "cumulative_mass": float(cumulative[rank_zero]),
                        "inside_top_p": rank_zero < k95,
                    }
                )
            groups.append(
                {
                    "group_index": group_index,
                    "representative": int(payload["groups"][group_index]["representative"]),
                    "members": [
                        int(member)
                        for member in payload["groups"][group_index]["members"]
                    ],
                    "k95": k95,
                    "answer_probability_mass": float(
                        sum(point["probability"] for point in answer_points)
                    ),
                    "answer_points": answer_points,
                    "probabilities_base64": encode_float32(group_probabilities),
                }
            )

        layer_data = {
            "task": task,
            "layer": layer,
            "source": payload["source"],
            "sequence_length": int(payload["sequence_length"]),
            "query_tile_start": int(payload["query_tile_start"]),
            "query_tile_end": int(payload["query_tile_end"]),
            "query_tile_length": int(payload["query_tile_length"]),
            "probe_offsets_in_tile": [
                int(value) for value in payload["probe_offsets_in_tile"].tolist()
            ],
            "probe_weights": [
                float(value) for value in payload["probe_weights"].tolist()
            ],
            "groups": groups,
        }
        key = f"{task}:{layer:02d}"
        script = (
            "window.__rankedProbabilityData = "
            "window.__rankedProbabilityData || {};\n"
            f"window.__rankedProbabilityData[{json.dumps(key)}] = "
            f"{json.dumps(layer_data, ensure_ascii=False, separators=(',', ':'))};\n"
            "window.dispatchEvent(new CustomEvent('ranked-probability-loaded', "
            f"{{detail: {json.dumps(key)}}}));\n"
        )
        relative_path = f"{task}/layer_{layer:02d}.js"
        (output_dir / relative_path).write_text(script, encoding="utf-8")
        layer_manifest.append(
            {
                "layer": layer,
                "source": payload["source"],
                "script": relative_path,
                "sequence_length": int(payload["sequence_length"]),
                "query_tile_start": int(payload["query_tile_start"]),
                "query_tile_end": int(payload["query_tile_end"]),
                "groups": [
                    {
                        "representative": group["representative"],
                        "members": group["members"],
                        "k95": group["k95"],
                        "answer_probability_mass": group[
                            "answer_probability_mass"
                        ],
                    }
                    for group in groups
                ],
            }
        )

    metric = load_metric_row(metric_path)
    return {
        "task": task,
        "display_name": "PassKey" if task == "passkey" else "Number String",
        "sequence_length": int(input_ids.shape[0]),
        "watched_key_ranges": watched_ranges,
        "answer_tokens": answer_tokens,
        "input_ids_sha256": metric.get("input_ids_sha256"),
        "input_tokens": metric.get("input_tokens", int(input_ids.shape[0])),
        "layers": layer_manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", action="append", type=parse_mapping, required=True)
    parser.add_argument("--metrics", action="append", type=parse_mapping, default=[])
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--method",
        default="shareprefill_ae3_token_block_auto_dense_topp_mass",
    )
    parser.add_argument(
        "--selection_mode",
        choices=["top_p", "top_k"],
        default="top_p",
    )
    parser.add_argument("--target_top_p", type=float, default=0.95)
    parser.add_argument("--target_top_k", type=int)
    parser.add_argument("--first_sparse_layer", type=int, default=10)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = dict(args.metrics)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True
    )
    tasks = [
        export_task(
            task,
            dump_dir,
            output_dir,
            tokenizer,
            metrics.get(task),
        )
        for task, dump_dir in args.task
    ]
    manifest = {
        "title": "Ranked Key-Token Probability",
        "method": args.method,
        "selection_mode": args.selection_mode,
        "target_top_p": args.target_top_p,
        "target_top_k": args.target_top_k,
        "first_sparse_layer": args.first_sparse_layer,
        "query_tile_size": 128,
        "probe_labels": ["one_third", "two_thirds", "last", "mean"],
        "probe_weights": [0.2, 0.3, 0.4, 0.1],
        "tasks": tasks,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "manifest.js").write_text(
        "window.__rankedProbabilityManifest = "
        + json.dumps(manifest, ensure_ascii=False, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
