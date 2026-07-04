#!/usr/bin/env python3
"""Export full and token-truncated prompts for the A/B similarity experiment samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

from attention_map.model import load_model_and_tokenizer, tokenize
from attention_map.run_longbench_v2 import build_longbench_v2_prompt

SAMPLES = [
    {
        "label": "A_code_repository",
        "sample_id": "66fa208bbb02136c067c5fc1",
        "line_index": 7,
        "out_sample_dir": (
            "outputs_longbench_v2/code_repository_understanding/code_repo_qa/"
            "sample_66fa208bbb02136c067c5fc1_line0007"
        ),
    },
    {
        "label": "B_single_document_qa",
        "sample_id": "66f36490821e116aacb2cc22",
        "line_index": 1,
        "out_sample_dir": (
            "outputs_longbench_v2/single_document_qa/financial/"
            "sample_66f36490821e116aacb2cc22_line0001"
        ),
    },
]


def load_row(jsonl_path: Path, sample_id: str, line_index: int) -> Dict[str, Any]:
    with jsonl_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == line_index:
                row = json.loads(line)
                if str(row.get("_id")) != sample_id:
                    raise ValueError(
                        f"line {line_index}: expected _id={sample_id}, got {row.get('_id')}"
                    )
                return row
    raise KeyError(f"line_index {line_index} not found in {jsonl_path}")


def export_one(
    *,
    row: Dict[str, Any],
    label: str,
    export_dir: Path,
    sample_dir: Path | None,
    model_path: str,
    max_input_tokens: int | None,
    device: str,
) -> Tuple[Path, Path, int]:
    prompt = build_longbench_v2_prompt(row)
    export_dir.mkdir(parents=True, exist_ok=True)

    full_name = f"{label}_prompt_full.txt"
    if max_input_tokens is None:
        actual_name = f"{label}_prompt_actual.txt"
    else:
        actual_name = f"{label}_prompt_truncated_{max_input_tokens}tok.txt"
    full_path = export_dir / full_name
    trunc_path = export_dir / actual_name

    full_path.write_text(prompt, encoding="utf-8")

    _, tokenizer = load_model_and_tokenizer(model_path, device=device)
    enc = tokenize(tokenizer, prompt, max_length=max_input_tokens, device="cpu")
    truncated = tokenizer.decode(enc["input_ids"][0], skip_special_tokens=False)
    trunc_path.write_text(truncated, encoding="utf-8")

    meta = {
        "label": label,
        "sample_id": row.get("_id"),
        "domain": row.get("domain"),
        "sub_domain": row.get("sub_domain"),
        "question": row.get("question"),
        "answer": row.get("answer"),
        "full_chars": len(prompt),
        "truncated_tokens": int(enc["input_ids"].shape[1]),
        "max_input_tokens": max_input_tokens,
        "full_path": str(full_path),
        "truncated_path": str(trunc_path),
    }
    (export_dir / f"{label}_prompt_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if sample_dir is not None:
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / "input_prompt_full.txt").write_text(prompt, encoding="utf-8")
        (sample_dir / f"input_prompt_truncated_{max_input_tokens}tok.txt").write_text(
            truncated, encoding="utf-8"
        )

    return full_path, trunc_path, int(enc["input_ids"].shape[1])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--sample_a_id",
        default=None,
        help="Override sample A _id (requires matching jsonl line)",
    )
    p.add_argument("--sample_a_line", type=int, default=0)
    p.add_argument("--sample_b_id", default=None)
    p.add_argument("--sample_b_line", type=int, default=1)
    p.add_argument(
        "--jsonl_path",
        default="/home/ubuntu/work/datasets/longbench_v2/longbench_v2_train.jsonl",
    )
    p.add_argument(
        "--export_dir",
        default="/home/ubuntu/work/attention_map/analysis/ab_similarity_experiment_k95/inputs",
    )
    p.add_argument(
        "--repo_root",
        default="/home/ubuntu/work/attention_map",
    )
    p.add_argument(
        "--model_path",
        default="/home/ubuntu/work/model/Qwen2.5-3B",
    )
    p.add_argument(
        "--max_input_tokens",
        type=int,
        default=None,
        nargs="?",
        const=None,
        help="If set, also export tokenizer-truncated prompt; default None (full only)",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--copy_to_sample_dirs",
        action="store_true",
        default=True,
    )
    args = p.parse_args()

    jsonl_path = Path(args.jsonl_path)
    export_dir = Path(args.export_dir)
    repo_root = Path(args.repo_root)

    if args.sample_a_id and args.sample_b_id:
        samples = [
            {
                "label": "A",
                "sample_id": args.sample_a_id,
                "line_index": args.sample_a_line,
                "out_sample_dir": None,
            },
            {
                "label": "B",
                "sample_id": args.sample_b_id,
                "line_index": args.sample_b_line,
                "out_sample_dir": None,
            },
        ]
    else:
        samples = SAMPLES

    index = []
    for spec in samples:
        row = load_row(jsonl_path, spec["sample_id"], spec["line_index"])
        out_rel = spec.get("out_sample_dir")
        sample_dir = (
            repo_root / out_rel
            if args.copy_to_sample_dirs and out_rel
            else None
        )
        full_path, trunc_path, n_tok = export_one(
            row=row,
            label=spec["label"],
            export_dir=export_dir,
            sample_dir=sample_dir,
            model_path=args.model_path,
            max_input_tokens=args.max_input_tokens,
            device=args.device,
        )
        index.append(
            {
                "label": spec["label"],
                "sample_id": spec["sample_id"],
                "line_index": spec["line_index"],
                "full_prompt": str(full_path),
                "truncated_prompt": str(trunc_path),
                "truncated_tokens": n_tok,
            }
        )
        print(f"[{spec['label']}] full -> {full_path} ({full_path.stat().st_size} bytes)")
        print(f"[{spec['label']}] truncated ({n_tok} tok) -> {trunc_path}")

    (export_dir / "README.json").write_text(
        json.dumps(
            {
                "description": "A/B similarity experiment input prompts",
                "max_input_tokens": args.max_input_tokens,
                "model_path": args.model_path,
                "samples": index,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Wrote index: {export_dir / 'README.json'}")


if __name__ == "__main__":
    main()
