#!/usr/bin/env python3
"""CLI: run Qwen2.5-3B on LongBench-v2 jsonl and save attention maps."""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from attention_map.model import (
    extract_last_token_attention,
    forward_prefill,
    load_model_and_tokenizer,
    tokenize,
)
from attention_map.recorder import AttentionHookRecorder, record_via_output_attentions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class LongBenchV2Sample:
    sample_id: str
    input_text: str
    outputs: List[str]
    source_file: str
    line_index: int
    domain: str
    sub_domain: str
    metadata: Dict[str, Any]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Record attention maps (layer_i, head_j) for LongBench-v2 + Qwen2.5-3B"
    )
    p.add_argument(
        "--model_path",
        type=str,
        default="/home/ubuntu/work/model/Qwen2.5-3B",
    )
    p.add_argument(
        "--jsonl_path",
        type=str,
        default="/home/ubuntu/work/datasets/longbench_v2/longbench_v2_train.jsonl",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="/home/ubuntu/work/attention_map/outputs_longbench_v2",
    )
    p.add_argument("--domain", type=str, default=None)
    p.add_argument("--sub_domain", type=str, default=None)
    p.add_argument("--difficulty", type=str, default=None)
    p.add_argument("--length", type=str, default=None)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument(
        "--start_line",
        type=int,
        default=0,
        help="Skip first N lines in the jsonl before loading (0-based)",
    )
    p.add_argument(
        "--max_input_tokens",
        type=int,
        default=None,
        help="Truncate input; set e.g. 512 for smoke test",
    )
    p.add_argument(
        "--query_slice",
        type=str,
        choices=["all", "last", "index"],
        default="last",
        help="all: full map [q,k]; last: last query row only (recommended)",
    )
    p.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help=(
            "Chunked-prefill size for query_slice=last (memory-efficient long "
            "context). E.g. 4096. None = single prefill forward."
        ),
    )
    p.add_argument(
        "--query_index",
        type=int,
        default=None,
        help="Token index for query_slice=index",
    )
    p.add_argument(
        "--no_hooks",
        action="store_true",
        help="Use output_attentions tuple instead of per-layer hooks",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="bfloat16")
    return p.parse_args()


def normalize_path_component(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^0-9a-zA-Z]+", "_", value)
    return value.strip("_") or "unknown"


def format_sample_tag(sample_id: str) -> str:
    sample_str = sample_id.strip()
    if sample_str.isdigit():
        return f"sample_{int(sample_str):06d}"
    return f"sample_{normalize_path_component(sample_str)[:32]}"


def sample_output_dir(
    base: Path, domain: str, sub_domain: str, sample_id: str, line_index: int
) -> Path:
    return (
        base
        / normalize_path_component(domain)
        / normalize_path_component(sub_domain)
        / f"{format_sample_tag(sample_id)}_line{line_index:04d}"
    )


def build_longbench_v2_prompt(row: Dict[str, Any]) -> str:
    context = str(row.get("context", "")).strip()
    question = str(row.get("question", "")).strip()
    choice_a = str(row.get("choice_A", "")).strip()
    choice_b = str(row.get("choice_B", "")).strip()
    choice_c = str(row.get("choice_C", "")).strip()
    choice_d = str(row.get("choice_D", "")).strip()

    return f"""You are given a long context and a multiple-choice question.
Read the context carefully and choose the correct answer from A, B, C, or D.

Context:
{context}

Question:
{question}

Choices:
A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}

Please answer with only one letter: A, B, C, or D.
"""


def load_longbench_v2_samples(
    jsonl_path: Path,
    max_samples: Optional[int] = None,
    start_line: int = 0,
    domain: Optional[str] = None,
    sub_domain: Optional[str] = None,
    difficulty: Optional[str] = None,
    length: Optional[str] = None,
) -> List[LongBenchV2Sample]:
    if not jsonl_path.exists():
        raise FileNotFoundError(f"LongBench-v2 jsonl not found: {jsonl_path}")

    samples: List[LongBenchV2Sample] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if line_idx < start_line:
                continue
            if max_samples is not None and len(samples) >= max_samples:
                break

            row = json.loads(line)
            row_domain = str(row.get("domain", "unknown"))
            row_sub_domain = str(row.get("sub_domain", "unknown"))
            row_difficulty = str(row.get("difficulty", "unknown"))
            row_length = str(row.get("length", "unknown"))

            if domain is not None and row_domain != domain:
                continue
            if sub_domain is not None and row_sub_domain != sub_domain:
                continue
            if difficulty is not None and row_difficulty != difficulty:
                continue
            if length is not None and row_length != length:
                continue

            samples.append(
                LongBenchV2Sample(
                    sample_id=str(row.get("_id", line_idx)),
                    input_text=build_longbench_v2_prompt(row),
                    outputs=[str(row["answer"])] if "answer" in row else [],
                    source_file=str(jsonl_path),
                    line_index=line_idx,
                    domain=row_domain,
                    sub_domain=row_sub_domain,
                    metadata={
                        "_id": row.get("_id"),
                        "domain": row_domain,
                        "sub_domain": row_sub_domain,
                        "difficulty": row_difficulty,
                        "length": row_length,
                        "question": row.get("question"),
                        "choice_A": row.get("choice_A"),
                        "choice_B": row.get("choice_B"),
                        "choice_C": row.get("choice_C"),
                        "choice_D": row.get("choice_D"),
                        "answer": row.get("answer"),
                    },
                )
            )

    if not samples:
        raise ValueError(
            "No LongBench-v2 samples matched "
            f"jsonl_path={jsonl_path}, domain={domain}, sub_domain={sub_domain}, "
            f"difficulty={difficulty}, length={length}"
        )
    return samples


def save_last_token_attention(
    last_token_attn: "torch.Tensor",
    out_dir: Path,
) -> List[Dict[str, Any]]:
    """Persist [num_layers, num_heads, seq_len] as layer_XX/head_YY.npy.

    Each head file is saved as a [1, seq_len] float16 row, matching the
    query_slice='last' layout produced by the hook recorder so downstream
    tooling keeps working.
    """
    num_layers, num_heads, seq_len = last_token_attn.shape
    arr = last_token_attn.numpy()
    layers: List[Dict[str, Any]] = []
    for layer_idx in range(num_layers):
        layer_dir = out_dir / f"layer_{layer_idx:02d}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        heads: List[Dict[str, Any]] = []
        for head_idx in range(num_heads):
            row = arr[layer_idx, head_idx][None, :].astype(np.float16)  # [1, seq_len]
            rel = f"layer_{layer_idx:02d}/head_{head_idx:02d}.npy"
            np.save(layer_dir / f"head_{head_idx:02d}.npy", row)
            heads.append(
                {"layer": layer_idx, "head": head_idx, "path": rel, "shape": list(row.shape)}
            )
        layers.append(
            {"layer": layer_idx, "num_heads": num_heads, "q_len": 1, "k_len": seq_len, "heads": heads}
        )
    return layers


def run_sample(
    model,
    tokenizer,
    text: str,
    out_dir: Path,
    device: str,
    max_input_tokens: int | None,
    query_slice: str,
    query_index: int | None,
    use_hooks: bool,
    chunk_size: int | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    enc = tokenize(tokenizer, text, max_length=max_input_tokens, device=device)
    seq_len = enc["input_ids"].shape[1]

    meta = {
        "seq_len": seq_len,
        "query_slice": query_slice,
        "num_layers": model.config.num_hidden_layers,
        "num_attention_heads": model.config.num_attention_heads,
    }

    if query_slice == "last":
        # Memory-efficient two-stage extraction: never materializes the full
        # [num_heads, seq_len, seq_len] attention map.
        last_token_attn = extract_last_token_attention(
            model,
            enc["input_ids"],
            enc.get("attention_mask"),
            chunk_size=chunk_size,
        )
        layers = save_last_token_attention(last_token_attn, out_dir)
        meta["layers"] = layers
        meta["indexing"] = "(layer_i, head_j) -> layer_{i:02d}/head_{j:02d}.npy"
        meta["chunk_size"] = chunk_size
        meta["extraction"] = "two_stage_last_token"
        manifest_path = out_dir / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        meta["manifest"] = str(manifest_path)
        return meta

    if use_hooks:
        recorder = AttentionHookRecorder(
            model,
            out_dir,
            query_slice=query_slice,
            query_index=query_index,
        )
        recorder.register()
        try:
            forward_prefill(
                model,
                enc["input_ids"],
                enc.get("attention_mask"),
                output_attentions=True,
            )
        finally:
            recorder.remove()
        manifest_path = recorder.write_manifest(meta)
    else:
        outputs = forward_prefill(
            model,
            enc["input_ids"],
            enc.get("attention_mask"),
            output_attentions=True,
        )
        if outputs.attentions is None:
            raise RuntimeError(
                "attentions is None. Set attn_implementation='eager' and output_attentions=True."
            )
        layers = record_via_output_attentions(
            outputs.attentions,
            out_dir,
            query_slice=query_slice,
            query_index=query_index,
        )
        meta["layers"] = layers
        meta["indexing"] = "(layer_i, head_j) -> layer_{i:02d}/head_{j:02d}.npy"
        manifest_path = out_dir / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    meta["manifest"] = str(manifest_path)
    return meta


def main() -> None:
    args = parse_args()
    use_hooks = not args.no_hooks

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"

    jsonl_path = Path(args.jsonl_path)
    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model from %s", args.model_path)
    model, tokenizer = load_model_and_tokenizer(
        args.model_path,
        device=args.device,
        torch_dtype=args.dtype,
        attn_implementation="eager",
    )

    samples = load_longbench_v2_samples(
        jsonl_path,
        max_samples=args.max_samples,
        start_line=args.start_line,
        domain=args.domain,
        sub_domain=args.sub_domain,
        difficulty=args.difficulty,
        length=args.length,
    )
    logger.info("Loaded %d samples", len(samples))

    run_config = {
        "model_path": args.model_path,
        "jsonl_path": str(jsonl_path),
        "domain": args.domain,
        "sub_domain": args.sub_domain,
        "difficulty": args.difficulty,
        "length": args.length,
        "query_slice": args.query_slice,
        "query_index": args.query_index,
        "max_input_tokens": args.max_input_tokens,
        "chunk_size": args.chunk_size,
        "use_hooks": use_hooks,
    }
    with (output_base / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)

    for idx, sample in enumerate(samples):
        out_dir = sample_output_dir(
            output_base,
            sample.domain,
            sample.sub_domain,
            sample.sample_id,
            sample.line_index,
        )
        logger.info(
            "[%d/%d] sample_id=%s -> %s (chars=%d)",
            idx + 1,
            len(samples),
            sample.sample_id,
            out_dir,
            len(sample.input_text),
        )

        meta = run_sample(
            model,
            tokenizer,
            sample.input_text,
            out_dir,
            args.device,
            args.max_input_tokens,
            args.query_slice,
            args.query_index,
            use_hooks,
            chunk_size=args.chunk_size,
        )
        meta.update(
            {
                "sample_id": sample.sample_id,
                "line_index": sample.line_index,
                "source_file": sample.source_file,
                "outputs": sample.outputs,
                "domain": sample.domain,
                "sub_domain": sample.sub_domain,
                "sample_metadata": sample.metadata,
            }
        )
        with (out_dir / "sample_meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        if args.device == "cuda":
            torch.cuda.empty_cache()

    logger.info("Done. Outputs under %s", output_base)


if __name__ == "__main__":
    main()
