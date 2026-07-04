#!/usr/bin/env python3
"""CLI: run Qwen2.5-3B on RULER data and save attention maps by (layer, head)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from attention_map.data import load_samples
from attention_map.model import (
    build_input_text,
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Record attention maps (layer_i, head_j) for RULER + Qwen2.5-3B"
    )
    p.add_argument(
        "--model_path",
        type=str,
        default="/home/ubuntu/work/model/Qwen2.5-3B",
    )
    p.add_argument(
        "--data_root",
        type=str,
        default="/home/ubuntu/work/ruler_data",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="/home/ubuntu/work/attention_map/outputs",
    )
    p.add_argument("--split", type=str, default=None, help="e.g. 4k or 8k")
    p.add_argument("--task", type=str, default=None, help="e.g. niah_single_1")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument(
        "--start_line",
        type=int,
        default=0,
        help="Skip first N lines in validation.jsonl before loading (0-based)",
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
        help="all: full map [q,k]; last: last query row only (recommended for 4k/8k)",
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


def sample_output_dir(
    base: Path, split: str, task: str, sample_id: int, line_index: int
) -> Path:
    return base / split / task / f"sample_{sample_id:06d}_line{line_index:04d}"


def infer_split_task(data_root: Path, source_file: str) -> tuple[str, str]:
    rel = Path(source_file).relative_to(data_root)
    parts = rel.parts
    split_name = parts[0] if parts else "unknown"
    task_name = parts[1] if len(parts) > 1 else "unknown"
    return split_name, task_name


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

    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)

    logger.info("Loading model from %s", args.model_path)
    model, tokenizer = load_model_and_tokenizer(
        args.model_path,
        device=args.device,
        torch_dtype=args.dtype,
        attn_implementation="eager",
    )

    samples = load_samples(
        data_root,
        task=args.task,
        split=args.split,
        max_samples=args.max_samples,
        start_line=args.start_line,
    )
    logger.info("Loaded %d samples", len(samples))

    run_config = {
        "model_path": args.model_path,
        "data_root": str(data_root),
        "query_slice": args.query_slice,
        "max_input_tokens": args.max_input_tokens,
        "use_hooks": use_hooks,
    }
    with (output_base / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    for idx, sample in enumerate(samples):
        split_name, task_name = infer_split_task(data_root, sample.source_file)
        out_dir = sample_output_dir(
            output_base, split_name, task_name, sample.sample_id, sample.line_index
        )
        text = build_input_text(sample.input_text)
        logger.info(
            "[%d/%d] sample_id=%s -> %s (chars=%d)",
            idx + 1,
            len(samples),
            sample.sample_id,
            out_dir,
            len(text),
        )

        q_index = args.query_index
        if q_index is None and args.query_slice == "index":
            q_index = sample.token_position_answer

        meta = run_sample(
            model,
            tokenizer,
            text,
            out_dir,
            args.device,
            args.max_input_tokens,
            args.query_slice,
            q_index,
            use_hooks,
        )
        meta.update(
            {
                "sample_id": sample.sample_id,
                "line_index": sample.line_index,
                "source_file": sample.source_file,
                "outputs": sample.outputs,
            }
        )
        with (out_dir / "sample_meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        if args.device == "cuda":
            torch.cuda.empty_cache()

    logger.info("Done. Outputs under %s", output_base)


if __name__ == "__main__":
    main()
