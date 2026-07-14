#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import torch

REPO = Path("/home/ubuntu/work")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.official_sparse_baselines import run_official_sparse_baseline as official
from experiments import run_shared_layer_mask_experiment as unified


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def generate_official(model, tok, prompt: str, max_input_length: int, max_new_tokens: int) -> Dict[str, Any]:
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
    input_len = int(enc["input_ids"].shape[1])
    enc = {k: v.to("cuda:0") for k, v in enc.items()}
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    text = tok.decode(out[0][input_len:], skip_special_tokens=True)
    return {
        "text": text,
        "answer": official.extract_mcq_answer(text),
        "input_tokens": input_len,
        "latency_sec": time.time() - t0,
    }


def generate_unified(model, tok, prompt: str, max_input_length: int, max_new_tokens: int, prefill_chunk_size: int) -> Dict[str, Any]:
    input_ids, attention_mask = unified.tokenize_prompt(
        tok, prompt, SimpleNamespace(max_input_length=max_input_length), "cuda"
    )
    t0 = time.time()
    gen = unified.generate_new_tokens(
        model,
        tok,
        input_ids,
        attention_mask,
        controller=None,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        analysis_seq_len=int(input_ids.shape[1]),
        prefill_chunk_size=prefill_chunk_size,
    )
    text = gen["generated_text"]
    return {
        "text": text,
        "answer": unified.extract_mcq_answer(text),
        "input_tokens": int(input_ids.shape[1]),
        "latency_sec": time.time() - t0,
        "new_token_ids": gen.get("new_token_ids", []),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["minference", "flexprefill"])
    ap.add_argument("--model", default="/home/ubuntu/work/model/Qwen2.5-7B")
    ap.add_argument("--data", default="/home/ubuntu/work/experiments/data/longbench_v2_32k_full_7b.jsonl")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_samples", type=int, default=20)
    ap.add_argument("--max_input_length", type=int, default=32768)
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--prefill_chunk_size", type=int, default=2048)
    args = ap.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.manual_seed(42)
    torch.cuda.empty_cache()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "stage": "baseline_faithfulness_stage1",
        "method": args.method,
        "model": args.model,
        "data": args.data,
        "num_samples": args.num_samples,
        "max_input_length": args.max_input_length,
        "max_new_tokens": args.max_new_tokens,
        "prefill_chunk_size": args.prefill_chunk_size,
        "official_path": "experiments.official_sparse_baselines.run_official_sparse_baseline",
        "unified_path": "experiments.run_shared_layer_mask_experiment",
        "comparison": "same official-patched model, official model.generate vs unified chunked generate_new_tokens",
    }
    write_json(out_dir / "manifest.json", manifest)

    tok, model = official.load_model(args.method, args.model, args.max_input_length)
    samples = official.read_eval_samples(Path(args.data), args.num_samples)

    records: List[Dict[str, Any]] = []
    counts = {
        "prompt_equal": 0,
        "input_tokens_equal": 0,
        "answer_equal": 0,
        "text_equal": 0,
        "official_correct": 0,
        "unified_correct": 0,
    }

    for idx, sample in enumerate(samples, 1):
        sid = sample.get("_id", str(idx))
        gold = str(sample.get("answer", "")).strip().upper() or None
        prompt_off = official.build_prompt(sample)
        prompt_uni = unified.build_prompt(sample)
        prompt_equal = prompt_off == prompt_uni
        if prompt_equal:
            counts["prompt_equal"] += 1

        # Use unified prompt for both generation paths if prompts are equal; otherwise keep both for diagnosis.
        off = generate_official(model, tok, prompt_off, args.max_input_length, args.max_new_tokens)
        torch.cuda.empty_cache()
        uni = generate_unified(model, tok, prompt_uni, args.max_input_length, args.max_new_tokens, args.prefill_chunk_size)
        torch.cuda.empty_cache()

        if off["input_tokens"] == uni["input_tokens"]:
            counts["input_tokens_equal"] += 1
        if off["answer"] == uni["answer"]:
            counts["answer_equal"] += 1
        if off["text"] == uni["text"]:
            counts["text_equal"] += 1
        if off["answer"] == gold:
            counts["official_correct"] += 1
        if uni["answer"] == gold:
            counts["unified_correct"] += 1

        rec = {
            "index": idx,
            "id": sid,
            "gold": gold,
            "prompt_equal": prompt_equal,
            "prompt_hash_official": sha(prompt_off),
            "prompt_hash_unified": sha(prompt_uni),
            "official": off,
            "unified": uni,
            "answer_equal": off["answer"] == uni["answer"],
            "text_equal": off["text"] == uni["text"],
            "official_correct": off["answer"] == gold,
            "unified_correct": uni["answer"] == gold,
        }
        records.append(rec)
        write_json(out_dir / "samples" / f"sample_{idx:03d}.json", rec)
        print(
            f"{args.method} {idx:03d}/{len(samples):03d} "
            f"gold={gold} official={off['answer']} unified={uni['answer']} "
            f"equal={rec['answer_equal']}",
            flush=True,
        )

    n = max(len(records), 1)
    summary = {
        "method": args.method,
        "count": len(records),
        "prompt_equal_rate": counts["prompt_equal"] / n,
        "input_tokens_equal_rate": counts["input_tokens_equal"] / n,
        "answer_equal_rate": counts["answer_equal"] / n,
        "text_equal_rate": counts["text_equal"] / n,
        "official_accuracy": counts["official_correct"] / n,
        "unified_accuracy": counts["unified_correct"] / n,
        "official_correct": counts["official_correct"],
        "unified_correct": counts["unified_correct"],
        "avg_official_latency_sec": sum(r["official"]["latency_sec"] for r in records) / n,
        "avg_unified_latency_sec": sum(r["unified"]["latency_sec"] for r in records) / n,
        "pass_criteria_note": "Prompt/input should match exactly. If answer_equal_rate is low, unified generation changes baseline behavior and should not be used without adapter fixes.",
    }
    write_json(out_dir / "records.json", records)
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
