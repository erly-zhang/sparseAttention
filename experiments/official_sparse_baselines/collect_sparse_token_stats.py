#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


def build_prompt(sample: Dict[str, Any]) -> str:
    context = str(sample.get("context", "")).strip()
    question = str(sample.get("question", "")).strip()
    choice_a = str(sample.get("choice_A", "")).strip()
    choice_b = str(sample.get("choice_B", "")).strip()
    choice_c = str(sample.get("choice_C", "")).strip()
    choice_d = str(sample.get("choice_D", "")).strip()
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

Answer:
"""


def read_eval_samples(path: Path, limit: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("_sample_role") == "head_selection":
                continue
            rows.append(row)
            if len(rows) >= limit:
                break
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def append_jsonl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def mean(xs: List[float]) -> Optional[float]:
    return float(sum(xs) / len(xs)) if xs else None


def median(xs: List[float]) -> Optional[float]:
    return float(statistics.median(xs)) if xs else None


def pct(xs: List[float], q: float) -> Optional[float]:
    if not xs:
        return None
    ys = sorted(xs)
    idx = min(len(ys) - 1, max(0, int(round((len(ys) - 1) * q))))
    return float(ys[idx])


def summarize_values(xs: List[float]) -> Dict[str, Optional[float]]:
    return {
        "mean": mean(xs),
        "median": median(xs),
        "min": float(min(xs)) if xs else None,
        "max": float(max(xs)) if xs else None,
        "p10": pct(xs, 0.10),
        "p90": pct(xs, 0.90),
    }


def collect_minference(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir) / "minference"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Reuse the exact input lengths from the completed official 500-run when present.
    prior_results = Path(args.official_output_root) / "minference" / "results.json"
    if prior_results.exists():
        prior = json.loads(prior_results.read_text())
        input_lengths = [int(r["input_tokens"]) for r in prior[: args.eval_num_samples]]
        ids = [r.get("id", str(i + 1)) for i, r in enumerate(prior[: args.eval_num_samples])]
    else:
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        samples = read_eval_samples(Path(args.data), args.eval_num_samples)
        input_lengths = []
        ids = []
        for i, sample in enumerate(samples, 1):
            enc = tok(build_prompt(sample), return_tensors="pt", truncation=True, max_length=args.max_input_length)
            input_lengths.append(int(enc["input_ids"].shape[1]))
            ids.append(sample.get("_id", str(i)))

    from minference.configs.model2path import MODEL2PATH
    cfg_path = Path(MODEL2PATH["Qwen/Qwen2.5-7B-Instruct"])
    if not cfg_path.exists():
        cfg_path = Path("/home/ubuntu/work/MInference/minference/configs") / cfg_path.name
    cfg = json.loads(cfg_path.read_text())
    head_budgets: List[int] = []
    head_rows: List[Dict[str, Any]] = []
    for layer_idx, layer in enumerate(cfg):
        for head_s, item in layer.items():
            pattern = item[0]
            vertical = int(item[1]) if len(item) > 1 else 0
            slash = int(item[2]) if len(item) > 2 else 0
            budget = vertical + slash
            head_budgets.append(budget)
            head_rows.append({"layer": layer_idx, "head": int(head_s), "pattern": pattern, "vertical_size": vertical, "slash_size": slash, "budget_tokens_raw": budget})

    sample_rows = []
    ratios_all = []
    capped_ratios_all = []
    mean_budget = mean([float(x) for x in head_budgets]) or 0.0
    for i, (sid, length) in enumerate(zip(ids, input_lengths), 1):
        raw_ratios = [b / length for b in head_budgets]
        capped_ratios = [min(b, length) / length for b in head_budgets]
        ratios_all.extend(raw_ratios)
        capped_ratios_all.extend(capped_ratios)
        rec = {
            "index": i,
            "id": sid,
            "input_tokens": length,
            "mean_budget_tokens_per_head": mean_budget,
            "mean_budget_ratio_raw": mean(raw_ratios),
            "median_budget_ratio_raw": median(raw_ratios),
            "mean_budget_ratio_capped_by_input": mean(capped_ratios),
            "median_budget_ratio_capped_by_input": median(capped_ratios),
        }
        sample_rows.append(rec)
        append_jsonl(out_dir / "per_sample_stats.jsonl", rec)

    summary = {
        "method": "minference",
        "stat_type": "configuration_budget_ratio",
        "note": "MInference official Qwen2.5 config gives vertical_size and slash_size per layer/head. This summarizes (vertical_size + slash_size) / input_tokens over the 500 completed prompts; overlap and causal/query-position effects are not deducted.",
        "config_path": str(cfg_path),
        "count": len(sample_rows),
        "layers": len(cfg),
        "heads_total": len(head_budgets),
        "input_tokens": summarize_values([float(x) for x in input_lengths]),
        "budget_tokens_per_head_raw": summarize_values([float(x) for x in head_budgets]),
        "budget_ratio_raw_over_all_sample_heads": summarize_values(ratios_all),
        "budget_ratio_capped_by_input_over_all_sample_heads": summarize_values(capped_ratios_all),
        "per_sample_mean_budget_ratio_raw": summarize_values([float(r["mean_budget_ratio_raw"]) for r in sample_rows]),
        "per_sample_mean_budget_ratio_capped_by_input": summarize_values([float(r["mean_budget_ratio_capped_by_input"]) for r in sample_rows]),
    }
    write_json(out_dir / "head_budget_config.json", head_rows)
    write_json(out_dir / "summary.json", summary)


def collect_flexprefill(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir) / "flexprefill"
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from flex_prefill import patch_model
    import flex_prefill.modules.qwen2.flex_prefill_attention as qwen2_flex_mod

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        _attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).cuda()
    cfg = {
        "block_size": args.block_size,
        "flex_prefill_gamma": args.gamma,
        "flex_prefill_tau": args.tau,
        "flex_prefill_min_budget": args.min_budget,
        "flex_prefill_max_budget": None if args.max_budget < 0 else args.max_budget,
    }
    patch_model(model, "flex_prefill", cfg)
    model.eval()

    original_flex_attention = qwen2_flex_mod.flex_prefill_attention
    active_stats: List[Dict[str, Any]] = []

    def recording_flex_prefill_attention(q, k, v, *fa_args, **fa_kwargs):
        # Ask the official function for its own computational ratio and return only attn_out to the model.
        if q.shape[1] > 1:
            fa_kwargs = dict(fa_kwargs)
            fa_kwargs["return_computational_ratio"] = True
            out, ratio = original_flex_attention(q, k, v, *fa_args, **fa_kwargs)
            block_size = int(fa_kwargs.get("block_size", args.block_size))
            q_len = int(q.shape[1])
            num_blocks = int(math.ceil(q_len / block_size))
            active_stats.append({
                "q_len": q_len,
                "block_size": block_size,
                "num_blocks": num_blocks,
                "num_heads": int(q.shape[2]),
                "computational_ratio": float(ratio),
                "estimated_activated_block_pairs_all_heads": float(ratio) * num_blocks * num_blocks * int(q.shape[2]),
                "total_block_pairs_all_heads": num_blocks * num_blocks * int(q.shape[2]),
            })
            return out
        return original_flex_attention(q, k, v, *fa_args, **fa_kwargs)

    qwen2_flex_mod.flex_prefill_attention = recording_flex_prefill_attention
    samples = read_eval_samples(Path(args.data), args.eval_num_samples)
    if (out_dir / "per_sample_stats.jsonl").exists():
        (out_dir / "per_sample_stats.jsonl").unlink()
    sample_rows = []
    all_layer_ratios: List[float] = []
    t_start = time.time()
    for idx, sample in enumerate(samples, 1):
        sid = sample.get("_id", str(idx))
        enc = tok(build_prompt(sample), return_tensors="pt", truncation=True, max_length=args.max_input_length)
        input_tokens = int(enc["input_ids"].shape[1])
        enc = {k: v.to("cuda:0") for k, v in enc.items()}
        active_stats.clear()
        t0 = time.time()
        with torch.inference_mode():
            _ = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize()
        layer_stats = list(active_stats)
        ratios = [float(x["computational_ratio"]) for x in layer_stats]
        all_layer_ratios.extend(ratios)
        rec = {
            "index": idx,
            "id": sid,
            "input_tokens": input_tokens,
            "prefill_layers_recorded": len(layer_stats),
            "mean_computational_ratio": mean(ratios),
            "median_computational_ratio": median(ratios),
            "min_computational_ratio": float(min(ratios)) if ratios else None,
            "max_computational_ratio": float(max(ratios)) if ratios else None,
            "latency_sec": time.time() - t0,
            "layer_stats": layer_stats,
        }
        sample_rows.append(rec)
        append_jsonl(out_dir / "per_sample_stats.jsonl", rec)
        if idx % args.log_every == 0 or idx == 1:
            elapsed = time.time() - t_start
            print(f"flexprefill stats {idx:03d}/{len(samples):03d} input={input_tokens} layers={len(layer_stats)} mean_ratio={rec['mean_computational_ratio']} elapsed={elapsed:.1f}s", flush=True)
        torch.cuda.empty_cache()

    summary = {
        "method": "flexprefill",
        "stat_type": "official_dynamic_computational_ratio",
        "note": "Computed by calling the official FlexPrefill attention function with return_computational_ratio=True during the prefill pass. Ratio is activated block pairs / total causal block grid pairs across heads; block_size=128.",
        "count": len(sample_rows),
        "config": cfg,
        "max_new_tokens": args.max_new_tokens,
        "input_tokens": summarize_values([float(r["input_tokens"]) for r in sample_rows]),
        "prefill_layers_recorded": summarize_values([float(r["prefill_layers_recorded"]) for r in sample_rows]),
        "per_layer_computational_ratio": summarize_values(all_layer_ratios),
        "per_sample_mean_computational_ratio": summarize_values([float(r["mean_computational_ratio"]) for r in sample_rows if r["mean_computational_ratio"] is not None]),
        "avg_stats_latency_sec": mean([float(r["latency_sec"]) for r in sample_rows]),
    }
    write_json(out_dir / "summary.json", summary)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["minference", "flexprefill", "both"], required=True)
    ap.add_argument("--model", default="/home/ubuntu/work/model/Qwen2.5-7B")
    ap.add_argument("--data", default="/home/ubuntu/work/experiments/data/longbench_v2_32k_full_7b.jsonl")
    ap.add_argument("--official_output_root", default="/home/ubuntu/work/experiments/outputs/official_sparse_baselines_500eval")
    ap.add_argument("--output_dir", default="/home/ubuntu/work/experiments/outputs/official_sparse_baselines_500eval_sparse_stats")
    ap.add_argument("--eval_num_samples", type=int, default=500)
    ap.add_argument("--max_input_length", type=int, default=32768)
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--block_size", type=int, default=128)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--min_budget", type=int, default=512)
    ap.add_argument("--max_budget", type=int, default=-1)
    ap.add_argument("--log_every", type=int, default=5)
    args = ap.parse_args()
    set_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    write_json(Path(args.output_dir) / "manifest.json", vars(args))
    if args.method in {"minference", "both"}:
        collect_minference(args)
    if args.method in {"flexprefill", "both"}:
        collect_flexprefill(args)


if __name__ == "__main__":
    main()
