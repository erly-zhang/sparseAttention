#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import json
import os
import re
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


def patch_transformers_flash_window_flag() -> None:
    try:
        import transformers.modeling_flash_attention_utils as flash_utils
        if not hasattr(flash_utils, "_flash_supports_window_size"):
            flash_utils._flash_supports_window_size = False
    except Exception:
        pass


def patch_minference_dense_decode_fallback() -> None:
    import torch.nn.functional as F
    import minference.modules.forward as mf_forward

    def sdpa_flash_compatible(query_states, key_states, value_states, attention_mask, query_length, position_ids=None, dropout=0.0, sliding_window=None, is_causal=True, **kwargs):
        q = query_states.transpose(1, 2)
        k = key_states.transpose(1, 2)
        v = value_states.transpose(1, 2)
        causal = bool(is_causal and attention_mask is None and q.shape[-2] > 1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, dropout_p=dropout, is_causal=causal)
        return out.transpose(1, 2)

    mf_forward._flash_attention_forward = sdpa_flash_compatible


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


def extract_mcq_answer(text: str) -> Optional[str]:
    upper = (text or "").upper().strip()
    for pattern in [
        r"(?:ANSWER|CHOICE|OPTION)\s*[:：]?\s*([ABCD])\b",
        r"(?:THE\s+)?(?:CORRECT\s+)?(?:ANSWER|CHOICE|OPTION)\s+IS\s+([ABCD])\b",
        r"^\s*([ABCD])\b",
        r"\b([ABCD])\b",
    ]:
        m = re.search(pattern, upper)
        if m:
            return m.group(1)
    return None


def read_eval_samples(path: Path, limit: int) -> List[Dict[str, Any]]:
    rows = []
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
    return float(ys[min(len(ys) - 1, max(0, int(round((len(ys) - 1) * q))))])


def summarize(xs: List[float]) -> Dict[str, Optional[float]]:
    return {
        "mean": mean(xs),
        "median": median(xs),
        "min": float(min(xs)) if xs else None,
        "max": float(max(xs)) if xs else None,
        "p10": pct(xs, 0.10),
        "p90": pct(xs, 0.90),
    }


def count_vertical_slash_pairs(v_idx: torch.Tensor, s_idx: torch.Tensor, context_size: int) -> Dict[str, float]:
    v = sorted(set(int(x) for x in v_idx.detach().flatten().cpu().tolist() if 0 <= int(x) < context_size))
    dists = sorted(set(int(x) for x in s_idx.detach().flatten().cpu().tolist() if 0 <= int(x) < context_size))
    vertical_pairs = sum(context_size - c for c in v)
    slash_pairs = sum(context_size - d for d in dists)
    overlap_pairs = 0
    for d in dists:
        overlap_pairs += bisect.bisect_right(v, context_size - 1 - d)
    selected_pairs = vertical_pairs + slash_pairs - overlap_pairs
    causal_pairs = context_size * (context_size + 1) // 2
    return {
        "vertical_unique_tokens": len(v),
        "slash_unique_diagonals": len(dists),
        "vertical_pairs": float(vertical_pairs),
        "slash_pairs": float(slash_pairs),
        "overlap_pairs": float(overlap_pairs),
        "selected_pairs": float(selected_pairs),
        "causal_pairs": float(causal_pairs),
        "pair_ratio": float(selected_pairs / causal_pairs) if causal_pairs else None,
        "mean_selected_keys_per_query": float(selected_pairs / context_size) if context_size else None,
        "mean_selected_keys_per_query_ratio_to_len": float((selected_pairs / context_size) / context_size) if context_size else None,
    }


class RuntimeRecorder:
    def __init__(self):
        self.current_sample_index: Optional[int] = None
        self.current_sample_id: Optional[str] = None
        self.events: List[Dict[str, Any]] = []

    def clear(self, idx: int, sid: str) -> None:
        self.current_sample_index = idx
        self.current_sample_id = sid
        self.events.clear()

    def record(self, query: torch.Tensor, v_idx: torch.Tensor, s_idx: torch.Tensor, block_size_M: int, block_size_N: int) -> None:
        batch_size, num_heads, context_size, head_dim = query.shape
        if context_size <= 1:
            return
        for b in range(batch_size):
            for h in range(num_heads):
                stats = count_vertical_slash_pairs(v_idx[b:b + 1, h:h + 1], s_idx[b:b + 1, h:h + 1], int(context_size))
                stats.update({
                    "sample_index": self.current_sample_index,
                    "sample_id": self.current_sample_id,
                    "batch": b,
                    "call_index": len(self.events),
                    "head": h,
                    "context_size": int(context_size),
                    "head_dim": int(head_dim),
                    "block_size_M": int(block_size_M),
                    "block_size_N": int(block_size_N),
                })
                self.events.append(stats)


def install_runtime_hook(recorder: RuntimeRecorder):
    import minference.modules.minference_forward as mf
    original = mf.vertical_slash_sparse_attention

    def wrapped(query, key, value, v_idx, s_idx, block_size_M=64, block_size_N=64):
        recorder.record(query, v_idx, s_idx, block_size_M, block_size_N)
        return original(query, key, value, v_idx, s_idx, block_size_M, block_size_N)

    mf.vertical_slash_sparse_attention = wrapped
    return original


def load_model(model_path: str):
    patch_transformers_flash_window_flag()
    from minference import MInference
    patch_minference_dense_decode_fallback()
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map={"": "cuda:0"},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model = MInference("minference", "Qwen/Qwen2.5-7B-Instruct")(model)
    model.eval()
    return tok, model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/ubuntu/work/model/Qwen2.5-7B")
    ap.add_argument("--data", default="/home/ubuntu/work/experiments/data/longbench_v2_32k_full_7b.jsonl")
    ap.add_argument("--output_dir", default="/home/ubuntu/work/experiments/outputs/official_sparse_baselines_500eval_minference_runtime_sparse_stats")
    ap.add_argument("--eval_num_samples", type=int, default=500)
    ap.add_argument("--max_input_length", type=int, default=32768)
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--save_layer_events", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = vars(args)
    manifest.update({"method": "minference", "stat_type": "runtime_vertical_slash_union"})
    write_json(out_dir / "manifest.json", manifest)
    for name in ["per_sample_stats.jsonl", "layer_head_events.jsonl"]:
        p = out_dir / name
        if p.exists():
            p.unlink()

    recorder = RuntimeRecorder()
    tok, model = load_model(args.model)
    install_runtime_hook(recorder)
    samples = read_eval_samples(Path(args.data), args.eval_num_samples)
    results = []
    correct = 0
    all_pair_ratios: List[float] = []
    all_token_ratios: List[float] = []
    all_mean_keys: List[float] = []
    t_start = time.time()

    for idx, sample in enumerate(samples, 1):
        sid = str(sample.get("_id", idx))
        prompt = build_prompt(sample)
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=args.max_input_length)
        input_tokens = int(enc["input_ids"].shape[1])
        enc = {k: v.to("cuda:0") for k, v in enc.items()}
        recorder.clear(idx, sid)
        t0 = time.time()
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize()
        gen = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = extract_mcq_answer(gen)
        gold = str(sample.get("answer", "")).strip().upper() or None
        ok = pred == gold if gold else None
        if ok:
            correct += 1
        events = list(recorder.events)
        pair_ratios = [float(e["pair_ratio"]) for e in events if e.get("pair_ratio") is not None]
        token_ratios = [float(e["mean_selected_keys_per_query_ratio_to_len"]) for e in events if e.get("mean_selected_keys_per_query_ratio_to_len") is not None]
        mean_keys = [float(e["mean_selected_keys_per_query"]) for e in events if e.get("mean_selected_keys_per_query") is not None]
        all_pair_ratios.extend(pair_ratios)
        all_token_ratios.extend(token_ratios)
        all_mean_keys.extend(mean_keys)
        rec = {
            "index": idx,
            "id": sid,
            "input_tokens": input_tokens,
            "prefill_head_events": len(events),
            "expected_layer_head_events": 28 * 28,
            "mean_pair_ratio": mean(pair_ratios),
            "median_pair_ratio": median(pair_ratios),
            "min_pair_ratio": float(min(pair_ratios)) if pair_ratios else None,
            "max_pair_ratio": float(max(pair_ratios)) if pair_ratios else None,
            "mean_selected_keys_per_query": mean(mean_keys),
            "mean_selected_keys_per_query_ratio_to_len": mean(token_ratios),
            "pred_answer": pred,
            "gold_answer": gold,
            "correct": ok,
            "generation": gen,
            "latency_sec": time.time() - t0,
        }
        results.append(rec)
        append_jsonl(out_dir / "per_sample_stats.jsonl", rec)
        if args.save_layer_events:
            for event in events:
                append_jsonl(out_dir / "layer_head_events.jsonl", event)
        if idx % args.log_every == 0 or idx == 1:
            elapsed = time.time() - t_start
            print(f"minference runtime stats {idx:03d}/{len(samples):03d} input={input_tokens} events={len(events)} mean_pair_ratio={rec['mean_pair_ratio']} acc={correct/idx:.4f} elapsed={elapsed:.1f}s", flush=True)
        torch.cuda.empty_cache()

    summary = {
        "method": "minference",
        "stat_type": "runtime_vertical_slash_union",
        "note": "Runtime hook on official MInference vertical_slash_sparse_attention. v_idx fixed columns and s_idx slash diagonals are union-counted under causal masking. pair_ratio = selected query-key pairs / full causal query-key pairs; mean_selected_keys_per_query_ratio_to_len is average selected key tokens per query divided by sequence length.",
        "count": len(results),
        "correct": correct,
        "accuracy": correct / max(len(results), 1),
        "input_tokens": summarize([float(r["input_tokens"]) for r in results]),
        "prefill_head_events": summarize([float(r["prefill_head_events"]) for r in results]),
        "per_head_pair_ratio": summarize(all_pair_ratios),
        "per_sample_mean_pair_ratio": summarize([float(r["mean_pair_ratio"]) for r in results if r["mean_pair_ratio"] is not None]),
        "per_head_mean_selected_keys_per_query": summarize(all_mean_keys),
        "per_sample_mean_selected_keys_per_query_ratio_to_len": summarize([float(r["mean_selected_keys_per_query_ratio_to_len"]) for r in results if r["mean_selected_keys_per_query_ratio_to_len"] is not None]),
        "avg_latency_sec": mean([float(r["latency_sec"]) for r in results]),
    }
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "results.json", results)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
