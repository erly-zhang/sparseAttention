#!/usr/bin/env python3
"""Run supported sparse-attention baselines through their official runtime code.

This runner is intentionally conservative: if a baseline repository does not
support Qwen2/Qwen2.5 directly, the script records an explanatory manifest
instead of substituting a local proxy implementation. That keeps official
baseline numbers separate from the unified-runner ablations.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


def patch_transformers_flash_window_flag() -> None:
    """Handle transformers versions that do not expose the FA2 window flag."""
    try:
        import transformers.modeling_flash_attention_utils as flash_utils

        if not hasattr(flash_utils, "_flash_supports_window_size"):
            flash_utils._flash_supports_window_size = False
    except Exception:
        pass


def patch_minference_dense_decode_fallback() -> None:
    """Map MInference's flash-attention call to SDPA for this environment."""
    import torch.nn.functional as F
    import minference.modules.forward as mf_forward

    def sdpa_flash_compatible(
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        position_ids=None,
        dropout=0.0,
        sliding_window=None,
        is_causal=True,
        **kwargs,
    ):
        # MInference passes [batch, seq, heads, dim] here. Return the same layout.
        q = query_states.transpose(1, 2)
        k = key_states.transpose(1, 2)
        v = value_states.transpose(1, 2)
        attn_mask = attention_mask
        causal = bool(is_causal and attn_mask is None and q.shape[-2] > 1)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout,
            is_causal=causal,
        )
        return out.transpose(1, 2)

    mf_forward._flash_attention_forward = sdpa_flash_compatible

LOG = logging.getLogger("official_sparse_baseline")

SUPPORTED = {"minference", "flexprefill"}
UNSUPPORTED: Dict[str, str] = {
    "moa": "Official MoA code only wires Llama/Vicuna/LongChat model names in MoA.models.interface.update_model_function; Qwen2/Qwen2.5 is not implemented. Official Qwen use would require adding a Qwen attention integration and generating a MoA config/search plan.",
    "duoattention": "Official DuoAttention patch dispatch supports llama and mistral/mixtral model_type only; Qwen2/Qwen2.5 raises ValueError. Official Qwen use would require implementing Qwen patch plus training retrieval-head patterns.",
    "dam": "Official DAM implementation defines DamLlamaForCausalLM/DamLlamaAttention and scripts hard-code Llama-3.2 models; Qwen2/Qwen2.5 is not implemented. Official Qwen use would require a Qwen DAM model wrapper plus generated masks.",
    "survey_heads": "No standalone official sparse-attention pipeline was identified for survey_heads; the prior version was a local heuristic and is not an official baseline.",
}


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


def load_model(method: str, model_path: str, max_input_length: int):
    patch_transformers_flash_window_flag()
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    common = dict(torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True)
    if method == "minference":
        from minference import MInference
        patch_minference_dense_decode_fallback()
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map={"": "cuda:0"}, **common)
        model = MInference("minference", "Qwen/Qwen2.5-7B-Instruct")(model)
    elif method == "flexprefill":
        from flex_prefill import patch_model
        model = AutoModelForCausalLM.from_pretrained(
            model_path, _attn_implementation="flash_attention_2", **common
        ).cuda()
        cfg = {
            "block_size": 128,
            "flex_prefill_gamma": 0.9,
            "flex_prefill_tau": 0.1,
            "flex_prefill_min_budget": 512,
            "flex_prefill_max_budget": None,
        }
        patch_model(model, "flex_prefill", cfg)
    else:
        raise ValueError(method)
    model.eval()
    return tok, model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--model", default="/home/ubuntu/work/model/Qwen2.5-7B")
    ap.add_argument("--data", default="/home/ubuntu/work/experiments/data/longbench_v2_32k_full_7b.jsonl")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--eval_num_samples", type=int, default=500)
    ap.add_argument("--max_input_length", type=int, default=32768)
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "baseline_id": args.method,
        "implementation": "official_repository",
        "model": args.model,
        "data": args.data,
        "eval_num_samples": args.eval_num_samples,
        "max_input_length": args.max_input_length,
        "max_new_tokens": args.max_new_tokens,
        "uses_representative_head_selection": False,
        "uses_single_cluster_code": False,
    }
    if args.method in UNSUPPORTED:
        manifest.update({"status": "not_run_official_qwen_unsupported", "reason": UNSUPPORTED[args.method]})
        write_json(out_dir / "manifest.json", manifest)
        write_json(out_dir / "summary.json", {"baseline_id": args.method, "status": manifest["status"], "reason": manifest["reason"], "count": 0})
        LOG.warning("%s not run: %s", args.method, manifest["reason"])
        return
    if args.method not in SUPPORTED:
        raise SystemExit(f"Unknown method: {args.method}")

    manifest["status"] = "running"
    write_json(out_dir / "manifest.json", manifest)
    tok, model = load_model(args.method, args.model, args.max_input_length)
    samples = read_eval_samples(Path(args.data), args.eval_num_samples)
    LOG.info("Loaded %d eval samples", len(samples))
    results = []
    correct = 0
    for idx, sample in enumerate(samples, 1):
        sid = sample.get("_id", str(idx))
        prompt = build_prompt(sample)
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=args.max_input_length)
        input_tokens = int(enc["input_ids"].shape[1])
        enc = {k: v.to("cuda:0") for k, v in enc.items()}
        t0 = time.time()
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = extract_mcq_answer(gen)
        gold = str(sample.get("answer", "")).strip().upper() or None
        ok = pred == gold if gold else None
        if ok:
            correct += 1
        rec = {
            "index": idx,
            "id": sid,
            "domain": sample.get("domain"),
            "sub_domain": sample.get("sub_domain"),
            "gold_answer": gold,
            "pred_answer": pred,
            "correct": ok,
            "generation": gen,
            "input_tokens": input_tokens,
            "latency_sec": time.time() - t0,
        }
        results.append(rec)
        write_json(out_dir / "task_eval" / f"sample_{idx:03d}_{sid}" / "result.json", rec)
        if idx % 1 == 0:
            LOG.info("%s eval %03d/%03d id=%s pred=%s gold=%s correct=%s acc=%.4f", args.method, idx, len(samples), sid, pred, gold, ok, correct / idx)
        torch.cuda.empty_cache()
    summary = {
        "baseline_id": args.method,
        "implementation": "official_repository",
        "status": "completed",
        "count": len(results),
        "correct": correct,
        "accuracy": correct / max(len(results), 1),
        "avg_latency_sec": sum(r["latency_sec"] for r in results) / max(len(results), 1),
        "avg_input_tokens": sum(r["input_tokens"] for r in results) / max(len(results), 1),
    }
    manifest["status"] = "completed"
    write_json(out_dir / "manifest.json", manifest)
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "results.json", results)
    LOG.info("DONE %s accuracy=%.4f", args.method, summary["accuracy"])

if __name__ == "__main__":
    main()
