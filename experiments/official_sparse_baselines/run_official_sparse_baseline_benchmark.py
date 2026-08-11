#!/usr/bin/env python3
"""Run supported sparse-attention baselines through their official runtime code.

This runner is intentionally conservative: if a baseline repository does not
support Qwen2/Qwen2.5 directly, the script records an explanatory manifest
instead of substituting a local proxy implementation. That keeps official
baseline numbers separate from the unified-runner ablations.
"""

from __future__ import annotations

import argparse
import hashlib
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
            if row.get("_sample_role") != "task_eval":
                continue
            rows.append(row)
            if len(rows) >= limit:
                break
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def input_ids_sha256(input_ids: torch.Tensor) -> str:
    values = ",".join(str(int(token)) for token in input_ids.view(-1).tolist())
    return sha256_text(values)


def tokenize_prompt(tokenizer, prompt: str, max_input_length: int):
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
    )
    return {key: value.to("cuda:0") for key, value in encoded.items()}


@torch.inference_mode()
def generate_with_synchronized_timing(model, tokenizer, encoded, max_new_tokens):
    """Measure the first prefill forward and synchronized generate wall time."""

    prefill_start = torch.cuda.Event(enable_timing=True)
    prefill_end = torch.cuda.Event(enable_timing=True)
    timing_state = {"started": False, "finished": False}

    def before_forward(_module, _args, kwargs):
        input_ids = kwargs.get("input_ids")
        if (
            not timing_state["started"]
            and input_ids is not None
            and input_ids.ndim == 2
            and input_ids.shape[1] > 1
        ):
            prefill_start.record()
            timing_state["started"] = True

    def after_forward(_module, _args, _kwargs, _output):
        if timing_state["started"] and not timing_state["finished"]:
            prefill_end.record()
            timing_state["finished"] = True

    pre_handle = model.register_forward_pre_hook(
        before_forward, with_kwargs=True
    )
    post_handle = model.register_forward_hook(after_forward, with_kwargs=True)
    torch.cuda.synchronize()
    total_start = time.perf_counter()
    try:
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    finally:
        pre_handle.remove()
        post_handle.remove()
    torch.cuda.synchronize()
    total_seconds = time.perf_counter() - total_start
    if not timing_state["finished"]:
        raise RuntimeError("Could not identify the prefill forward")
    prefill_seconds = prefill_start.elapsed_time(prefill_end) / 1000.0
    return output, prefill_seconds, max(total_seconds - prefill_seconds, 0.0)


def percentile(values: List[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


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
    ap.add_argument("--warmup_num_samples", type=int, default=1)
    ap.add_argument("--reference_input_manifest")
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
        "warmup_num_samples": args.warmup_num_samples,
        "timing": "synchronized_wall_clock_with_cuda_event_prefill",
        "hardware": {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_total_memory_bytes": torch.cuda.get_device_properties(
                0
            ).total_memory,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
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
    reference = None
    if args.reference_input_manifest:
        reference = json.loads(
            Path(args.reference_input_manifest).read_text(encoding="utf-8")
        )
        if len(reference["samples"]) < len(samples):
            raise ValueError("Reference input manifest is too short")
        for index, sample in enumerate(samples):
            prompt = build_prompt(sample)
            encoded = tok(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_input_length,
            )
            expected = reference["samples"][index]
            if (
                str(sample.get("_id")) != str(expected["sample_id"])
                or int(encoded["input_ids"].shape[1])
                != int(expected["input_tokens"])
                or sha256_text(prompt) != expected["prompt_sha256"]
                or input_ids_sha256(encoded["input_ids"])
                != expected["input_ids_sha256"]
            ):
                raise RuntimeError(
                    "Preflight input alignment failed at eval index "
                    f"{index}: {sample.get('_id')}"
                )
        LOG.info(
            "Preflight input alignment passed for all %d eval samples",
            len(samples),
        )

    for warmup_index, sample in enumerate(
        samples[: args.warmup_num_samples], start=1
    ):
        prompt = build_prompt(sample)
        encoded = tokenize_prompt(tok, prompt, args.max_input_length)
        generate_with_synchronized_timing(
            model, tok, encoded, args.max_new_tokens
        )
        del encoded
        torch.cuda.empty_cache()
        LOG.info(
            "Completed unmeasured warmup %d/%d",
            warmup_index,
            args.warmup_num_samples,
        )

    results = []
    correct = 0
    for idx, sample in enumerate(samples, 1):
        sid = sample.get("_id", str(idx))
        prompt = build_prompt(sample)
        enc = tokenize_prompt(tok, prompt, args.max_input_length)
        input_tokens = int(enc["input_ids"].shape[1])
        prompt_hash = sha256_text(prompt)
        token_hash = input_ids_sha256(enc["input_ids"].cpu())
        if reference is not None:
            expected = reference["samples"][idx - 1]
            if (
                str(sid) != str(expected["sample_id"])
                or input_tokens != int(expected["input_tokens"])
                or prompt_hash != expected["prompt_sha256"]
                or token_hash != expected["input_ids_sha256"]
            ):
                raise RuntimeError(f"Input alignment failed for sample {sid}")
        out, prefill_seconds, decode_seconds = (
            generate_with_synchronized_timing(
                model, tok, enc, args.max_new_tokens
            )
        )
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
            "prompt_sha256": prompt_hash,
            "input_ids_sha256": token_hash,
            "prefill_latency_sec": prefill_seconds,
            "decode_latency_sec": decode_seconds,
            "latency_sec": prefill_seconds + decode_seconds,
        }
        results.append(rec)
        write_json(out_dir / "task_eval" / f"sample_{idx:03d}_{sid}" / "result.json", rec)
        write_json(out_dir / "results.json", results)
        if idx % 1 == 0:
            LOG.info("%s eval %03d/%03d id=%s pred=%s gold=%s correct=%s acc=%.4f", args.method, idx, len(samples), sid, pred, gold, ok, correct / idx)
        torch.cuda.empty_cache()
    latencies = [float(result["latency_sec"]) for result in results]
    summary = {
        "baseline_id": args.method,
        "implementation": "official_repository",
        "status": "completed",
        "count": len(results),
        "correct": correct,
        "accuracy": correct / max(len(results), 1),
        "avg_prefill_latency_sec": sum(
            float(result["prefill_latency_sec"]) for result in results
        ) / max(len(results), 1),
        "avg_decode_latency_sec": sum(
            float(result["decode_latency_sec"]) for result in results
        ) / max(len(results), 1),
        "avg_latency_sec": sum(latencies) / max(len(results), 1),
        "median_latency_sec": percentile(latencies, 0.5),
        "p95_latency_sec": percentile(latencies, 0.95),
        "avg_input_tokens": sum(r["input_tokens"] for r in results) / max(len(results), 1),
    }
    manifest["status"] = "completed"
    write_json(out_dir / "manifest.json", manifest)
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "results.json", results)
    LOG.info("DONE %s accuracy=%.4f", args.method, summary["accuracy"])

if __name__ == "__main__":
    main()
