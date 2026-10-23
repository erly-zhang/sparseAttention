#!/usr/bin/env python3
"""Offline vLLM throughput: identical fixed random prompts for every arm, ignore_eos.

Same measurement as `vllm bench throughput` (output tokens / wall time of one
llm.generate over all prompts, ignore_eos=True) plus a preemption count from the
engine metrics (the TriAttention runtime is invalid under preemption).
The arm (dense / rp / tri / hybrid) is selected purely by the launcher's environment.
"""
import argparse
import json
import os
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/scratch/yl258/jz648/Qwen3-4B")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--input-len", type=int, default=1024)
    ap.add_argument("--output-len", type=int, default=8192)
    ap.add_argument("--num-prompts", type=int, default=64)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--max-model-len", type=int, default=0)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    rng = np.random.default_rng(args.seed)
    max_len = args.max_model_len or max(16384, args.input_len+args.output_len+256)
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=max_len, enforce_eager=True,
              enable_prefix_caching=False, max_num_seqs=args.max_num_seqs,
              gpu_memory_utilization=args.gpu_memory_utilization, disable_log_stats=False, seed=args.seed)
    vocab = llm.get_tokenizer().vocab_size
    # Token ids in [1000, vocab-1000): avoids special tokens; identical across arms.
    prompts = [TokensPrompt(prompt_token_ids=rng.integers(1000, vocab-1000, args.input_len).tolist())
               for _ in range(args.num_prompts)]
    warm = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=16, ignore_eos=True)
    llm.generate(prompts[:2], warm, use_tqdm=False)

    def preemptions():
        total = 0
        for m in llm.get_metrics():
            if m.name == "vllm:num_preemptions":
                total += int(getattr(m, "value", 0))
        return total

    pre0 = preemptions()
    sp = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=args.output_len, ignore_eos=True, seed=args.seed)
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    elapsed = time.perf_counter()-t0
    out_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
    result = dict(arm=args.arm, input_len=args.input_len, output_len=args.output_len, num_prompts=args.num_prompts,
                  max_num_seqs=args.max_num_seqs, max_model_len=max_len, elapsed_s=elapsed,
                  output_tokens=out_tokens, output_tok_per_s=out_tokens/elapsed,
                  total_tok_per_s=(out_tokens+args.input_len*args.num_prompts)/elapsed,
                  preemptions=preemptions()-pre0,
                  env={k: v for k, v in os.environ.items() if k.startswith(("TRIATTN", "HYBRID", "ENABLE_TRI"))},
                  all_full_length=all(len(o.outputs[0].token_ids) == args.output_len for o in outs))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print("RESULT", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
