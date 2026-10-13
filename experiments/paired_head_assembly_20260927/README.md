# Paired Head Output Assembly Smoke

Status: local correctness tests passed; remote validation and submission pending.
Do not submit until the user approves the resource proposal.

## Exact Hypothesis

Change only assembly after attention, not the attention graph. Each original
position i belongs to head i % H. The frozen q_kv routing kernel computes that
head's output O_h = A_h V_h on its assigned causal token subset. Scatter the
outputs back into M of shape [B,N,d].

Old: concat H copies of M, then use the original full o_proj.
New: each output slot receives M only on the rows owned by either of its two
assigned heads, with all other rows zero; concatenate H slots, then use the
same full o_proj. These are output vectors, not N-by-N attention probabilities.
Do not zero attention logits before softmax to implement this operation.

For H=3 the ordered slots are (0,1), (0,2), (1,2), matching the requested toy.
For H=32 use sorted undirected cycle edges: (0,1), (0,31), (1,2), ..., (30,31).
There are exactly H output slots and every source head appears twice.
This is not all possible pairs, nor H-1 heads in every slot.

No magnitude compensation is applied. Each token appears in 2 rather than H
slots. Thus differences can reflect both channel layout and output magnitude;
this experiment alone cannot separate those effects or guarantee lower KL.
No new cross-group attention edges or missing original head outputs are restored.

## Matched Protocol

- Model: frozen PSC Llama-3.1-8B-Instruct, BF16, FlashAttention-2, batch 1.
- Methods: dense, old repeated assembly, new pair assembly, in that order.
- Same 3 frozen token sequences: source indices 331, 61, 16. No new tokenization.
- Dense layers (zero based): 2,5,8,11,14,17,20,23,26,29,31; other 21 routed.
- Full original-position KV cache, correct GQA and RoPE; every decode step dense.
- Greedy generation, seed 42, max_new_tokens 128, original EOS and official scorer.
- Compare dense and repeat token IDs against historical results; fail closed on
  either regression before running the pair variant.
- Save prediction, generated IDs, input hash, source hashes, pair mapping,
  memory and GPU-synchronized prefill/decode timings for each input.
- KL(dense || method) uses the next-token vocabulary distribution at the final
  prompt position, float64, natural logarithm. It is not layerwise attention KL.
- Timing excludes model loading, full-length warmup, KL, scoring and serialization.
  Decode time is dependent on generated length; it is not a matched throughput test.
- Pair and repeat must have identical routed attention pair counts. Pair assembly
  still materializes [B,N,H*d] and uses dense o_proj; zeroing is not sparse-kernel speedup.

## Resource Proposal

One independent PSC GPU-shared job, account cis260309p, one H100-80GB, 8 CPUs,
128GB host RAM, maximum 30 minutes, no requeue. No full benchmark is authorized.
The current V100 development allocation cannot run the matched BF16/FA2 protocol.

Before submitting: verify the live allocation and environment, deploy into this
new directory without overwriting any run, compare local/remote SHA256, run
preflight and local/remote correctness tests, obtain approval, then use the actual
run.sbatch with sbatch --test-only. If rejected, stop; do not change routes to bypass.
Only a successful real sbatch response is a job ID. Record it and cross-check Slurm.

Completion requires Slurm COMPLETED/0:0 and complete.json containing 9 aligned
records with both historical regressions and equal routed pair counts passing.
Preserve partial failures; no automatic retry or overwrite.

## Local Checks

```sh
/opt/anaconda3/envs/pyt/bin/python experiments/paired_head_assembly_20260927/test_assembly.py
bash -n experiments/paired_head_assembly_20260927/run.sbatch
```

Tests cover the exact 3-head example, balanced pairs at H=3/4/16/32, short and
nondivisible sequences, padding, projection bias, original-position alignment,
repeat regression, a separate causal/GQA oracle, and tiny Llama cached dense decode.
