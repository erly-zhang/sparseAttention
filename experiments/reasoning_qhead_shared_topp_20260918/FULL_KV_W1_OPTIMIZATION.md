Full-history W1 implementation optimization (2026-10-08)
========================================================

The opt-in implementation preserves the HMMT selector grid algorithm: dense
prefill, every KV retained until generation ends, one stable-seeded representative
Q head per layer and decode step, correct GQA mapping, BF16 QK/division followed
by float32 softmax, and selection before every decode attention. All KV heads
share selected positions while retaining their own values. K is the total
attention-selection budget, including the full prompt and recent64 protection.
Exact ties favor earlier original positions; selected IDs are chronological.

Implementation changes:

* Compute eligible/protected ranges from the full-history length instead of
  extracting Boolean masks with CUDA nonzero and CPU synchronization.
* Rank both active rollouts in one stable argsort and sort only the selected
  middle range before concatenating the already ordered prompt and recent64.
* Reuse the ID and attention-length buffers. Lengths are determined by the
  CPU-side full-history invariant; no per-step CPU-to-GPU list upload is needed.
* Append K/V, original position and cache length in one Triton kernel.
* Read the current W1 query directly and reuse the private CPU generator while
  re-seeding it exactly as the historical representative-head sampler.

An optional packed-integer TopK backend uses the exact float32 probability bits
and original-position tie priority. It passed mask tests but was slower at
32700 candidates in the first microbenchmark; stable_sort remains the default.
No approximate score computation, changed cadence or permanent eviction is used.

Usage (all ordinary eval_hf arguments remain valid):

```bash
PYTHONPATH=ra_harness python scripts/run_full_w1_optimized.py \
  --eviction_mode grid_full_w1_topk_protected [other eval_hf arguments]
```

The wrapper opts in only in its own process. Existing runners and frozen
experiment sources continue to use the historical implementation. Modes other
than full/W1/TopK use the original selector class.

Evidence and the gated three-question comparison are stored in:

`results/full_kv_w1_implementation_optimization_20261008/`

Core tests verify exact probability scores, exact masks including finite-score
ties/underflow, bitwise attention equality, an independent FP32 attention
reference, full-cache contents, GQA, batch shrink, pair counters and unchanged
global RNG. The additional tiny Qwen3 test checks a complete sampled token
trajectory and the actual runner/factory with RoPE. It is not an accuracy test.

The first microbenchmark used the same A100 MIG 2g.20gb slice for all variants.
Synchronized40-call wall measurements were repeated in four alternating rounds;
warmup, correctness checks and serialization were excluded. For batch2,
N8192/32700 and K1024/2048, selector time decreased about62–63% and the combined
single-layer cache/attention flow about57–59%. These are not full-model speedups
and are not directly comparable to historical full-A100 timings.

The paired generation pilot uses HMMT canonical IDs0,1,2, Qwen3-4B BF16, K2048,
prompt+recent64 protection, total max_length32768, two sampled answers/question,
seed42, representative seed0, temperature0.6 and generation top_p0.95. Baseline
and optimized runs use the same allocated A10080GB for each question; ordering
alternates across questions. At most one new GPU job runs at once. Token IDs,
EOS/lengths, original input hashes, full-cache retention and raw pair counts must
match exactly. Loading/warmup and post-timing token-ID serialization are excluded
from decode timing. Results from three questions do not establish full-HMMT
accuracy or a whole-dataset speedup.
