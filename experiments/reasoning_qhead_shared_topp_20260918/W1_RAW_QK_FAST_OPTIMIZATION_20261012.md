# Raw-QK W1 implementation optimization (2026-10-12)

This opt-in implementation optimizes full-history W1 and W1 with the bounded
random4096 candidate pool. It preserves the existing raw-BF16-QK algorithm,
representative identity, GQA mapping, prompt+recent64 protection inside K,
chronological IDs, attention normalization and generation settings.
Historical implementation files and frozen experiments remain intact.

## Implementation

- Encode each raw BF16 score and earlier-slot tie priority into a unique signed
  int64 or int32 key; use global TopK instead of sorting all candidates. For
  int32,16 score bits plus at most16 index bits cover up to65536 candidates;
  center the rank to preserve signed-integer order without overflow. CUDA preserves
  its existing CUB radix order, including NaN payload/sign for large arrays.
  Integer bit checks preserve signed-zero ties without subnormal flush.
  For fewer than4096 eligible CUDA positions use the original stable sort.
- Sort only the selected middle positions. One Triton kernel writes the
  protected prompt, chronological middle and recent64 into a reusable ID buffer.
- When all live positions fit inside K, skip unused QK scoring. Diagnostics still
  calculate the scores when enabled, and representative sampling stays unchanged.
- Hybrid compaction retains exactly the same primary Uniform float32 draws and
  independent secondary random draws from the same private per-rollout generator.
  Fuse priority packing and exclude the core before global TopK. Reuse the key
  buffer and omit the residual sort before the final chronological union sort.
- Preserve the existing fused KV append/gather and cache buffer swap. Compaction
  remains after attention every1000 decode forwards and retains K+4096 positions.

The new implementation is in `ra_harness/kvcompress/engine/w1_qk_fast_cache.py`.
`scripts/run_w1_qk_fast.py` opts in only within its process.
`RA_W1_QK_FAST_BACKEND` chooses `stable_sort`, `packed_qk_topk` (64-bit) or
`packed_qk32_topk` (32-bit); GPU microbenchmarks
select the exact backend independently for full history and bounded candidates.
Scores are unchanged BF16 matmul outputs; this does not fuse/recompute QK.

## Validation and HMMT

`tests/test_w1_qk_fast.py` checks exhaustive BF16 encoding order, actual GQA
geometry, exact masks/attention, an independent FP32 attention oracle, unchanged
private/global RNG, exact candidate KV/positions at two compactions, batch shrink,
and complete sampled token trajectories through a tiny Qwen3 model.
Historical raw-QK and interval1000 regression suites run too.

The initial validation exposed the CPU/CUDA NaN-order distinction. Its snapshot,
failure and diagnostic output are retained under
`results/hmmt_w1_raw_qk_fast_full_hybrid_k1024_20261012/`.
The corrected64-bit validation and first full-generation smoke are under
`results/hmmt_w1_raw_qk_fast_full_hybrid_k1024_20261012_validationfix/`.
The final three-backend comparison and full HMMT run are under
`results/hmmt_w1_raw_qk_fast_full_hybrid_k1024_20261012_packed32/`.
The latter waits for the earlier complete smoke/gate, keeping at most2 GPU
allocations active and retaining all preceding validation evidence.
Inspect the final run's `correctness.json`, `microbenchmark.json`, `jobs.json`,
`smoke_gate.json` and final `summary.json` for actual status and measurements.

Both authorized methods use HMMT60 canonical questions with2 sampled outputs per
question (120 answers/method), Qwen3-4B BF16, batch2, K1024, seed42, representative
seed0, temperature0.6, generation top-p0.95 and total prompt+output limit32768.
Full history attends K each decode step while retaining all KV. Hybrid attends K
from its live candidates and updates the physical K+4096 pool every1000 steps.

GPU correctness and synchronized layer timings precede complete generation on
the same3 HMMT inputs per method. The gate requires outputs to match the existing
same-storage raw-QK results exactly. Formal inference repeats all60 inputs in
20 independent3-question shards/method, with at most2 additional A10080GB GPUs.
Every shard verifies input length/SHA256, source/model hashes, exact reference
output text, actual selection budgets and analytic selected/causal pair counts.
Token IDs are saved after timed generation. Collection requires all120 distinct
answers per method and `input_alignment=passed/full`.

Correctness/microbenchmarks use an available A100 MIG20GB slice. Their timing
changes are synthetic same-device measurements, not full-HMMT speedups. Formal
inference uses full A10080GB allocations; decode wall time includes selection,
random generation, KV operations, attention and generation bookkeeping, excluding
loading, warmup, correctness and serialization. Maintenance and GPU availability
can delay queued jobs.
