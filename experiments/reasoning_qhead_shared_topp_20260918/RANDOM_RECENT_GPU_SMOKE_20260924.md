# Random 40% + recent64 + GPU sampling smoke

New method: `qhead_shared_mask_random_recent_gpu`.
Frozen run: `results/full_kv_random_r040_recent64_gpu_smoke_20260924`.

The user stopped the previous `full_kv_random_r040_20260923` smoke before its first question completed. Its pipeline and inference process were terminated, its partial logs retained, and its automatic 500-question submission disabled. Other experiments were left running. This new pipeline runs correctness and three complete questions only.

## Selection and cache semantics

For each active rollout, layer, and single-token decode step, let P be the prompt length and R the number of non-prompt KV entries including the current step's input token. Set K=ceil(0.4R), L=min(64,K). Keep all P prompt positions, the latest L non-prompt positions, and a uniformly sampled subset of K-L positions from the older R-L non-prompt positions. Thus exactly P+K positions participate in attention. The recent window is **inside** the 40% budget. In the short-history case K<=64, only the latest K reasoning positions are retained for this attention call.

Every layer reconstructs its own mask every step and shares it across all Q heads. The nominal representative head changes each step and is shared across layers, but has no effect on uniform selection and requires no QK selector scores. W16/W64 and selection Top-p are not used.

All KV entries remain in the persistent full-history cache. Only the attention input is gathered into a temporary buffer. Original RoPE positions, GQA mapping, and independent per-head attention normalization remain unchanged. An unselected historical position can return at a later step. No 1024 eviction threshold or persistent eviction exists in this method.

## Three changes

1. Budgeted recent64 protection as above. This changes the mask distribution; it is a deliberate algorithm change, not just a kernel optimization.
2. Private GPU generator keyed by a stable hash of seed42, input index, step, layer, and original rollout identity. Generate a 32-bit uniform primary key and an independent secondary random key, pack them into a nonnegative 63-bit integer with position as the final exact tie-break, and select global TopK over eligible older positions. Sort selected indices by original position. Exact double-key collisions favor larger position; their probability is very small but not zero. This stream does not match the old CPU randperm sampler, and CPU/GPU bitwise equivalence is not claimed. Global generation RNG state is unchanged. Same-device reruns and rollout filtering are tested; this smoke uses only one physical A100, so it does not independently validate cross-GPU reproducibility.
3. A single Triton kernel gathers K and V directly into reusable, geometrically growing temporary buffers. It removes intermediate per-row index_select allocations and the subsequent copy. Runtime lengths and strides are not specialized, avoiding per-token compilation. Python full-history length also avoids unnecessary GPU-to-CPU length readback. This still copies selected KV once; it is not a direct indexed sparse-attention kernel.

## Validation and timing

20 tests cover budget, protection, uniqueness, short and >1024 histories, no QK selector storage, private RNG state, repeatability, batch filtering, small-population marginal frequencies, exact gathered values, reusable buffers, and BF16 GQA attention against an independent reference. All passed before model smoke.

Smoke uses canonical MATH500 inputs 0,1,2, Qwen3-4B BF16, two rollouts per question, existing chat preprocessing, seed42, temperature0.6, generation Top-p0.95, and total length32768. Selection Top-p is absent. Full outputs are generated, not a shortened token-budget benchmark. Validation checks input token counts and hashes, full-history cache lengths, exact selected/causal pair totals, EOS/length-limit status, and six graded completions.

The runner records synchronized prefill/decode wall time, generated lengths, decode throughput, pair sparsity and peak allocated memory. Model loading and warmup are excluded. Compare throughput separately from completion length. Old W16/W64 differ in masks and sparsity, so their end-to-end timing comparison does not isolate gather/sampler speed. The previous random40 run has no completed first-question result and cannot provide a matched full-smoke speedup. Dense historical batch time has a different timing scope. Three questions cannot establish MATH500 accuracy.

Progress: see `pipeline_status.json`; final results: `smoke/summary.json`, `smoke/grades.jsonl`, and `smoke/cache_metrics.jsonl` beneath the frozen run directory. No automatic full-run submission is enabled.

One read-only live stack snapshot was taken during the second smoke question to confirm progress; any short tracing pause remains included in its decode time. The snapshot counters and caveat are saved as `live_progress_snapshot.json`.

## Completed smoke results

20/20 tests passed. Three questions / six rollouts completed; 6/6 correct, 6/6 EOS. Input hashes, full-history continuity and exact pair counts passed. Mean decode 195.75s/question, aggregate throughput 30.91 tokens/s, global pair sparsity 56.82%, peak allocated memory 11.18 GiB. Same-three-input W16/W64 throughput was 19.96/20.04 tokens/s. See [full smoke report](results/full_kv_random_r040_recent64_gpu_smoke_20260924/smoke/REPORT.md) for lengths, timing scope and comparison limits. The pipeline is completed with automatic full-run submission disabled.

## Subsequently authorized full MATH500 run

Following the user’s explicit request, the passed smoke inference implementation was frozen unchanged in `results/full_kv_random_r040_recent64_gpu_20260924`. GPU array **1320734** (8 shards, one A100 each) and dependent collector **1320735** were submitted. Scope: 500 questions, two rollouts each; comprehensive timing, accuracy, sparsity, memory, input-alignment and baseline-comparison exports. See [submission record](results/full_kv_random_r040_recent64_gpu_20260924/SUBMISSION.md). This is a separate explicitly submitted run; the earlier smoke pipeline remains completed and has no automatic submission enabled.
