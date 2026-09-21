# AE3 Shared FlexPrefill: three-input smoke

## Status

Implementation and both three-input full-generation smokes are complete on
Wulver developer allocation 1250826, node n0046, one A100-SXM4-80GB.
No 497-input formal job has been submitted. No historical experiment was changed.

The treatment computes the existing AE3 representatives' official FlexPrefill
final two-dimensional block masks and broadcasts them within each group.
Members retain their own Q and correctly GQA-mapped K/V. The official block
attention kernel and decode path are retained. There is no AutoBlock projection,
token-compact kernel, residual, or fixed 8192/10240 token cap.

## Verified configuration

- Llama-3.1-8B-Instruct; existing benchmark-specific AE K=3 JSON.
- Same three prior KV Retrieval inputs: 123959, 123816, 123741 input tokens.
- BF16, batch one, greedy, seed42, chat template, context limit131072.
- Standard full generation, maximum128 new tokens; not one-token profiling.
- Both methods: block128, gamma0.95, tau0.1, minimum512, maximumNone.
- PyTorch2.6.0+cu124, CUDA12.4, same physical GPU, sequential runs.
- Both exits were0; both input alignments passed, ordered_subsequence, count3,
  reference_count497. The two runs also have identical input hashes and order.
- Raw dense causal pairs agree exactly with 32 layers x32 Q heads xN(N+1)/2.

The local old runner used gamma0.90, but the live Wulver official installer uses
gamma0.95. These results must not be labelled gamma0.90 or assumed identical to
an unverified historical baseline configuration. The isolated entry now rejects
configuration drift. The generic runner's logged fixed_topk_budget=8192 is an
unused parser default for these methods, not an active selection constraint.

## Results

| Method | Correct | Accuracy | Pair sparsity | Mean prefill (s) | Mean decode (s) | Mean total (s) |
|---|---:|---:|---:|---:|---:|---:|
| AE3 Shared FlexPrefill | 0/3 | 0.00% | 83.09% | 16.945 | 9.989 | 26.934 |
| Official per-head FlexPrefill | 1/3 | 33.33% | 85.86% | 14.971 | 2.228 | 17.198 |

Shared selected pairs: 3,983,385,620,608. Per-head selected pairs:
3,330,059,266,816. Dense causal pairs for each run: 23,556,322,372,608.

Shared prefill by input: 22.002, 14.533, 14.301 seconds.
Per-head prefill by input: 16.570, 14.122, 14.219 seconds.
Shared generated lengths:128,128,128. Per-head generated lengths:27,38,24.

Timing is synchronized wall-clock, including selector, broadcast, block kernel,
and pair-count instrumentation. The canonical4096-token warmup does not ensure
all long-shape compilation is excluded, so these are **smoke measurements, not
formal speed benchmarks**. Original summary.json files inherited a stale
CUDA-event timing label; the verified report uses the actual per-row wall-clock
fields. The isolated writer is now corrected for future runs; original outputs
and the exact code versions used by both smokes remain preserved.

The decode difference is also confounded by generated length: shared outputs
hit the128-token cap, whereas the per-head outputs stop sooner under the same
generation settings. This is not evidence of a slower decode kernel.

## Correctness and interpretation

Three unit tests passed on the allocated GPU, covering group partitions, GQA
routing, member-mask broadcasting, agreement with original representative masks,
V/S and query-aware selector branches, full and partial tiles, independent dense
masked numerical references, and fixed-mask kernel causality.

The shared method uses more final causal pairs yet answers fewer of these three
inputs correctly. This small sample does not support a simple "insufficient
total budget" explanation. It is consistent with mask/head mismatch, but is not
statistical evidence about the full497-task accuracy. This is also not a
matched-sparsity ablation: sharing changes the distribution of per-head budgets.

## Formal proposal, not submitted

Run the shared method on the existing497 KV inputs using the same settings,
one A10080GB, independent sbatch, six-hour time limit, approximately4-5 hours
execution based on this small smoke, excluding queue time. Add untimed long-input
warmup and record any compilation caveat. Proposed output:
`/project/yl258/jz648/sparseAttention/experiments/outputs/shared_flexprefill_20260911/formal_shared`.

Before reusing a historical full FlexPrefill comparator, verify its input hashes,
checkpoint, gamma/tau/budget configuration and timing scope. A new full baseline
run would be additional scope/resources and needs approval separately.

## Evidence

Local machine-readable copy: [verified_smoke_comparison.json](verified_smoke_comparison.json).
Remote root: `/project/yl258/jz648/sparseAttention/experiments/outputs/shared_flexprefill_20260911`.
Treatment: `smoke_flexprefill_ae3_shared_pattern_attempt2`.
Control: `smoke_flexprefill_attempt1`.
Each contains command/source hashes, correctness log, runner log, exit record,
online_metrics, official evaluator output and summary. Attempt1 of the shared
smoke stopped at argument parsing before loading the model; it is preserved.
