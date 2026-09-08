# Shared60% + Residual40%, Llama KV Retrieval

Two methods, seed42, 497 formal inputs each:

- `probmean_shared6144_random4096_allmembers_seed42`
- `probmean_shared6144_member2q4096_allmembers`

Each layer retains the existing AE K=3 groups. Three representative Q heads
keep the original protected Shared10240 selection. All other 29 heads reserve
a protected Shared6144 core and fill the remainder of B=min(10240,N).
For N<=10240 the final set contains all visible keys. Sink first128 and current
real local tile count inside the core budget. No calibration gating or tail.

Representative P is the mean of all real tile-query causal probabilities.
No second softmax, block projection or density filter is used. A samples
uniformly without replacement from the full core complement. Stable SHA256
streams depend on seed, input identity, layer, head and query tile, independent
of global model RNG and GPU assignment. Complement ranks are mapped to original
positions using sorted core indices, without a per-member dense boolean mask.

B samples min(2,m) actual member queries at a+floor((u+0.5)*m/s), computes each
query's global causal normalization including shared positions, averages by
original key position, and only then excludes the core for TopK selection.
Equal float32 scores break ties toward lower original positions. This is a
two-query sampling approximation, not full128 mean, feature pruning or SparQ.

The attention kernel is unchanged. Shared core short indices are reused and
merged with member short residuals into sorted fixed-stride CSR. Every query
still uses exact original-position causal masking and correct GQA KV mapping.
There is one common attention softmax, not separately normalized core/residual.

## Validation

BF16 thresholds, fixed before testing: relative L2 <=0.005 and maximum absolute
output error <=0.02 against independent float32 masked attention. Original
ragged CSR and optimized fixed-stride CSR must produce identical kernel output.
Tests cover real-query counts, partial/padded tiles, contiguous/interleaved GQA,
protection, disjointness, exact budgets, deterministic ties, sampled probabilities,
global RNG independence and cross-GPU random masks.

After correctness.json passes, schedule.py runs both methods on the same three
formal inputs with full max_new_tokens128 generation and finite-logit/index
validation. Both smoke checks must pass before 994 formal generations start.

## Timing and diagnostics

Formal prefill is GPU-synchronized wall-clock for the first forward, including
representative scoring, member scoring/sampling, selection, sorted indices,
attention and minimal actual index pair counts. No expensive coverage/true-member
probability diagnostics or their serialization run in formal timing.

After formal generation, two independent three-input profiling runs compute
member full128 probabilities, member2q proxies separately, replacement loss,
novel mass, causal-pair recall/Jaccard against same-trajectory Per-head10240,
and value/KV evidence coverage. Raw layer/head/tile arrays and phase GPU-event
durations are preserved; these phase durations can include host submission gaps.

Same rule does not imply identical core masks across A/B forward trajectories.
Historical random9216+1024 changes both budget ratio and index/instrumentation
implementation relative to A. Historical times include extra GPU statistics.
A/B is the closer selection-signal control. No additional full seed or ratio
ablation is run, so ratio-only and implementation-only causal effects are not
identified. Increased mask diversity alone is not evidence of accuracy gains.

All outputs and source hashes: `/local/results/residual6040_20260908`.
Main sources and historical results remain unchanged. Completion requires both
497/497, unique full input alignment, scoring, exact measured pair comparison,
and the separate profiling artifacts.
