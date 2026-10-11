Current-query W1: raw-QK TopK selection
====================================

The opt-in `run_full_w1_qk.py` runner uses the optimized full-history W1 flow
with raw BF16 representative-query/key dot products as selection scores. It
removes the selector's positive scaling, FP32 softmax and query-mean reduction.
Actual attention still applies its scaling and softmax over the selected keys.

With one query and positive temperature, softmax is strictly order-preserving
in exact arithmetic, so raw QK and probability TopK have the same mathematical
ranking. This is not a bitwise equivalence guarantee: BF16 division can merge
distinct logits and FP32 softmax can underflow or round probabilities to ties.
The raw-QK variant ranks its BF16 matmul outputs directly; equal raw scores favor
earlier original positions. Stable sorting accepts negative scores. The existing
packed-probability backend is rejected because its encoding assumes nonnegative
FP32 probabilities. This simplification does not apply to multi-query probability
means or Top-p selection.

Full KV retention, current-query W1, representative RNG/GQA mapping, total K,
prompt and recent64 protection, chronological selected indices, every-step
selection and fused KV gathering are unchanged. The default probability runner
and previously frozen experiment sources continue to use their existing flow.
No ongoing experiment is switched to this variant.

Usage, with a new experiment output directory:

```bash
PYTHONPATH=ra_harness python scripts/run_full_w1_qk.py \
  --eviction_mode grid_full_w1_topk_protected [ordinary eval_hf arguments]
```

Metrics distinguish `full_w1_fixed_ranges_batched_qk_v1` and
`full_w1_score_mode=qk` from the probability implementation. All ordinary
generation softmax/temperature/Top-p settings remain separate from the selector.

CPU checks cover mathematical rank equivalence, raw negative-score selection,
protection, short histories, budgets, GQA, private RNG, active-batch shrink,
default probability regression, factory isolation and an explicit softmax
underflow case whose mask differs. GPU attention equivalence, full generation,
speed and HMMT accuracy of the raw-QK variant require separate GPU validation;
no speedup or accuracy result is claimed by these checks.
