# Calibrated Cross-Representative Residual

Independent Llama-3.1-8B-Instruct KV Retrieval experiment. Production source and
historical results are not modified. The existing AE K=3 groups are retained.

## Calibration

- Recover original public KV records 0, 1, 2, excluded from the formal 497.
  Verify record 3 exactly matches current formal record 0 and all calibration
  identities are disjoint from formal evaluation.
- Fit on records 0 and 1, gate once on record 2. This is a very small calibration
  set, not evidence of broad generalization. Gate results are not independent
  final benchmark results.
- Use full-length standard chat inputs and a per-head probability-mean
  compact10240 forward trajectory. Compare every quota on the same Q/K.
- Collect 32 uniformly spaced query tiles per record/layer. Each selected tile
  uses all real queries and globally causal per-query softmax; sampling tiles
  does not approximate the probability computation inside a tile.
- Reference masks use the member's own probabilities, the corresponding fixed
  or Top-p rule, and identical sink/local protection.
- Scan residual=1024 and A:B quotas 0:1024, 256:768, 512:512, 768:256, 1024:0.
  A and B are the other AE groups in ascending group index. Borrow token indices,
  not their K/V vectors. Candidates are the source representative's own final
  selected mask, ranked by its own P, excluding the destination shared core.
- Primary score: net member target intersection relative to original shared,
  not relative to the reduced core. Tie-break by member probability mass gain.
- Enable only when net count gain is positive on both fit records and the gate
  record, and mean member probability mass loses no more than 0.01 on any record.
  Otherwise retain original shared. Representative heads are never replaced.
- Keep first-priority A quota, scan B after deduplication, fill from remaining A,
  then from removed own-group keys. Invalid entries do not consume budget.

## Online

Only three representative probability distributions are computed per layer.
Each real query normalizes over all its legal keys before the tile average.
There is no second softmax, no global tail, no block projection or density filter.

Fixed method: original final budget 10240 includes sink first128 and current
query tile. Reserve up to1024 positions for calibrated residual replacement.

Top-p matched variant: .99 cumulative prefix with minimum1024 defines the
original target, then union sink/local without a fixed cap. Residual replacement
preserves that original final count on the same current Q/K. It does NOT promise
that final representative mass remains .99. End-to-end dynamic counts may also
change after earlier layers change hidden states. This interpretation requires
user confirmation before online execution; calibration itself is independent.

All final indices remain sorted original positions, with original RoPE, correct
Q-to-KV mapping, and per-query exact kernel causality. Heads with identical source
and quota reuse an index pattern; other members get their own resulting masks.

## Validation and Reporting

Run `test_policy.py`, then calibration, then three aligned full-generation smoke
records before formal 497 per method. BF16, batch1, greedy, seed42, context131072,
standard KV generation budget128, original runner chat and truncation.

Synchronized first-model-forward prefill includes all selector and residual list
work, index building, attention and required GPU statistics. CPU serialization
and calibration are excluded. Equal attention pairs do not imply equal latency.
Save full input hashes, predictions, raw selected/causal pairs, per-layer/pattern
counts, proxy probability mass, novel/dropped counts, evidence coverage and memory.
Online probability mass is representative proxy mass; member distributions are
not computed online. Compare full aligned results to existing shared/per-head
probability-mean and historical baselines, with hardware and timing caveats.

Remote code: `/home/ubuntu/work/experiments/rep_residual_20260907`
Remote output: `/local/results/rep_residual_20260907`
