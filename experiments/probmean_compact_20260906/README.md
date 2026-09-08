# Probability-Mean Token-Compact KV Retrieval

## Scope

Instance2 only, Llama-3.1-8B-Instruct, 497 canonical KV Retrieval inputs per
method, BF16, batch 1, greedy, seed 42, chat template, context 131072 and
max_new_tokens 128. Production runner and sparse kernel are imported without
editing their files. Historical results remain untouched.

Methods:

| Method | Mask sharing | Target rule | Final rule |
|---|---|---|---|
| probmean_shared_compact10240 | Existing AE K=3 | Unprotected TopK10240 recorded separately | Reserve sink/local, then fill to min(10240, legal keys) |
| probmean_perhead_compact10240 | None | Same | Same |
| probmean_shared_compact_topp99 | Existing AE K=3 | Shortest 0.99 prefix, floor1024 | Union sink/local; no fixed cap |
| probmean_perhead_compact_topp99 | None | Same | Same |

Each real query independently softmaxes over all original-position causal
keys. Tile probabilities are the arithmetic mean of those distributions,
divided by the count of real queries, including zero future-key entries.
There is no second softmax, no tail1024, no block projection or density filter.
Sink protects [0,128); local protects the current query tile. Original Q/K
positions, the installed model RoPE, and each Q head's GQA KV mapping remain
unchanged. Index rows are sorted, unique and nonempty; execution applies the
original-position per-query causal mask.

The isolated control logitmean_notail_shared_compact10240 is smoke-only:
for key j it averages QK logits over queries i in the tile satisfying i>=j.
Its denominator is the number of visible queries for that key. This is the
historical causal-logit-mean definition with no tail, not a probability mean.
Its softmax is only a proxy-mass representation; fixed TopK ranks the raw
logits directly, avoiding underflow-induced probability ties. It is not a
four-probe selector.

## Correctness and Timing

`test_selector.py` compares a two-pass Triton calculation with independently
materialized FP32 causal softmax on small tensors. It tests incomplete tiles,
future zeros, GQA mapping, non-contiguous layout, shared/per-head CSR masks,
exact final budgets and unbounded Top-p. Production Torch GQA reference is not
used as the truth implementation.

LSE is computed over the full legal key prefix for every query; the second
pass recomputes logits and reduces probabilities over queries. Key chunks are
never independently normalized. No N-by-N attention map is retained.

Runtime wrapper preserves the canonical first-forward/decode boundary and
adds synchronized wall-clock prefill measurement. Required GPU reductions
for count/mass/evidence statistics are included in the measured prefill.
Model loading, compilation warmup, CPU transfer of statistics, NPZ/JSON
serialization and diagnostic analysis are outside measured generation.
Existing CUDA-event prefill is retained as a separate field. Do not compare
new L40S instrumented timings to historical A100 or differently instrumented
results as a direct speedup.

For each sample, `selection_stats/*.npz` stores the target/final counts,
target/final probability masses, P sums and answer-value/full-pair coverage
numerators and denominators for every source head and query tile of every
layer. JSON maps each source to its member Q heads. Shared rows can be expanded
exactly; no averaged representative is substituted. These evidence counts
use the union of legal keys for a query tile, not a per-query pair recall.
Exact selected/causal token-pair sums are separately recorded in online metrics.

## Execution

Use the attentionmap environment and PYTHONPATH=/home/ubuntu/work.
Run `test_selector.py`, then `schedule.py --phase smoke`. It runs the same
three data records for all four methods and the logit control, checks full
generation and input alignment, and writes `smoke_passed.json` only on success.
After that, `schedule.py --phase formal` launches two disjoint sample shards
per method on eight otherwise unoccupied GPUs. Each GPU has at most one
inference process. It refuses occupied GPUs and never stops other processes.

Each attempt has a distinct output directory. Completed attempts are reused;
failed attempts are retained. Incomplete attempts are not called complete or
merged. The final merge requires all 497 unique reference identities per
method, produces full input alignment, and recomputes raw pair sparsity and
latency mean/median/P90. `summarize.py` aggregates raw tile statistics and
compares existing baseline/historical summaries without rerunning them.

## Interpretation

Historical shared compact10240 used tile_sum_plus_tail1024 and already
aggregated all tile queries. This experiment changes normalization order,
sum/mean scaling and removes tail; its difference from that historical run
cannot be assigned only to normalization order. The no-tail logit smoke
control helps, but a three-sample accuracy difference is not a full ablation.
TopK and Top-p are not guaranteed to have equal sparsity. Neither target mass
nor answer-slot recall alone establishes causal task importance.
