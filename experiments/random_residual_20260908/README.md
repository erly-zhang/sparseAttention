# All-member random residual, Llama KV Retrieval

Independent experiment; production selectors and historical outputs are unchanged.

## Definition

- Model: Llama-3.1-8B-Instruct, existing AE K=3 groups.
- Representative score: mean of independently causal-normalized real-query
  probabilities in each 128-query tile, using the correct GQA KV head.
- Three representatives per layer retain Shared compact10240.
- All 29 other heads use B=min(10240,N), C=min(9216,B), R=B-C.
- Reserve sink positions [0,128) and the current real query tile inside C;
  fill C using representative probabilities.
- Draw R distinct keys uniformly from all tile-visible positions outside C.
- No calibrated gating, proxy-source quotas, tail term, block projection, or
  block-density filter. No second softmax.
- Sort original key positions before compact attention; apply exact causal
  masking separately for every real query inside the kernel.

## Reproducibility

Seed42 streams use SHA256 of experiment seed, full input-ID hash, layer,
Q head and query tile. Triton Philox draws use rejection before modulo to
avoid integer-range bias. Retaining the first R distinct draws implements
sampling without replacement. Sampler state does not use global model RNG.

## Gates and outputs

`test_random.py` checks independent numerical references, short and partial
tiles, budgets, protection, all-member activation, GQA, global RNG preservation,
and identical masks across two GPUs. Its result is `correctness.json`.

`schedule.py` first runs the same three formal inputs with full generation.
Input identity, budgets, and exact selected/causal pair equality against
Shared10240 must pass before eight disjoint shards cover 497 inputs.
Each GPU is checked for occupancy before launch; no existing job is stopped.

Remote output root: `/local/results/random_residual_allmembers_20260908`.
`manifest.json`, `source_hashes.json`, frozen source dependencies, command files,
and per-attempt logs retain provenance. Original outputs are never overwritten.
Merged success requires 497 distinct input identities, passed/full alignment,
and exact per-sample and total pair-budget equality.

Per-input compressed arrays retain layer/head/tile core, residual, final,
and novel-residual counts, representative probability mass, and answer/value
evidence coverage. CPU serialization is outside prefill timing. GPU statistics,
sampling, sorting, deduplication, indexing and attention remain inside timing.
Evidence coverage describes token slots, not causal importance.

Only one full random seed is run. The gated proxy residual comparison changes
both activation scope and selection rule, and is not a strict single-factor
control. Increased diversity alone does not establish improved accuracy.
