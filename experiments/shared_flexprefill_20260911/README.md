# AE3 Shared FlexPrefill Pattern

## Question

Does broadcasting the representative's OFFICIAL FlexPrefill final block mask
reproduce the quality loss seen with shared token-compact selection?

The treatment is `flexprefill_ae3_shared_pattern`. It reuses the existing
benchmark-specific AE K=3 partition. Each representative runs the unchanged
FlexPrefill selector, including pattern detection, dynamic budgets, vertical/
slash projection, and query-aware block selection. All members inherit the
representative's final two-dimensional block index set. They retain their own
Q and correctly GQA-mapped K/V. Attention outputs and Q/K/V are NEVER shared.

No AutoBlock token target selection, 8192/10240 cap, 12.5% block filter,
token-compacted kernel, or residual is added. Official short-input fallback,
RoPE, causal block kernel and decode remain unchanged. Frozen-mask kernel
causality is tested separately from the selector's use of tail queries.

## Controls and status

- Intended first task: Llama-3.1-8B-Instruct, KV Retrieval.
- Same existing AE3 JSON, canonical 497 input hashes and token counts.
- Official FlexPrefill comparator, not the old AutoBlock Full configuration.
- Live Wulver installer: block128, gamma0.95, tau0.1, minimum512 explicitly
  passed by the isolated adapter, no maximum. Both completed smokes use this
  configuration. The local old installer had gamma0.9; do not conflate them.
- BF16, batch1, greedy, seed42, max context131072, standard KV generation128.
- First run tensor tests, then identical three-input full-generation smoke.
- A new full 497-input sbatch requires a separate resource/config approval.
- This is NOT a matched-sparsity claim. Report actual final causal pairs,
  accuracy, synchronized prefill/decode/total, and peak memory.
- Pair statistics are installed after mask broadcasting, so all Q heads count.
- Detailed primitive dumps are rejected until their group remapping is added.
- Do not use compilation/smoke latency as formal latency. Warmup must be
  performed outside timed measurement before formal submission.

## Commands

Current status: tensor correctness and both aligned three-input full-generation
smokes passed. See `RESULTS.md` and `verified_smoke_comparison.json`. A formal
497-input job has not been submitted.

Run from the project root with its existing torch/CUDA/FlexPrefill environment:

```bash
python -m unittest experiments.shared_flexprefill_20260911.test_adapter -v
python -m experiments.shared_flexprefill_20260911.run --help
```

Use the canonical runner arguments for model, task config directory, group
config, reference metrics, and a NEW output root. Never supply `--create_reference`
for this comparison. For smoke use `--limit 3`; standard task generation remains
128, not single-token profiling. This directory contains no automatic sbatch
submission and does not modify any prior experiment.
