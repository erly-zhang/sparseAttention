# Current code guide

Updated 14 October 2026. Start here instead of choosing a dated trial script.
[Current report](../../../report_build/current/hmmt_results_en.pdf) ·
[Current results](RESULTS.md) · [Complete file inventory](CODE_INVENTORY.json).

## Version status

| Configuration | Formal HMMT status | Relevant implementation |
|---|---|---|
| Dense | 60 inputs /120 outputs | Native DynamicCache and FA2; sealed 2026-10-08 runner |
| Official random_pp | K1024 and2048 complete; no local K4096 result | `official_random_pp_cache.py` and vendored official code |
| Shared Random | Total K1024/2048/4096; compact64/500 and full KV complete | `selector_grid_cache.py` |
| W1 probability controls | K1024/2048/4096 full and compact64/500 complete | `selector_grid_cache.py`, `full_w1_optimized_cache.py` |
| W1 raw-QK compact | K1024 compact500; K1024/2048 compact1000 complete | `selector_grid_cache.py`, `w1_interval1000_cache.py` |
| W1 raw-QK full v2 | K1024 complete;120 outputs exactly match the prior raw-QK implementation | `w1_qk_fast_cache.py` |
| W1 raw-QK + random4096 | K1024 v2 and K2048 reference implementation complete; interval1000 | `w1_qk_fast_cache.py`, `w1_interval1000_cache.py` |
| W1 raw-QK fused v3 | GPU correctness and single-layer timings only; no new full-model HMMT result | `w1_qk_fused_cache.py` |

Raw-QK and probability scoring remain distinct. A better single-seed accuracy
does not make a different algorithm a replacement. Only exact same-algorithm
implementations with identical input hashes, all120 output texts and pair counts
replace the previous implementation in the main result table.

## Current entry directory

[`scripts/current/`](scripts/current/README.md) contains six short navigation
aliases. They resolve to the maintained scripts below, so their original
imports, data paths and process-local installation behavior are unchanged.

| Purpose | Maintained script | Alias |
|---|---|---|
| Formal raw-QK v2 full / hybrid | [run_w1_qk_fast.py](scripts/run_w1_qk_fast.py) | `scripts/current/w1_raw_qk_v2.py` |
| Validated fused v3 prototype | [run_w1_qk_fused.py](scripts/run_w1_qk_fused.py) | `scripts/current/w1_raw_qk_v3.py` |
| Compact1000 / hybrid reference | [run_w1_interval1000.py](scripts/run_w1_interval1000.py) | `scripts/current/w1_interval1000.py` |
| Official per-KV-head random_pp | [run_official_random_pp.py](scripts/run_official_random_pp.py) | `scripts/current/official_random_pp.py` |
| Validate and rebuild result catalog | [consolidate_results.py](scripts/consolidate_results.py) | `scripts/current/refresh_results.py` |
| Build English PDF/HTML report | [build_current_report.py](scripts/build_current_report.py) | `scripts/current/build_report.py` |

The raw-QK v2/v3 runners accept the normal `eval_hf.py` arguments.
`--eviction_mode grid_full_w1_topk_protected` selects full KV;
`--eviction_mode grid_hybrid1000_w1_topk_protected` selects the bounded pool.
Use `RA_W1_QK_FAST_BACKEND=stable_sort` to reproduce the latest formal full-KV
backend, and `packed_qk_topk` for the latest formal hybrid backend. Other
generation and cache arguments must follow the source row's frozen config.

`run_full_w1_optimized.py` now defaults to raw-QK. Reproducing its older
probability-scored result requires `RA_FULL_W1_SCORE_MODE=probability`; do not
infer scoring from the runner filename alone.

## Engine flow

1. `harness/eval_hf.py` constructs the model, prompt and cache configuration.
2. `harness/generation_utils.py` performs the common sampled generation loop.
3. `engine/modify_forward.py` preserves Q/K normalization, RoPE and GQA while
   dispatching to the selected cache's attention path.
4. The selected layer scores positions, forms the shared chronological mask,
   gathers K/V and invokes FlashAttention. Position sharing never shares K/V values.
5. The hybrid layer reuses the current W1 core after attention at each1000-step
   update, adds4096 Uniform-TopK positions outside that core, and swaps cache buffers.

All engine paths above are under `ra_harness/kvcompress/`. Factory installation
is process-local. Existing experiments use frozen source snapshots and are not
silently switched when this checkout changes.

### Timing reproduction detail

The sealed Dense 2026-10-08 source records prefill/decode phases for DynamicCache.
The maintained generic generation loop still gates separate phase timing on
EvictCache. Use the frozen Dense runner for a comparison requiring the historical
Dense timing scope. This is a logging difference, not evidence of an accuracy defect.

## Validation and launch preparation

| Check / workflow | Files |
|---|---|
| Raw-QK v2 exact masks/RNG/tiny model | `tests/test_w1_qk_fast.py`, `scripts/benchmark_w1_qk_fast.py` |
| Fused v3 exact masks/KV/attention/pool updates | `tests/test_w1_qk_fused.py`, `scripts/benchmark_w1_qk_fused.py` |
| Interval1000 and residual sampler reference | `tests/test_w1_interval1000.py` |
| Historical probability/full-KV oracle | `tests/test_full_w1_optimized.py`, `tests/test_full_w1_qk.py` |
| Freeze/submit v2 full/hybrid HMMT | `prepare_hmmt_w1_qk_fast.py`, `submit_hmmt_w1_qk_fast.py` |
| Freeze/submit compact1000/hybrid reference | `prepare_hmmt_w1_interval1000.py`, `submit_hmmt_w1_interval1000.py` |

The 168 indexed script/test/engine files include preparation, grading,
dataset-specific experiments, older controls and dependencies. Their role is
recorded in `CODE_INVENTORY.json`; “auxiliary_or_historical” does not mean safe
to delete. Original paths are retained because frozen runs and old launchers
depend on them. This organization does not delete code or alter inference.

## Read-only maintenance and reproduction

From this experiment directory:

```bash
/usr/bin/python3 scripts/current/refresh_results.py --check
/usr/bin/python3 scripts/current/refresh_results.py
/usr/bin/python3 scripts/current/build_report.py
```

Catalog/report construction is CPU-only and does not submit experiments.
Report PDF construction uses ReportLab4.4.3 in `.tools/report-runtime` and the
previous report's Times New Roman fonts. HTML is self-contained.

Inference uses the separate `random-attention` conda environment: Python3.10,
PyTorch2.4.0+cu121, Transformers5.0.0, FlashAttention2.7.3 and NumPy2.2.6.
Model weights, authorized datasets, canonical input references and sealed
reference runs remain local. A fresh Git clone cannot reproduce a formal row
until those assets are restored. Scheduler preparation refuses duplicate or
already sealed launches; do not resubmit an existing result directory.
