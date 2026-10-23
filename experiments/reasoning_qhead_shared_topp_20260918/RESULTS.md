# Complete result archive and current comparisons

Updated: 2026-10-14. Fixed entry point; replaces dated comparison tables.

60 current complete result groups; 3 equivalent older implementations are listed only in history. All methods use Qwen3-4B.

[Current English PDF report](../../../report_build/current/hmmt_results_en.pdf) · [Code and method guide](CODE_INDEX.md) · [Report directory](REPORTS.md)

The main PDF selects8 latest protected raw-QK W1 / Dense / official random_pp configurations. This complete catalog retains earlier validated controls and other datasets, which now appear in a separate historical PDF.

## Reading the results

Accuracy is correct sampled outputs / total outputs, not pass@2. HMMT has 60 questions × 2 outputs = 120; MATH500 has 500 × 2 = 1000.

Decode seconds measure one question batch producing two outputs concurrently. Step ms is total synchronized decode seconds / total batch decode forwards. Output tokens are per individual output.
Shared Random and official random_pp use different mask sharing and eviction phases. `prompt + reasoning1024` gives prompt tokens extra budget; `total K=1024` includes protection inside 1024. These are separate algorithms.

Scores marked probability and raw-QK remain separate because BF16 ties/underflow can change selected tokens. Compaction intervals 64/500/1000 and hybrid random4096 are also separate.

HMMT settings: BF16, seed 42, representative seed 0, batch 2, temperature 0.6, generation Top-p 0.95, total prompt+output cap 32768. Different A100 allocations and freely generated lengths limit speed comparisons.

## Superseded implementation

Only same-algorithm implementations with120 exact completion pairs, identical input hashes and identical attention-pair counts replace an older result. Different scores, budgets, storage policies or compression intervals remain separate controls. Observed timing differences use different A100 allocations.

| Configuration | Previous decode s/question | Current decode s/question | Correct | Observed reduction |
|---|---:|---:|---:|---:|
| W1 probability, full KV, K2048 | 1,790.8 | 1,270.4 | 52/120 | 29.06% |
| W1 raw-QK, full KV, K1024 | 1,326.8 | 1,267.8 | 48/120 | 4.45% |
| W1 raw-QK + random4096, interval1000, K1024 | 1,316.2 | 1,252.3 | 46/120 | 4.85% |

Old result and proof: [implementation history](results/current/implementation_history.json). Raw run evidence remains recoverable.

## HMMT: latest complete implementations and matched baselines

| Method / score | Budget | KV storage | Interval | Protection | Correct | Accuracy | Decode s/question | Step ms | Output tokens | Source |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---|
| Dense | all | full | 0 | all | 48/120 | 40.00% | 1,003.7 | 47.3 | 19,316.4 | [summary](results/hmmt_dense_and_timing_diagnosis_20261008/formal/summary.json) |
| Official random_pp (Uniform TopK) | total K=1024 | compact | 64 | protected | 19/120 | 15.83% | 1,159.6 | 39.9 | 28,340.3 | [summary](results/hmmt_official_random_pp_k1024_k2048_20261008/summary.json) |
| W1 (raw-QK) | total K=1024 | compact | 500 | protected | 14/120 | 11.67% | 1,226.7 | 42.0 | 28,384.9 | [summary](results/hmmt_w1_raw_qk_k1024_full_compact500_20261009/cells/hmmt_k1024_w1_raw_qk_compact_protected/formal/summary.json) |
| W1 (raw-QK) | total K=1024 | compact | 1000 | protected | 22/120 | 18.33% | 1,226.5 | 43.2 | 27,333.0 | [summary](results/hmmt_w1_raw_qk_k1024_hybrid4096_compact1000_20261011/cells/hmmt_k1024_w1_raw_qk_compact1000_protected/formal/summary.json) |
| W1 (raw-QK) | total K=1024 | full | 0 | protected | 48/120 | 40.00% | 1,267.8 | 52.5 | 22,295.9 | [summary](results/hmmt_w1_raw_qk_fast_full_hybrid_k1024_20261012_packed32/cells/hmmt_k1024_w1_raw_qk_fast_full_protected/formal/summary.json) |
| W1 + random4096 (raw-QK) | total K=1024 | hybrid | 1000 | protected | 46/120 | 38.33% | 1,252.3 | 52.3 | 22,268.5 | [summary](results/hmmt_w1_raw_qk_fast_full_hybrid_k1024_20261012_packed32/cells/hmmt_k1024_w1_raw_qk_fast_hybrid_protected/formal/summary.json) |
| Official random_pp (Uniform TopK) | total K=2048 | compact | 64 | protected | 38/120 | 31.67% | 1,058.0 | 40.3 | 23,961.6 | [summary](results/hmmt_official_random_pp_k1024_k2048_20261008/summary.json) |
| W1 (raw-QK) | total K=2048 | compact | 1000 | protected | 40/120 | 33.33% | 1,077.4 | 41.7 | 23,836.8 | [summary](results/hmmt_w1_raw_qk_k2048_hybrid4096_compact1000_shortshards_20261012/cells/hmmt_k2048_w1_raw_qk_compact1000_protected/formal/summary.json) |
| W1 (probability) | total K=2048 | full | 0 | protected | 52/120 | 43.33% | 1,270.4 | 55.6 | 20,854.1 | [summary](results/hmmt_full_w1_optimized_k2048_protected_20261009/summary.json) |
| W1 + random4096 (raw-QK) | total K=2048 | hybrid | 1000 | protected | 55/120 | 45.83% | 1,249.6 | 54.2 | 20,344.7 | [summary](results/hmmt_w1_raw_qk_k2048_hybrid4096_compact1000_shortshards_20261012/cells/hmmt_k2048_w1_raw_qk_hybrid1000_protected/formal/summary.json) |

<details>
<summary>Other complete HMMT controls: K=1024/2048/4096, Random/W1, protection and intervals</summary>

These settings are independent controls; their algorithms differ from the latest implementations.

### Protected total-K controls

| Method / score | Budget | KV storage | Interval | Protection | Correct | Accuracy | Decode s/question | Step ms | Output tokens | Source |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---|
| Shared Random (Uniform TopK) | total K=1024 | compact | 64 | protected | 16/120 | 13.33% | 1,222.0 | 41.4 | 28,397.8 | [summary](results/hmmt_k1024_k2048_full_compact_k4096_full_20261001/cells/hmmt_k1024_grid_compact_random_topk_protected/formal/summary.json) |
| W1 (probability) | total K=1024 | compact | 64 | protected | 15/120 | 12.50% | 1,207.5 | 41.5 | 28,427.0 | [summary](results/hmmt_k1024_k2048_full_compact_k4096_full_20261001/cells/hmmt_k1024_grid_compact_w1_topk_protected/formal/summary.json) |
| Shared Random (Uniform TopK) | total K=1024 | compact | 500 | protected | 23/120 | 19.17% | 1,183.7 | 40.9 | 27,813.6 | [summary](results/hmmt_compact_interval500_k1024_k2048_k4096_20261007/cells/hmmt_k1024_grid_compact500_random_topk_protected/formal/summary.json) |
| W1 (probability) | total K=1024 | compact | 500 | protected | 18/120 | 15.00% | 1,198.7 | 41.1 | 28,435.8 | [summary](results/hmmt_compact_interval500_k1024_k2048_k4096_20261007/cells/hmmt_k1024_grid_compact500_w1_topk_protected/formal/summary.json) |
| Shared Random (Uniform TopK) | total K=1024 | full | 0 | protected | 14/120 | 11.67% | 2,487.9 | 81.3 | 30,113.6 | [summary](results/hmmt_k1024_k2048_full_compact_k4096_full_20261001/cells/hmmt_k1024_grid_full_random_topk_protected/formal/summary.json) |
| W1 (probability) | total K=1024 | full | 0 | protected | 50/120 | 41.67% | 2,013.5 | 81.2 | 22,685.1 | [summary](results/hmmt_k1024_k2048_full_compact_k4096_full_20261001/cells/hmmt_k1024_grid_full_w1_topk_protected/formal/summary.json) |
| Shared Random (Uniform TopK) | total K=2048 | compact | 64 | protected | 40/120 | 33.33% | 1,064.7 | 41.3 | 24,063.1 | [summary](results/hmmt_k1024_k2048_full_compact_k4096_full_20261001/cells/hmmt_k2048_grid_compact_random_topk_protected/formal/summary.json) |
| W1 (probability) | total K=2048 | compact | 64 | protected | 37/120 | 30.83% | 1,077.1 | 41.6 | 24,214.4 | [summary](results/hmmt_k1024_k2048_full_compact_k4096_full_20261001/cells/hmmt_k2048_grid_compact_w1_topk_protected/formal/summary.json) |
| Shared Random (Uniform TopK) | total K=2048 | compact | 500 | protected | 40/120 | 33.33% | 1,047.0 | 41.4 | 23,397.6 | [summary](results/hmmt_compact_interval500_k1024_k2048_k4096_20261007/cells/hmmt_k2048_grid_compact500_random_topk_protected/formal/summary.json) |
| W1 (probability) | total K=2048 | compact | 500 | protected | 38/120 | 31.67% | 1,046.3 | 40.7 | 24,080.8 | [summary](results/hmmt_compact_interval500_k1024_k2048_k4096_20261007/cells/hmmt_k2048_grid_compact500_w1_topk_protected/formal/summary.json) |
| Shared Random (Uniform TopK) | total K=2048 | full | 0 | protected | 31/120 | 25.83% | 2,170.3 | 78.9 | 26,494.9 | [summary](results/hmmt_k1024_k2048_full_compact_k4096_full_20261001/cells/hmmt_k2048_grid_full_random_topk_protected/formal/summary.json) |
| Shared Random (Uniform TopK) | total K=4096 | compact | 64 | protected | 51/120 | 42.50% | 927.3 | 41.2 | 20,003.3 | [summary](results/shared_selector_grid_20260929/schedule_hmmt_first_20260930/reports/hmmt_compact_random_w1_four_cells/hmmt_grid_compact_random_topk_protected_summary.json) |
| W1 (probability) | total K=4096 | compact | 64 | protected | 51/120 | 42.50% | 959.7 | 41.3 | 21,208.5 | [summary](results/shared_selector_grid_20260929/schedule_hmmt_first_20260930/reports/hmmt_compact_random_w1_four_cells/hmmt_grid_compact_w1_topk_protected_summary.json) |
| Shared Random (Uniform TopK) | total K=4096 | compact | 500 | protected | 52/120 | 43.33% | 922.2 | 41.9 | 19,599.5 | [summary](results/hmmt_compact_interval500_k1024_k2048_k4096_20261007/cells/hmmt_k4096_grid_compact500_random_topk_protected/formal/summary.json) |
| W1 (probability) | total K=4096 | compact | 500 | protected | 48/120 | 40.00% | 942.0 | 40.9 | 20,911.9 | [summary](results/hmmt_compact_interval500_k1024_k2048_k4096_20261007/cells/hmmt_k4096_grid_compact500_w1_topk_protected/formal/summary.json) |
| Shared Random (Uniform TopK) | total K=4096 | full | 0 | protected | 44/120 | 36.67% | 1,943.2 | 76.2 | 23,401.8 | [summary](results/hmmt_k1024_k2048_full_compact_k4096_full_20261001/cells/hmmt_k4096_grid_full_random_topk_protected/formal/summary.json) |
| W1 (probability) | total K=4096 | full | 0 | protected | 51/120 | 42.50% | 1,706.8 | 76.2 | 19,709.1 | [summary](results/hmmt_k1024_k2048_full_compact_k4096_full_20261001/cells/hmmt_k4096_grid_full_w1_topk_protected/formal/summary.json) |

### No forced protection and older extra-prompt budget

| Method / score | Budget | KV storage | Interval | Protection | Correct | Accuracy | Decode s/question | Step ms | Output tokens | Source |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---|
| Shared Random (Uniform TopK) | prompt + reasoning1024 | compact | 64 | protected | 17/120 | 14.17% | 1,146.7 | 38.9 | 28,632.0 | [summary](results/compact_multibench_20260926/hmmt/formal/summary.json) |
| Shared Random (Uniform TopK) | total K=1024 | compact | 64 | none | 3/120 | 2.50% | 1,351.9 | 41.4 | 31,900.5 | [summary](results/hmmt_k1024_k2048_full_compact_k4096_full_20261001/cells/hmmt_k1024_grid_compact_random_topk_none/formal/summary.json) |
| W1 (probability) | total K=1024 | compact | 64 | none | 6/120 | 5.00% | 1,350.5 | 41.4 | 31,936.7 | [summary](results/hmmt_k1024_k2048_full_compact_k4096_full_20261001/cells/hmmt_k1024_grid_compact_w1_topk_none/formal/summary.json) |
| Shared Random (Uniform TopK) | total K=1024 | compact | 500 | none | 5/120 | 4.17% | 1,347.8 | 41.3 | 32,049.2 | [summary](results/hmmt_compact_interval500_k1024_k2048_k4096_20261007/cells/hmmt_k1024_grid_compact500_random_topk_none/formal/summary.json) |
| W1 (probability) | total K=1024 | compact | 500 | none | 4/120 | 3.33% | 1,366.2 | 42.1 | 31,896.8 | [summary](results/hmmt_compact_interval500_k1024_k2048_k4096_20261007/cells/hmmt_k1024_grid_compact500_w1_topk_none/formal/summary.json) |
| Shared Random (Uniform TopK) | total K=1024 | full | 0 | none | 0/120 | 0.00% | 2,596.0 | 79.5 | 32,641.5 | [summary](results/hmmt_k1024_k2048_full_compact_k4096_full_20261001/cells/hmmt_k1024_grid_full_random_topk_none/formal/summary.json) |
| W1 (probability) | total K=1024 | full | 0 | none | 24/120 | 20.00% | 1,957.2 | 76.9 | 23,631.2 | [summary](results/hmmt_k1024_k2048_full_compact_k4096_full_20261001/cells/hmmt_k1024_grid_full_w1_topk_none/formal/summary.json) |
| Shared Random (Uniform TopK) | total K=2048 | compact | 64 | none | 13/120 | 10.83% | 1,360.7 | 41.7 | 32,172.8 | [summary](results/hmmt_k1024_k2048_full_compact_k4096_full_20261001/cells/hmmt_k2048_grid_compact_random_topk_none/formal/summary.json) |
| W1 (probability) | total K=2048 | compact | 64 | none | 19/120 | 15.83% | 1,329.8 | 41.7 | 30,507.8 | [summary](results/hmmt_k1024_k2048_full_compact_k4096_full_20261001/cells/hmmt_k2048_grid_compact_w1_topk_none/formal/summary.json) |
| Shared Random (Uniform TopK) | total K=2048 | compact | 500 | none | 13/120 | 10.83% | 1,285.4 | 41.1 | 30,543.8 | [summary](results/hmmt_compact_interval500_k1024_k2048_k4096_20261007/cells/hmmt_k2048_grid_compact500_random_topk_none/formal/summary.json) |
| W1 (probability) | total K=2048 | compact | 500 | none | 19/120 | 15.83% | 1,268.7 | 41.1 | 29,406.7 | [summary](results/hmmt_compact_interval500_k1024_k2048_k4096_20261007/cells/hmmt_k2048_grid_compact500_w1_topk_none/formal/summary.json) |
| Shared Random (Uniform TopK) | total K=2048 | full | 0 | none | 1/120 | 0.83% | 2,542.9 | 77.9 | 32,641.5 | [summary](results/hmmt_k1024_k2048_full_compact_k4096_full_20261001/cells/hmmt_k2048_grid_full_random_topk_none/formal/summary.json) |
| W1 (probability) | total K=2048 | full | 0 | none | 37/120 | 30.83% | 1,789.3 | 75.8 | 21,655.9 | [summary](results/hmmt_k1024_k2048_full_compact_k4096_full_20261001/cells/hmmt_k2048_grid_full_w1_topk_none/formal/summary.json) |
| Shared Random (Uniform TopK) | total K=4096 | compact | 64 | none | 29/120 | 24.17% | 1,186.6 | 41.7 | 26,354.1 | [summary](results/shared_selector_grid_20260929/schedule_hmmt_first_20260930/reports/hmmt_compact_random_w1_four_cells/hmmt_grid_compact_random_topk_none_summary.json) |
| W1 (probability) | total K=4096 | compact | 64 | none | 41/120 | 34.17% | 1,131.7 | 41.7 | 25,302.8 | [summary](results/shared_selector_grid_20260929/schedule_hmmt_first_20260930/reports/hmmt_compact_random_w1_four_cells/hmmt_grid_compact_w1_topk_none_summary.json) |
| Shared Random (Uniform TopK) | total K=4096 | compact | 500 | none | 30/120 | 25.00% | 1,192.1 | 41.3 | 26,961.7 | [summary](results/hmmt_compact_interval500_k1024_k2048_k4096_20261007/cells/hmmt_k4096_grid_compact500_random_topk_none/formal/summary.json) |
| W1 (probability) | total K=4096 | compact | 500 | none | 41/120 | 34.17% | 1,127.8 | 41.4 | 24,762.4 | [summary](results/hmmt_compact_interval500_k1024_k2048_k4096_20261007/cells/hmmt_k4096_grid_compact500_w1_topk_none/formal/summary.json) |
| Shared Random (Uniform TopK) | total K=4096 | full | 0 | none | 10/120 | 8.33% | 2,445.7 | 77.0 | 30,824.5 | [summary](results/hmmt_k1024_k2048_full_compact_k4096_full_20261001/cells/hmmt_k4096_grid_full_random_topk_none/formal/summary.json) |
| W1 (probability) | total K=4096 | full | 0 | none | 48/120 | 40.00% | 1,641.6 | 74.5 | 20,000.2 | [summary](results/hmmt_k1024_k2048_full_compact_k4096_full_20261001/cells/hmmt_k4096_grid_full_w1_topk_none/formal/summary.json) |

</details>

## MATH500

| Method / score | Budget | KV storage | Interval | Protection | Correct | Accuracy | Decode s/question | Step ms | Output tokens | Source |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---|
| Shared Random (Uniform TopK) | 40% of reasoning | full | 0 | protected | 932/1000 | 93.20% | 432.6 | 58.5 | 6,499.8 | [summary](results/full_kv_random_r040_recent64_gpu_20260924/random/summary.json) |
| Shared Random (Bernoulli) | Bernoulli p=0.20 | full | 0 | protected | 923/1000 | 92.30% | 420.9 | 45.0 | 8,451.1 | [summary](results/full_kv_bernoulli_r020_recent64_recovery_20260929/random/summary.json) |
| W1 (probability) | TopP=0.95 uncapped | full | 0 | prompt only | 626/1000 | 62.60% | 1,202.4 | 91.9 | 11,240.5 | [summary](results/full_kv_w1_p095_20260924/formal/summary.json) |
| W16 (probability) | TopP=0.95 uncapped | full | 0 | prompt only | 875/1000 | 87.50% | 812.2 | 91.5 | 7,751.8 | [summary](results/full_kv_w16_w64_20260922/comparison_summary.json) |
| W64 (probability) | TopP=0.95 uncapped | full | 0 | prompt only | 877/1000 | 87.70% | 776.1 | 91.5 | 7,450.5 | [summary](results/full_kv_w16_w64_20260922/comparison_summary.json) |
| Dense (legacy) | all | full | 0 | all | 940/1000 | 94.00% | — | — | 5,340.1 | [summary](results/full_kv_random_r040_recent64_gpu_20260924/reference_baselines/dense/summary.json) |
| Shared Random (Uniform TopK) | prompt + reasoning1024 | compact | 64 | protected | 891/1000 | 89.10% | 301.8 | 39.1 | 6,742.0 | [summary](results/compact_kv_random_reason1024_recent64_20260926/random/summary.json) |
| W1 (probability) | prompt + reasoning1024 | compact | 64 | protected | 871/1000 | 87.10% | 313.0 | 39.6 | 7,057.5 | [summary](results/representative_w1_compact_math_20260926/math/formal/summary.json) |
| Shared Random (Uniform TopK) | prompt + reasoning1024 | full | 0 | protected | 854/1000 | 85.40% | 554.0 | 57.7 | 8,766.3 | [summary](results/full_kv_random_reason1024_memory_repair_20260928/random/summary.json) |
| Official random_pp (Uniform TopK) | total K=1024 | compact | 64 | protected | 872/1000 | 87.20% | 323.6 | 40.3 | 7,060.1 | [summary](results/official_random_pp_k1024_r64_20260924/formal/summary.json) |

Legacy Dense has no separated decode time. Historical batch-generation mean 284.8s is a different timing scope.

Random full-KV reasoning1024: 854/1000 comes from 429 historical + 71 repaired inputs; its mixed mean 554.0s is not a fresh full500 repaired-speed benchmark. The fresh `repaired_full500` rerun failed on shard 4 and is excluded.

## Other complete dataset splits

| Method / score | Budget | KV storage | Interval | Protection | Correct | Accuracy | Decode s/question | Step ms | Output tokens | Source |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---|
| Shared Random (Uniform TopK) | prompt + reasoning1024 | compact | 64 | protected | 16/60 | 26.67% | 1,079.4 | 40.0 | 26,234.8 | [summary](results/compact_multibench_20260926/aime25/formal/summary.json) |
| Shared Random (Uniform TopK) | prompt + reasoning1024 | compact | 64 | protected | 16/60 | 26.67% | 1,054.3 | 38.1 | 26,232.0 | [summary](results/compact_multibench_20260926/aime26/formal/summary.json) |
| Shared Random (Uniform TopK) | prompt + reasoning1024 | compact | 64 | protected | 184/396 | 46.46% | 677.5 | 39.1 | 15,373.3 | [summary](results/compact_multibench_20260926/gpqa/formal/summary.json) |
| Shared Random (Uniform TopK) | prompt + reasoning1024 | compact | 64 | protected | 351/766 | 45.82% | 902.8 | 38.7 | 21,879.6 | [summary](results/compact_multibench_20260926/livecodebench/formal/summary.json) |

## Validation-only optimizations and historical evidence

The v3 fused raw-QK full/hybrid implementation has passed GPU correctness and same-device single-layer timing checks at K1024/K2048. It has no new full-HMMT result and does not replace a formal accuracy/timing row. See [v3 validation](W1_RAW_QK_FUSED_OPTIMIZATION_20261014.md). Run inventory classifies stored evidence; use the live scheduler for current job status.
Pilot W8, probe, length40960, profiler and failed/partial outputs are outside the full-dataset tables. Older valid algorithms remain independent controls; a higher single-seed score alone is not a replacement criterion.
[Run inventory](results/current/run_inventory.json) · [archive index](results/archive/README.md) · [CSV](results/current/results.csv) · [source-backed JSON](results/current/summary.json) · [validation](results/current/validation.json)

Scope excludes the separate Llama prefill/KV-retrieval and J-lens studies; those files remain independent. Older dated PDF/TeX reports are historical snapshots; use this entry point for current reasoning results.

Rebuild or audit from this experiment directory:

```bash
python scripts/consolidate_results.py
python scripts/consolidate_results.py --check
```
