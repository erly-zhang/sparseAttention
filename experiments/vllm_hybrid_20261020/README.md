# Hybrid W1 in vLLM 0.19 (TriAttention runtime)

Deployment, throughput and accuracy-transfer study of the hybrid W1 selector, with Dense,
random_pp, TriAttention and our port of SnapKV / R-KV / VaSE in the same vLLM runtime.
Run outputs (`results*/`), job logs and tokenised inputs stay on the cluster.

| Path | Content |
|---|---|
| `src/hybrid_attention.py` | Triton decode path: representative-head raw-QK scoring over the paged pool, exact chronological selection, split-K sparse attention; shared (default) or per-KV-head selection (`HYBRID_SHARED=0`, share ablation A1) |
| `src/selector_hybrid.py` | pool-compaction selector (W1 core + prompt + newer slots + Uniform fill), shared or per head |
| `src/baselines_vllm.py`, `src/baseline_scoring.py` | SnapKV / R-KV / VaSE port: query capture in the attention forward + upstream selection code |
| `src/apply_runtime_fix.py` | TriAttention runtime fix (row remap after `_update_states`), row-capacity check, stable seeds |
| `src/apply_hybrid_edit.py`, `src/apply_baselines_edit.py` | opt-in selector dispatch in a copy of the runtime |
| `src/tests/` | kernel, selector, share-ablation and upstream-equivalence tests |
| `bench_throughput.py`, `vllm_task_acc.py` | throughput benchmark; accuracy generation / prep / grading |
| `frozen_v2` ... `frozen_v5` | frozen scripts, environments and SHA-256 manifests of each experiment batch (v2 main grid, v3 baseline port, v4 share ablation, v5 random_pp cadence control) |
| `archive/` | superseded code versions referenced by earlier runs |

Reports: `report_build/vllm_throughput_20261020/` and `report_build/results_summary_20261020/` (outside this repository).
