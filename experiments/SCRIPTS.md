# Experiment Scripts

This file documents the scripts that currently exist in `experiments/`.
Generated outputs, logs, JSONL data, and `.bak_*` files are intentionally
excluded from GitHub by `.gitignore`.

## Active Entrypoints

| Script | Role |
| --- | --- |
| `build_longbench_v2_32k_domain_sample.py` | Build the aligned LongBench-v2 split: 3 head-selection samples plus task-eval samples. |
| `run_shared_layer_mask_experiment.py` | Unified Single-Cluster runner, including coverage, random, survey-fixed, top-p, top-k, adaptive-top-p, and fixed top-ratio variants. |
| `run_graph2vec_cluster_shared_mask_experiment.py` | Multi-cluster runner for `graph2vec`, `svd_kmeans`, and `bmm` head clustering. |
| `run_comparison_sweep_32k.sh` | 32k sweep orchestrator for Single-Cluster and cluster-based methods. |
| `run_full_longbench_128k_pipeline.sh` | LongBench-v2 pipeline with configurable token budget; defaults to 32k for faster smoke tests. |
| `run_random_single_cluster_500eval.sh` | Convenience wrapper for the random representative-head baseline. |
| `official_sparse_baselines/run_official_sparse_baseline.py` | Official-runtime MInference/FlexPrefill runner where Qwen2.5 support is available. |
| `official_sparse_baselines/run_faithfulness_stage1.py` | Stage-1 check comparing official generation and unified generation on the same patched model. |

## Common Environment Variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `MODEL_PATH` | `/home/ubuntu/work/model/Qwen2.5-7B` | Local model checkpoint. |
| `DATA_OUT` | `experiments/data/longbench_v2_32k_full_7b.jsonl` | Aligned dataset path. |
| `EXP_OUT` / `SWEEP_ROOT` | under `experiments/outputs/` | Output directory. |
| `HEAD_N` | `3` | Head-selection sample count. |
| `EVAL_N` | read from selection metadata | Task-eval sample count. |
| `MAX_TOTAL_TOKENS` | `32768` | Dataset token budget. |
| `MAX_INPUT_LENGTH` | `MAX_TOTAL_TOKENS` | Runtime prompt truncation budget. |
| `CHUNK_SIZE` | `2048` | Chunked prefill size for long contexts. |
| `EVAL_MODE_COMBOS` | `ff,sf,fs,ss` | Prefill/decode mode combinations. |
| `SKIP_DATA_BUILD` | `false` | Reuse an existing aligned split. |

## Canonical 32k Examples

Run one method/top-p cell:

```bash
METHOD=single_cluster TOP_P=0.85 bash experiments/run_comparison_sweep_32k.sh
METHOD=svd_kmeans TOP_P=0.85 SKIP_DATA_BUILD=true bash experiments/run_comparison_sweep_32k.sh
```

Run the full configured sweep:

```bash
nohup bash experiments/run_comparison_sweep_32k.sh \
  > experiments/outputs/comparison32k_sweep.log 2>&1 &
```

Run random representative-head baseline:

```bash
bash experiments/run_random_single_cluster_500eval.sh
```

## Official Baselines

The official sparse-baseline runner intentionally separates official runtime
pipelines from local proxy methods.

```bash
python experiments/official_sparse_baselines/run_official_sparse_baseline.py \
  --method minference \
  --output_dir experiments/outputs/official_minference_500eval

python experiments/official_sparse_baselines/run_official_sparse_baseline.py \
  --method flexprefill \
  --output_dir experiments/outputs/official_flexprefill_500eval
```

Unsupported official Qwen2.5 methods write a manifest explaining why they were
not run rather than silently falling back to a proxy implementation.

## Output Contract

Most runners write:

- `run.log`
- `experiment_manifest.json` or `manifest.json`
- per-sample JSON under `task_eval/` or `eval/`
- `task_eval_summary.json`, `eval_summary.json`, or `summary.json`

Keep raw output directories out of GitHub unless a result has been curated and
explicitly approved for publication.
