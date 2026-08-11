# Sparse Attention Experiments

This directory contains the project-specific LongBench-v2 sparse-attention
experiments used for the aligned Stage-2 comparison.

The code is designed around one canonical evaluation split:

- 3 head-selection samples, used only to choose representative heads.
- 500 task-evaluation samples, used for accuracy, sparsity, and PPL reporting.
- Qwen2.5-7B with a 32k input budget unless a script explicitly overrides it.

Large artifacts are intentionally kept out of version control. In particular,
`data/`, `outputs/`, model checkpoints, logs, and backup files should not be
committed.

## Canonical InfiniteBench and AutoBlock workflow

- `run_shareprefill_ae3_infinitebench.py`: main SharePrefill-AE InfiniteBench
  runner, including Full, Compact, HISA-style, and token-first automatic-block
  variants.
- `official_sparse_baselines/run_official_sparse_baseline_benchmark.py`:
  aligned FlexPrefill and MInference full-generation runner.
- `baseline_sparsity.py`: final-kernel causal token-pair accounting shared by
  baseline profiling runs.
- `aggregate_infinitebench_methods.py`: canonical cross-method InfiniteBench
  aggregation and comparison entry point.
- `run_shareprefill_autoencoder_clustering.py`: attention-map autoencoder and
  offline representative-head grouping.
- `token_compacted_sparse.py`: token compaction, HISA-style routing, AutoBlock
  projection, Triton kernels, and detailed selector metrics.

The older Stage-2 runners below remain available for reproducing the historical
LongBench-v2 experiments.

## Main Methods

| Method | Entry point | Notes |
| --- | --- | --- |
| Single-Cluster | `run_shared_layer_mask_experiment.py` | One representative head and one shared mask per layer. |
| Random Single-Cluster | `run_shared_layer_mask_experiment.py` | Same mask pipeline, but representative heads are random and seed-controlled. |
| Graph2Vec / SVD-KMeans / BMM clusters | `run_graph2vec_cluster_shared_mask_experiment.py` | Multi-cluster head grouping with per-head mask routing. |
| Official sparse baselines | `official_sparse_baselines/run_official_sparse_baseline.py` | Official MInference/FlexPrefill runtime path where supported. |

## Evaluation Modes

The unified runner reports four prefill/decode combinations:

| Key | Prefill | Decode |
| --- | --- | --- |
| `ff` | full attention | full attention |
| `fs` | full attention | sparse attention |
| `sf` | sparse attention | full attention |
| `ss` | sparse attention | sparse attention |

Generation is greedy by default (`do_sample=false`) with `max_new_tokens=8` for
the aligned Stage-2 runs.

## Typical Commands

Build or reuse the canonical 32k LongBench-v2 split and run one sweep cell:

```bash
METHOD=single_cluster TOP_P=0.85 bash experiments/run_comparison_sweep_32k.sh
```

Run the full comparison sweep:

```bash
nohup bash experiments/run_comparison_sweep_32k.sh \
  > experiments/outputs/comparison32k_sweep.log 2>&1 &
```

Run official sparse baselines:

```bash
python experiments/official_sparse_baselines/run_official_sparse_baseline.py \
  --method minference \
  --output_dir experiments/outputs/official_minference_500eval
```

## Repository Hygiene

Before publishing, check:

```bash
python -m compileall experiments
find experiments -type f -name '*.bak_*' -print
```
