# SparseAttention

SparseAttention contains the experiment code for aligned LongBench-v2 sparse
attention comparisons on Qwen2.5-7B. The repository is now focused on the
current Stage-2 evaluation pipeline rather than the earlier attention-map
exploration scripts.

## What Is Included

- `experiments/run_shared_layer_mask_experiment.py`
  - Single-Cluster sparse mask runner.
  - Coverage-selected, random, and fixed representative-head variants.
  - Four prefill/decode modes: `ff`, `fs`, `sf`, and `ss`.
- `experiments/run_graph2vec_cluster_shared_mask_experiment.py`
  - Multi-cluster sparse mask runner.
  - Supports `graph2vec`, `svd_kmeans`, and `bmm` head clustering.
- `experiments/official_sparse_baselines/`
  - Official-runtime baseline harness for supported methods such as
    MInference and FlexPrefill.
- `experiments/build_longbench_v2_32k_domain_sample.py`
  - Builds the aligned LongBench-v2 split used by the current experiments.
- `experiments/SCRIPTS.md`
  - Detailed command reference for active scripts.

## Evaluation Setup

The canonical Stage-2 comparison uses:

- Qwen2.5-7B
- LongBench-v2 multiple-choice samples
- 32k input budget
- 3 head-selection samples
- 500 aligned task-evaluation samples
- greedy generation with `max_new_tokens=8`

Mode names:

| Mode | Prefill | Decode |
| --- | --- | --- |
| `ff` | full attention | full attention |
| `fs` | full attention | sparse attention |
| `sf` | sparse attention | full attention |
| `ss` | sparse attention | sparse attention |

## Quick Start

Install dependencies in an environment with the correct CUDA-enabled PyTorch:

```bash
pip install -r experiments/requirements.txt
```

Build/reuse the aligned 32k split and run one comparison cell:

```bash
METHOD=single_cluster TOP_P=0.85 bash experiments/run_comparison_sweep_32k.sh
```

Run the full configured 32k sweep:

```bash
nohup bash experiments/run_comparison_sweep_32k.sh \
  > experiments/outputs/comparison32k_sweep.log 2>&1 &
```

Run an official baseline:

```bash
python experiments/official_sparse_baselines/run_official_sparse_baseline.py \
  --method minference \
  --output_dir experiments/outputs/official_minference_500eval
```

For more commands and environment variables, see
[`experiments/SCRIPTS.md`](experiments/SCRIPTS.md).

## Repository Layout

```text
experiments/
  README.md
  SCRIPTS.md
  build_longbench_v2_32k_domain_sample.py
  run_shared_layer_mask_experiment.py
  run_graph2vec_cluster_shared_mask_experiment.py
  run_comparison_sweep_32k.sh
  run_full_longbench_128k_pipeline.sh
  official_sparse_baselines/
```

## Validation

Before committing code changes:

```bash
python -m compileall experiments
bash -n experiments/run_comparison_sweep_32k.sh \
  experiments/run_full_longbench_128k_pipeline.sh \
  experiments/run_random_single_cluster_500eval.sh
```

