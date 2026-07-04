# Experiment Shell Scripts

## Active scripts (4)

| Script | Role |
|--------|------|
| `run_task_eval_10samples.sh` | **Single-Cluster** core runner (head selection + ff/sf/fs/ss eval) |
| `run_graph2vec_cluster_task_eval50_7b.sh` | **Graph2Vec 2-Cluster** core runner |
| `run_both_eval_sequential.sh` | Orchestrator: Single → Graph2Vec, **32k** default, shared dataset |
| `run_full_longbench_128k_pipeline.sh` | Orchestrator: build data + Single → Graph2Vec, **128k full** LongBench-v2 |

## Typical usage

```bash
# 32k, full LongBench-v2 (503 records)
bash experiments/run_both_eval_sequential.sh

# 32k, partial (50 eval samples)
EVAL_N=50 USE_ALL_RECORDS=false bash experiments/run_both_eval_sequential.sh

# 128k, full LongBench-v2 (truncate only if prompt > 128k)
nohup bash experiments/run_full_longbench_128k_pipeline.sh \
  > experiments/outputs/run_full_longbench_128k_pipeline.log 2>&1 &

# Run only one method (128k, reuse built data)
SKIP_DATA_BUILD=true SKIP_GRAPH2VEC=true bash experiments/run_full_longbench_128k_pipeline.sh
```

## Common env vars

| Variable | Default (32k orchestrator) | 128k pipeline |
|----------|---------------------------|---------------|
| `EVAL_N` | 500 | auto from `--use_all_records` |
| `USE_ALL_RECORDS` | true | true |
| `MAX_TOTAL_TOKENS` | 32768 | 131072 |
| `SKIP_DATA_BUILD` | false / true (2nd method) | false |
| `EVAL_MODE_COMBOS` | ff,sf,fs,ss | ff,sf,fs,ss |

## Data builder

Dataset construction is inlined via `experiments/build_longbench_v2_32k_domain_sample.py` (called from core/orchestrator scripts). No separate data-only shell script.

## Removed (superseded)

- `run_graph2vec_task_eval_50samples.sh` → use `run_graph2vec_cluster_task_eval50_7b.sh`
- `run_full_longbench_v2_128k_both_eval.sh` → use `run_full_longbench_128k_pipeline.sh`
- `build_full_longbench_v2_128k_data_only.sh` → inlined in 128k pipeline
- `run_graph2vec_cluster_task_eval_debug.sh` → use `DEBUG_LAYERS=0,1` on core script
- `run_shared_layer_mask_32k_3domain.sh` → early 3B analysis; use core + env
- `run_shared_layer_mask_debug.sh` → early smoke test; use `EVAL_N=1 MAX_INPUT_LENGTH=8192`
