# Further raw-QK W1 optimization (2026-10-14)

This review covers full-history W1 and the bounded W1 + random4096 candidate
pool. Both retain the existing single-query BF16 scoring, representative stream,
GQA mapping, chronological ties, prompt+recent64 protection inside total K,
attention normalization, absolute positions and generation settings.

## Implemented candidate

`ra_harness/kvcompress/engine/w1_qk_fused_cache.py` is an explicit v3 opt-in.
It combines chronological index assembly, K/V gathering and selected-length
updates in one Triton launch. The selected middle indices come from the same
v2 `exact_qk_topk`; the QK matmul expression is unchanged. All retained K/V
values are copied without arithmetic. A gather-ready marker prevents a second
copy in the inherited attention path and is invalidated on batch shrink.
The stable-sort path also avoids allocating an unused packed-key workspace.

The v2 source files, runners and historical results are unchanged. The new
runner is `scripts/run_w1_qk_fused.py`; `RA_W1_QK_FAST_BACKEND=stable_sort` matches
the latest formal full-history backend and `packed_qk_topk` matches the latest
formal bounded-candidate backend. The new implementation is not the default.

## Remaining costs

| Operation | Frequency | Assessment |
|---|---|---|
| Representative draw and Python/PyTorch dispatch | Every layer and decode step | Additional overhead; any schedule precomputation must preserve the exact stream and include preparation cost in timing. |
| Raw-QK scoring | Every layer and decode step after K is exceeded | Full history scores all historical keys; hybrid scores the live K+4096 pool plus newly appended keys. This is required by the present selector. |
| Score ranking and chronological ordering | Every layer and decode step after K is exceeded | Further exact selection kernels may help; BF16 ties, signed zero and the existing backend's NaN semantics must be preserved. |
| Index assembly and K/V gather | Every layer and decode step | Implemented fusion removes one launch and an intermediate index read between assembly and gathering. |
| Uniform residual selection and physical compaction | Every 1000 decode steps, when the pool exceeds its budget | Lower priority for total runtime than per-step work; changing the interval or candidate law changes the algorithm. |
| Projections, MLP and output sampling | Every decode step | Shared with Dense; these computations are not reduced by selecting fewer attention keys. |

Fusing/recomputing QK or replacing FlashAttention with indexed attention is a
separate numerical change unless exact scores, masks and sampled trajectories
are verified. It is not included in this candidate.

## Validation

`tests/test_w1_qk_fused.py` compares v2/v3 with the real 32-Q/8-KV GQA geometry,
K1024/K2048, short histories, a zero-sized selectable middle, all-score ties,
batch shrink, two physical compactions, private/global RNG and bitwise attention.
It also compares complete sampled token sequences on a controlled tiny Qwen3.

The frozen source, correctness log and same-device timings are under
`/mmfs1/project/yl258/jz648/.codex-persist/w1_qk_fused_20261014_validationfix/`.
The first snapshot is retained separately; static review caught redundant query
indexing in its new batch-shrink test, so that validation job was canceled and
the test corrected before the replacement run. The inference source did not change.

Status: completed. Validation job 1406993 finished with exit0 on n0001, using an
A100-SXM4-80GB MIG 2g.20gb slice. All four test methods passed without skips;
selection/KV/attention, private and global RNG, batch shrink, two physical pool
updates and the tiny-model sampled token trajectories matched v2.

Same-device synchronized wall-clock medians, six alternating rounds of80
repetitions with warmup excluded, batch2,32 Q heads,8 KV heads, head dimension128:

| Storage | K | Live candidate count N | v2 layer ms | v3 layer ms | Reduction |
|---|---:|---:|---:|---:|---:|
| Full | 1024 | 1024 | 0.1700 | 0.1475 | 13.24% |
| Full | 1024 | 8192 | 0.3984 | 0.3734 | 6.28% |
| Full | 1024 | 32700 | 0.3983 | 0.3744 | 6.01% |
| Full | 2048 | 32700 | 0.3995 | 0.3761 | 5.84% |
| Hybrid | 1024 | 5120 | 0.3779 | 0.3531 | 6.56% |
| Hybrid | 1024 | 6120 | 0.3754 | 0.3521 | 6.19% |
| Hybrid | 2048 | 6144 | 0.3787 | 0.3561 | 5.96% |
| Hybrid | 2048 | 7144 | 0.3789 | 0.3553 | 6.24% |

The measured layer includes the CPU representative draw, raw-QK selection,
chronological index handling, KV append/gather and FlashAttention. It excludes
MLP, projections, output sampling and physical compaction events; compaction
correctness is tested separately. All benchmark masks and attention outputs
were bitwise equal. The N=K short-history row does not perform QK ranking.

No new full-Qwen3-4B or full-HMMT generation experiment was submitted. These
single-layer improvements are not measured full-model speedups and do not
provide new HMMT accuracy, output-length or peak-memory results. Historical
formal results remain the current accuracy references. Inspect `completion.json`,
`correctness.json`, `correctness.log`, `microbenchmark.json` and the source manifest
in the frozen validation directory for the full evidence.
