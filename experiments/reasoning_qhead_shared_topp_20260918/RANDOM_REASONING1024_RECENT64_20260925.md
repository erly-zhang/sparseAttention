# Prompt + 1024 reasoning positions: random shared decode mask

User-requested budget: preserve the entire prompt separately, allow up to 1024 reasoning positions in each current-step attention mask, with recent64 INSIDE the reasoning budget. This is not the official random_pp total1024 budget.

Method: `qhead_shared_mask_random_fixed_recent_gpu`.

For prompt length P and full non-prompt history R, including this step's appended input key:

```
K = min(1024, R)
L = min(64, K)
M = all P prompt positions
    union the newest L reasoning positions
    union a uniform subset of size K-L from the older R-L reasoning positions
|M| = P + min(1024, R)
```

Prefill is dense. Decode appends full post-RoPE K/V, constructs the temporary mask BEFORE current attention, and preserves all historical K/V for future reselection. Short reasoning histories R<=1024 use every legal key. Each layer draws independently; the positions are shared by all Q heads in that layer. Own-head/GQA attention normalization is unchanged. There is no probability Top-p selector and no 40% ratio.

Only the reasoning keep-count rule changes relative to `RandomRecentSharedMaskLayer`: `ceil(.4*R)` becomes `min(1024,R)`. Its GPU primary/secondary random priorities, original-position final tie-break, seed42/sample/layer/step/original-rollout coordinates, original-position sorting, fused K/V gather and reusable temporary buffers are reused. The nominal representative-head metadata has no influence on uniform mask selection. Complete historical KV memory remains proportional to full sequence length; this is not permanent KV compression.

New CLI:

```bash
--eviction_mode qhead_shared_mask_random_fixed_recent_gpu \
--token_budget 0 --residual_length 0 --eviction_interval 1 \
--reasoning_budget 1024 --recent_window 64 --mask_seed 42
```

`token_budget=0` is the full-history-storage compatibility argument; `reasoning_budget=1024` is the actual non-prompt attention budget. These are deliberately distinct.

Correctness tests cover exact budgets around1024, full prompt protection including prompts>1024, recent coverage, uniqueness, short-history dense equivalence, full-KV preservation and reselection, private RNG/batch filtering, exact sampler parity with random40 when its count is also1024, BF16 GQA attention versus independent reference, reused gather buffers and exact logical pair counters. Existing shared-mask regression tests also run before generation. GPU-only tests are skipped on CPU and required on the allocated GPU.

Frozen trial: `results/full_kv_random_reason1024_recent64_smoke_20260925/`.
Job1342869: one A100, independent Slurm allocation. Scope: MATH500 IDs0,1,2; two complete generations each; same chat inputs/hashes, Qwen3-4B BF16, seed42, temperature0.6, generation Top-p0.95, max total length32768, untimed2-token warmup. No full500 automatic submission. Old results/frozen runtimes are untouched; pre-edit runtime copies are in `snapshots/before_random_fixed1024_20260925/`.

The validator checks input identity, exact output coverage, EOS/cap, full-cache length, and independent selected/causal pair counts. At reasoning step j, each Q head uses `P+min(1024,j)` keys. Thus decode selected pairs are independently reconstructed from observed lengths, without trusting a mean sparsity percentage. Timed decode includes sampling, selection, gather and attention, and excludes loading/warmup/diagnostic serialization.

Progress: `pipeline_status.json`, `correctness.log`, `smoke/runner.log`.
Final validation: `smoke/summary.json`, `smoke/grades.jsonl`, `smoke_gate.json`.

GPU correctness gate: all 33 shared-mask tests passed on 2026-09-25, including 4 new fixed-reasoning tests. The trial completed at 22:05:35 EDT; the smoke gate passed. Input hashes, exact pair budgets, output coverage and full-history continuity passed.


## Completed trial results

Qwen3-4B, MATH500 input IDs 0, 1, 2, two complete rollouts each. All three methods below used identical input token counts and SHA256 values. This is a six-generation trial, not the full MATH500 evaluation.

| Method | Correct | EOS | Mean generated tokens | Mean decode seconds/question batch | Decode tokens/s | Global pair sparsity | Peak allocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Prompt + reasoning1024, recent64 inside budget | 6/6 | 6/6 | 3329.2 | 206.293 | 32.266 | 56.91% | 11.937 |
| Random 40% + recent64 | 6/6 | 6/6 | 3026.0 | 194.555 | 31.097 | 56.82% | 10.335 |
| Official random_pp | 6/6 | 6/6 | 2578.5 | 115.186 | 44.754 | 43.38% | 7.821 |

Each question batch contains two concurrent rollouts. Timing excludes loading, warmup and metric serialization. Same A100 80GB GPU class, different nodes and run times. Logical pair sparsity includes dense prefill and follows each method's own generated trajectory; it is not memory compression or a matched-length comparison.

The new method generated about 10.0% more tokens and took about 6.0% longer per question batch than random40 on these inputs. These observations do not establish a general accuracy or speed advantage. The 1024 cap excludes prompt, differs from official random_pp's total budget, and retains all historical KV. No full500 run was submitted automatically.

Raw new-method counts: selected pairs 22,417,303,680; causal pairs 52,019,964,288. Mean prefill 0.0401 s; mean total question-batch time 206.402 s. No rollout hit the length cap.

Artifacts: [comparison report](results/full_kv_random_reason1024_recent64_smoke_20260925/smoke/REPORT.md), [comparison JSON](results/full_kv_random_reason1024_recent64_smoke_20260925/smoke/comparison.json), [validation summary](results/full_kv_random_reason1024_recent64_smoke_20260925/smoke/summary.json). Source hashes are recorded in comparison_sources.json and the frozen runtime manifest.


## Authorized full MATH500 run

The user authorized the full run after reviewing the completed smoke. Submitted 2026-09-25 23:15:36 EDT: array job `1343213`, CPU collector `1343214` with `afterany` dependency. Independent root: `results/full_kv_random_reason1024_recent64_20260925/`. Eight shards, one A100 80GB per worker, at most eight workers, 48-hour worker limit; 500 questions and 1000 completions. Inference/tests are byte-identical to the passed smoke. Prompt protection, reasoning1024 including recent64, full persistent KV and all generation settings are unchanged.

Full-run reporting was validated against the six completed smoke outputs before submission. All 500 IDs are assigned exactly once across eight shards. The collector verifies successful workers, input alignment, exact pair counts and cache continuity, then writes accuracy, termination, timing, memory and sparsity summaries. Frozen random40 recent64 and official random_pp baselines are included for paired accuracy and input-hash checks, along with historical W16/dense references. Incomplete or failed workers prevent publication of a full500 accuracy result.

Submission details: [submission.json](results/full_kv_random_reason1024_recent64_20260925/submission.json). Monitor `random/shard_*/preflight.json`, `runner.log`, `exit.json`, and the final `collection_status.json`. The earlier statement about no automatic full500 submission describes the smoke phase; this full run follows explicit user authorization.
