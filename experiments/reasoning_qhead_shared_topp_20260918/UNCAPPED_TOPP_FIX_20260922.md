> Historical permanent-eviction trial: this does NOT implement the user-requested temporary sparse-attention mask. Superseded by `FULL_KV_SHARED_MASK_FIX_20260922.md`; its 90.43% result must not be attributed to the corrected method.

# Shared Q-head Top-p cache repair and uncapped pilot (2026-09-22)

This change concerns Qwen3-4B / MATH500 reasoning-cache eviction. It does not change the older Llama KV Retrieval experiment. Existing formal outputs and the frozen Protocol B runtime are preserved. The redundant pre-edit copy formerly at `snapshots/before_uncapped_fix_20260922/` was removed during the approved2026-10-11 cleanup; all four files were byte-identical to the retained `snapshots/protocolB_20260919/` source snapshot.

## Repairs

- Each batch row uses its own valid KV length. Positions beyond that length are masked before representative-head softmax. A positive legacy token threshold is also tested separately for each row.
- Each row's unique selected indices are sorted before an independent gather/copy. No padded index vector enters the effective cache. The prompt remains a contiguous protected prefix; the same positions are gathered for every KV head.
- `prepare()` grows the shared-method cache before FlashAttention writes new keys. Capacity is checked against the longest active row plus the incoming tokens. Geometric storage growth is an allocation strategy, not a token-retention cap.
- Shared Top-p bypasses the discarded all-head QK calculation. Only the sampled representative Q head is scored.
- Short prompts use the actual populated query-buffer entries (at most 64), and a one-token prompt initializes that buffer correctly. Position tracking uses each row's insertion offset. Original cumulative length continues to drive RoPE.
- Logs carry original rollout identity through batch shrinking. Debug checks cover every row, prompt retention, unique chronological indices, and selected mass. Non-finite valid scores fail explicitly; complete reasoning-score underflow conservatively retains all candidates.

## New trial configuration

`--eviction_mode qhead_shared_topp --top_p 0.95 --token_budget 0 --residual_length 0 --eviction_interval 1`

Here `token_budget=0` disables the previous 1024 token threshold and fixed allocation. Every decode attention call first appends and attends over the valid cache, then selects the cache for the next call. Prompt KV is always kept. The number of reasoning positions is the smallest descending-score prefix whose normalized reasoning mass reaches 0.95; there is no 1024 ceiling. `eviction_interval` can independently specify a different cadence in uncapped mode.

The representative RNG remains sample-local (representative seed 0, sample ID, layer), and generation seed remains 42. GQA mapping, post-RoPE scoring, the probability-mean selector and own-head attention are retained. Historical queries in the selector still score the current candidate cache without a historical-query causal mask, matching the previous selector; normal attention remains causal. Selection is persistent eviction: already removed reasoning KV cannot be recovered.

More frequent selection changes the algorithm in addition to changing p. Thus p=0.95 here does **not** imply a larger final cache than the old p=0.9 threshold-triggered method. It also does not imply a 5% token sparsity. Generation sampling temperature/top-p are inherited from the unchanged model generation config; attention-selection p is a separate setting.

## Tests and pilot scope

`tests/test_shared_topp_cache.py` contains seven CPU/CUDA regression tests: independent scalar selector agreement, ragged rows in both length orders, NaN stale slots, exact padding-corruption reproduction, growth/prompt preservation beyond 1024, p=0.95 selecting more than 1024 reasoning tokens, row-specific legacy thresholds, batch shrinking, short prompts, selection cadence, pair counts, and FlashAttention GQA append/attention against an independent numerical reference.

The retired `qhead_topp_uncapped_trial.sh` launcher ran only MATH500 IDs0,1,2, with two rollouts each, on an already allocated idle GPU. It rejected existing output directories and froze source plus a SHA-256 manifest before running. A256-total-token integration smoke was completed; a separate2048-total-token trial also completed sustained decoding. These were bounded engineering trials, not full-answer accuracy experiments. The2048 limit was on total generated sequence length, not KV retention. The canonical trial launcher was removed during the approved2026-10-11 cleanup; historical outputs and frozen runtimes remain local. Use the current experiment guide for maintained launch workflows.

The launcher isolates regression telemetry from model telemetry. The original 256-token smoke preceded that launcher fix: its raw file includes 25 synthetic test records (sample IDs 5 or null). The summary explicitly excludes those records; original raw data remain intact. The subsequent 2048-token trial uses separate test logging.

## Metrics

`cache_metrics.jsonl` saves input-token length and little-endian int64 input-ID SHA-256, explicit EOS flags (in the 2048-token trial), actual logical selected/causal pair totals, maximum active KV length, allocated capacity, synchronized batch wall time, and peak allocated GPU memory. Pair totals cover all layers, all active rollouts, prefill and every decode step:

For one attention call appending m tokens, with row b's previous retained length L_b and the full original history length T,

```
selected_pairs += Hq * sum_b [m * L_b + m*(m+1)/2]
causal_pairs   += Hq * number_of_active_rows * [m * T + m*(m+1)/2]
global_pair_sparsity = 1 - sum(selected_pairs) / sum(causal_pairs)
```

These count the logical causal attention pairs actually visible in the compact cache, not padded kernel FLOPs or selector QK work. The full-history denominator uses the same realized trajectories and stopping points; it is not the separate dense model's generated trajectory. Allocated memory can exceed active cache occupancy.

`scripts/summarize_uncapped_trial.py` independently reconstructs prompts, verifies all trial input hashes and both rollout coverages, validates the source manifest, checks per-row length continuity across selections, and sums pair counts before taking ratios. Event-local pruning percentages and full-history pair sparsity are distinct metrics. Timing includes diagnostics and per-event JSON serialization, so it is not an optimized throughput comparison. The old reported 61.92% was a requested post-eviction event ratio from buggy runs and cannot serve as a controlled baseline for the new full-history pair ratio.

## Artifacts

- Short smoke: `results/uncapped_p095_smoke_20260922/summary.json`
- Longer trial: `results/uncapped_p095_trial2048_20260922/`
- Longer-trial final summary: `results/uncapped_p095_trial2048_20260922/summary.json`; exit code 0 and validation passed.
- No 500-question evaluation or additional GPU job was submitted. Existing dense jobs were left running.

## Completed 2048-token trial

All seven regression tests passed on the allocated A100. All three inputs and both rollouts completed the bounded run, with full trial input-token/hash alignment, source-manifest validation, and per-row length continuity passing. Independent reconstruction from 422,352 raw selection records exactly matches runtime pair counters. No synthetic regression records entered this trial's telemetry. All six 256-token prefixes reproduce the earlier short smoke.

| Measurement | Result |
| --- | ---: |
| Selected logical attention pairs | 1,386,634,432 |
| Full-history causal pairs on the same trajectories | 14,488,436,736 |
| Global pair sparsity | 90.4294% |
| Event-local prompt + reasoning pruning ratio | 0.9935% |
| Event-local reasoning pruning ratio | 11.0756% |
| Maximum active KV length across layers/rows | 154 |
| Minimum retained reasoning mass | 0.949999988 (float32 rounding) |
| Peak allocated GPU memory, including model | 8,412,720,128 bytes |
| Sum of synchronized batch wall times | 613.35 seconds |
| Rollouts reaching EOS before the limit | 0 / 6 |

All six outputs reached the 2048 total-length ceiling. No full-answer accuracy or accuracy improvement is claimed. The low active cache occupancy despite the higher p reflects repeated every-step persistent eviction; removing the 1024 limit permits growth but does not require it. The dedicated uniform-distribution regression separately verifies that p=0.95 can retain more than 1024 reasoning positions. The GPU was released at trial completion. Full MATH500 evaluation remains unrun for this method.
