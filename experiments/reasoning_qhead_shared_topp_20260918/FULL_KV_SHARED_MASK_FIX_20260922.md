# Full KV history with temporary shared Top-p masks

This is the correction requested by the user after the permanent-eviction misunderstanding. The earlier `uncapped_p095_*` experiments permanently removed KV and do not evaluate the requested method. Their raw outputs and frozen source are preserved.

## Runtime semantics

1. Prefill runs dense causal attention and stores the complete prompt K/V in every layer.
2. Each decode call appends that layer's new post-RoPE K/V exactly once to its full historical cache. Cache length increases by one; capacity grows as necessary, with no 1024 cap.
3. A representative Q head is sampled per layer and decode call. Its GQA-mapped K head scores the full valid history, including the just-appended key. Prompt positions are protected; reasoning positions are selected with Top-p=0.95.
4. The selected original positions are shared across that layer's KV heads and their Q heads. K/V are gathered into temporary buffers. Each Q head computes its own attention, normalized over its selected prompt-plus-reasoning keys. This mask is used for the **current call**, before its attention output is produced.
5. The temporary gather is discarded. It never replaces the full persistent cache or shortens its valid length. An unselected key remains available and can be selected again on the very next call. Subsequent layers perform the same procedure before the model samples the next token ID.

A sampled token ID obtains its layer K/V when fed through the model in the next forward call. This timing does not require deleting earlier KV.

The selector itself is deliberately preserved: probability mean over the most recent at most W real post-RoPE queries (`--query_window`, default 64; W16/W64 formal comparison submitted), then reasoning-only normalization and descending Top-p; prompt is excluded from the reasoning mass budget and always retained. Historical buffered-query scoring remains as in the previous selector (no extra historical-query causal mask); the actual token-by-token attention is causal. Ties use the existing torch sort behavior. Representative RNG remains private and sample/layer seeded (rep_seed=0); generation seed remains 42. The existing runner batches two copies of the same prompt; this change does not introduce heterogeneous padded-prompt batching.

## Code and method names

- `ra_harness/kvcompress/engine/shared_mask_cache.py`: `SharedMaskLayer`, read-only mask selection, append-only persistent K/V, temporary attention buffers.
- `ra_harness/kvcompress/engine/modify_forward.py`: `cache_attention` routes Qwen3, Phi3 and Llama forwards to current-step shared masking.
- `EvictCache` constructs `SharedMaskLayer` for both `qhead_shared_mask_topp` (preferred explicit name) and `qhead_shared_topp` (corrected alias).
- Permanent eviction is isolated under `qhead_shared_topp_eviction`. Its old regression suite explicitly uses that name. Frozen historical runtimes are unchanged.
- `SharedMaskLayer` rejects token budgets other than 0, residual reserves other than 0, and cadence other than 1. It explicitly forbids post-attention eviction calls. The runner's internal `sparsity_method=eviction` API is retained for compatibility; the selected cache class determines the actual full-history semantics.

Configuration:

```bash
--sparsity_method eviction --eviction_mode qhead_shared_mask_topp \
--top_p 0.95 --token_budget 0 --residual_length 0 --eviction_interval 1
```

Complete historical K/V consumes memory proportional to full sequence length. Temporary gathers add working memory. This method sparsifies attention computation; it does not claim permanent KV-memory compression.

## Validation

`tests/test_shared_mask_cache.py` has six tests covering:

- Both production method aliases dispatch to the correct class; permanent eviction is rejected.
- A high-value historical key skipped on one mask is recovered by the next mask, with persistent K/V and cache lengths bitwise unchanged by selection.
- Independent selector agreement, unequal selected row lengths, and exclusion of invalid storage slots containing NaN.
- Local reproducible representative RNG without modifying the global RNG.
- Multistep CUDA append-only history; prompt preservation; actual current-step sparse output versus independent GQA attention; p=1 dense equivalence; surviving full history after finished-row filtering.
- Full cache growth beyond 1024 and single-token prompt initialization.

The real-model smoke uses MATH500 IDs 0, 1, 2, two rollouts each, max total length 256, with BF16, the existing chat template and generation settings. The 256 limit belongs only to this engineering smoke, not to KV storage. No full accuracy evaluation is claimed.

The original bounded trial used an allocated idle GPU. Its one-off launcher
was retired during the approved2026-10-11 cleanup. Historical outputs and
frozen runtimes remain local; the existing trial can still be audited with:

```bash
python scripts/summarize_shared_mask_trial.py results/full_kv_shared_mask_p095_smoke_20260922
```

The retired launcher rejected existing output directories and froze runtime source plus a SHA-256 manifest. `cache_metrics.jsonl` distinguishes full-cache lengths from selected-mask counts. The audit reconstructs all input-ID hashes and verifies complete trial coverage, `full_cache_tokens == original step` at every layer/decode event, contiguous event coverage, final full-cache lengths, and independent selected/causal pair sums. No previous-mask recurrence is used as the candidate range. See the current experiment guide for maintained launch workflows.

Results: `results/full_kv_shared_mask_p095_smoke_20260922/`. The smoke and audit have completed successfully.

## Completed validation

- Six new semantic regression tests passed, including CUDA numerical references and previously unselected key reselection.
- Three real inputs / six rollouts completed the 256-total-token smoke with exit code 0.
- All 35,280 layer/row/decode records retained full historical KV: full-cache length equals the original processed-token step. Final per-layer cache lengths also passed; selection never shortened storage.
- Full trial input-ID alignment, output coverage, frozen-source manifest, and independent attention-pair reconstruction passed.
- Selected pairs: 193,406,464; causal pairs: 225,607,680; bounded-smoke global pair sparsity: 14.2731%.
- Maximum full-cache length: 255 (the final sampled token has not yet been forwarded); maximum selected mask length: 236. These are separate counters.
- Minimum selected reasoning mass: 0.9500002265. All six generations reached the 256 trial limit; EOS count 0/6. No accuracy claim is made.
- Timed batches total 53.88 seconds including diagnostic assertions and JSON logging; this is not a performance benchmark. The allocated GPU was released after the smoke.

Authoritative audit: `results/full_kv_shared_mask_p095_smoke_20260922/summary.json`.

## W16/W64 formal comparison submitted

The query window is now configurable. Both windows passed three-input/two-rollout smoke and exact pair/input audits; the untimed warmup also reproduced outputs. GPU array `1309009` runs each window on all 500 questions (two rollouts each), with 8 shards/window and at most 8 A100 workers concurrent. Collector `1309011` runs after all tasks end and gates accuracy on complete aligned coverage. Decode wall time and decode-only/global pair sparsity are recorded separately. Frozen configuration, provenance and job IDs: `results/full_kv_w16_w64_20260922/`.
