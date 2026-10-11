# Random shared KV compaction trial

User authorized a random_pp-like permanent KV compression trial on 2026-09-26. Method: `qhead_shared_random_compact_recent_gpu`. Previous full-history implementations/results remain intact.

Keep all prompt P separately. Capacity per layer is P+1024+64. Dense prefill; decode appends one key and attends to all surviving KV. After reasoning cache reaches1088, retain newest64 plus960 uniform random older surviving reasoning positions and all prompt. Compact to P+1024, preserving chronological order and original post-RoPE K/V values. Repeat every64 appended reasoning tokens. Evicted positions cannot return. Budget excludes prompt unlike official random_pp. Different layers and rollouts sample independently; all Q and KV heads within a layer share positions, while K/V values remain distinct and GQA mapping stays unchanged.

Reuse current GPU random-priority sampler: independent32-bit discrete uniform primary scores, secondary independent random bits, original candidate position as final exact tie-break. Private RNG seed coordinates: mask_seed42, sample_idx, cumulative reasoning step, layer, original rollout. This is a uniform subset in the ideal no-tie model, with negligible finite-key positional tie bias. No attention weights or softmax in selection. Attention itself uses normal QK softmax.

Fixed persistent K/V and fixed gather scratch; compression occurs after attention, as in random_pp. This allows up to1088 reasoning positions for the step that triggers compression. Changes are not storage-only: eviction cadence, candidate history, and between-compaction attention sets differ from the full-history per-step mask.

Tests cover prompt>1024, exact budgets, private RNG, marginal frequencies, sorted unique retained positions, batch filtering, permanent removal and numerical BF16 GQA attention against independently indexed full original tensors. GPU test must pass before generation. Trial: Qwen3-4B, MATH500 IDs0,1,2, two full sampled rollouts per question; seed42, BF16, temperature0.6, generation Top-p0.95, total max32768, untimed2-token warmup. No full500 automatic submission.

Frozen output: `results/compact_kv_random_reason1024_recent64_smoke_20260926`. Post-run validation independently reconstructs exact attention pairs, periodic compaction counts, bounded capacity and full cumulative RoPE positions. Compare with identical inputs from full-history reasoning1024, random40 recent64 and official random_pp; small trial does not establish full500 accuracy.

Job1344038 on n0003 A10080GB: all37 GPU/regression tests passed; complete-generation smoke completed successfully at 2026-09-26 10:46:56 EDT. Frozen runtime matches the current implementation. Scheduler COMPLETED, exit0.

## Completed trial

All six generations correct and EOS-terminated; no length-cap exits or OOM. All37 tests pass. Input lengths/SHA256 identical across methods. Independent periodic-cache, compaction-event and pair-budget checks pass.

| Method | Correct | Mean output tokens | Mean decode seconds/question batch | Decode tokens/s | Peak allocated GiB | Pair sparsity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| compact_prompt_plus_reasoning1024_recent64 | 6/6 | 2656.0 | 125.509 | 42.308 | 9.107 | 47.08% |
| full_history_reasoning1024 | 6/6 | 3329.2 | 206.293 | 32.266 | 11.937 | 56.91% |
| random40_recent64 | 6/6 | 3026.0 | 194.555 | 31.097 | 10.335 | 56.82% |
| official_random_pp | 6/6 | 2578.5 | 115.186 | 44.754 | 7.821 | 43.38% |

Same A10080GB class but different nodes/times. Two concurrent rollouts per question; timing excludes loading/warmup/diagnostic serialization. Changed output lengths and selection cadence contribute to latency differences. Three questions cannot establish MATH500 accuracy or exclude all long-run memory defects; bounded-cache behavior is validated, and full500 has not been submitted.

Detailed results: [comparison](results/compact_kv_random_reason1024_recent64_smoke_20260926/smoke/comparison.json), [validation summary](results/compact_kv_random_reason1024_recent64_smoke_20260926/smoke/summary.json).


## Authorized full MATH500 experiment

Explicitly authorized after the passed compressed-KV smoke. Submitted 2026-09-26 10:55:45 EDT: GPU array `1344076`, CPU validator/grader `1344077` (`afterany` all workers). Eight shards, one A10080GB each, maximum eight concurrent GPUs, 48h worker limit. Scope500 questions, two sampled rollouts per question,1000 outputs. Independent root: `results/compact_kv_random_reason1024_recent64_20260926/`.

Inference and tests are byte-identical to the passed37-test/six-generation smoke. Prompt excluded from reasoning1024 budget; recent64 inside it; all KV heads within a layer share retained positions; persistent capacity P+1088; after-attention compaction every64 reasoning tokens. All generation settings unchanged.

Full model/tokenizer and dataset hashes frozen and rechecked by workers. Reporting validated with the saved smoke output and independently simulated periodic budgets through32767 reasoning steps. The collector requires all500 unique inputs/1000 outputs and passed/full alignment; failures remain incomplete. It reports accuracy, EOS/cap, generated lengths, synchronized timing distributions, peak GPU memory, raw pairs and sparsity, plus same-input comparisons with random40 recent64 and official random_pp (historical W16/dense also preserved). The earlier no-full500 statement applies to the smoke phase; this submission follows new explicit authorization.

Monitor [submission](results/compact_kv_random_reason1024_recent64_20260926/submission.json), worker `random/shard_*/runner.log` and `exit.json`, then `collection_status.json` and `random/REPORT.md`.


## Full run completed — 2026-09-26

All8 workers and the dependent collector completed successfully. Collector finished2026-09-26 17:57:17 EDT. Full500 questions/1000 outputs; input_alignment=passed/full; no duplicate/missing IDs, no OOM, no grading errors. Periodic pair budgets, bounded cache capacity P+1088 and cumulative RoPE positions passed validation.

Correct891/1000 (89.10%): run0 451/500 (90.2%), run1 440/500 (88.0%). EOS916/1000;84 length-cap exits. Mean output6741.953tokens. Mean prefill0.04166s; mean/median/P90 decode301.8229/131.9322/1248.3487s per two-rollout question batch. Aggregate decode44.6683tokens/s. Peak allocated15.303GiB. Overall pair sparsity88.39495%; selected/causal pairs8,550,627,005,952 /73,680,244,618,752.

Against full500 random_pp: accuracy+1.9pp (36new-only correct,17baseline-only); mean decode301.82 vs323.58s,6.7%lower; peak memory15.30 vs10.30GiB. Against full500 random40 recent64: accuracy-4.1pp (17new-only,58baseline-only); mean decode301.82 vs432.65s,30.2%lower; peak15.30 vs60.15GiB. These are single-seed observations on different A100 nodes with different output lengths, budgets and selection semantics, not isolated kernel or statistically established general improvements.84capped outputs remain a quality issue; random_pp93,random40 22.

Final report: [REPORT.md](results/compact_kv_random_reason1024_recent64_20260926/random/REPORT.md). Raw outputs and comprehensive metrics remain in the independent full-run directory.
