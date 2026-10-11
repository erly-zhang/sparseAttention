# Official Random-Attention matched MATH500 baseline

User request: run the concrete SalesforceAIResearch/Random-Attention implementation and measure accuracy, sparsity and timing with the same preprocessing/data/input settings as our experiments.

Upstream: https://github.com/SalesforceAIResearch/Random-Attention , verified main commit `64db9688a12f0926db45fec039370f6ceb1ab4fe`. Unmodified upstream `cache_utils.py`, launch defaults and license/provenance are under `ra_harness/kvcompress/engine/official_random_attention/`. Adapter: `official_random_pp_cache.py`; independent method name `official_random_pp`. Existing shared-mask variants and frozen running jobs are preserved.

Official algorithm: MATH default total budget1024, residual64, `random_pp`. Independently per KV head, protect the prompt (upstream caps protection at959, but all500actual prompts are32..814), assign uniform random scores to older cached candidates, choose960 including protected prompt, sort chronologically and append the most recent64. Permanent compaction happens AFTER attention when cache length>1024 and divisible by64: first at1088, then every64appends. Subsequent queries cannot retrieve permanently dropped KV. Each group of Q heads shares its GQA KV head's retained history; this is not a whole-layer shared mask. Absolute RoPE sequence length remains cumulative. These are algorithm differences, not changes to our own full-KV methods.

Evaluation controls match the existing runner: Qwen3-4B identical weight/tokenizer SHA256; BF16; exactly500canonical MATH inputs; same chat template, prompt construction, no additional truncation, seed42 with stable per-input reset, generation temperature0.6/Top-p0.95/no top_k, batch2/two rollouts, total length32768, same EOS and finished-row filtering, same2-token untimed warmup, oneA10080GB per worker. All500input token counts, IDs hashes, decoded prompt and gold answers were reconstructed and matched. The input/preprocessing/grader utilities are byte-identical. Removing only the explicit algorithm dispatch/config/instrumentation additions restores all five touched runner/engine integration files byte-for-byte to the W1+recent64 frozen source; this is recorded in `protocol_alignment.json` and `runner_adapter.diff`.

One explicit reproducibility adaptation: execute upstream `_run_eviction` unchanged inside a private per-input/layer RNG context; restore the global sampling RNG afterward. This prevents KV random draws changing the generation RNG stream, unlike the coupled upstream launcher. It preserves the upstream sampler/sort/compaction exactly given the same RNG state; CPU and GPU bitwise parity tests pass. This is an official algorithm baseline under our matched evaluation protocol, not a claim to recreate upstream published answers with its original launcher. RNG context overhead is included in decode time.

Metrics: actual attention selected pairs counted BEFORE eviction, complete causal-history pairs as denominator, global sparsity1-sum(selected)/sum(causal), synchronized full-prefill/decode/batch wall time, aggregate active-token throughput, output-length distributions, EOS/limit exits, peak allocated GPU memory, per-run/pooled/per-level accuracy, paired outcomes and hardware. An independent integer recurrence verifies every selected-pair total, eviction count and final retained length. Loading, warmup and per-step diagnostic serialization are outside timing. Retention budgets differ across algorithms, so this is not a matched-sparsity experiment. Dense historical timing retains its documented limitations.

Experiment root: `results/official_random_pp_k1024_r64_20260924`.
- `input_preflight.json`: passed/full,500inputs.
- `protocol_alignment.json` and `runner_adapter.diff`: controlled source audit.
- `correctness.log`:29shared-mask regressions plus4official-specific tests passed.
- `smoke/`: identical inputs0,1,2,two complete rollouts each; automatic grading and pair validation.
- `pipeline_status.json`: correctness,smoke,gate,full-submit progression.
- `submission.json`: GPU array and CPU collector IDs once the smoke gate passes.
- `formal/`:500question outputs across8shards,max4concurrentA100GPUs,48h per worker; independent directories, no overwrite.
- `formal/REPORT.md`, `summary.json`, `question_metrics.jsonl/csv`, `rollout_metrics.csv`: final measured results only after complete aligned1000answers.

The user authorized the matched baseline; the pipeline submits full500 automatically only on technical smoke success. No extra tasks, seeds or budget sweeps are launched.

Completed smoke2026-09-24: 6/6correct, 6/6EOS; all input and exact eviction-pair checks passed. Mean prefill0.0430s, mean decode116.167s/question batch, weighted decode44.376tokens/s, global pair sparsity43.38%, peak allocated7.982GiB. Full500submitted as GPUarray1325433 and dependent collector1325434,8shards/max4concurrentA100. These are smoke measurements, not final500accuracy.
