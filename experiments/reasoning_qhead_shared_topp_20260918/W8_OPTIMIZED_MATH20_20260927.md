# Optimized W8: first20 MATH500 paired trial

User request: replace optimized W1's current-query selector with W8, run only20 MATH500 questions, and compare accuracy and average speed. This trial keeps Qwen3-4B and the current reasoning runner; it does not resume the earlier Llama KV Retrieval task.

Result directory: `results/w8_optimized_math20_20260927`.
Jobs: GPU correctness1348109 → four paired GPU workers1348111 → validation/grade1348113. No automatic500-question submission. Historical and unrelated jobs use frozen sources and are not modified.

## Algorithm

At each layer/event draw the same single representative Q head h from the existing private sample/layer stream (rep_seed0). Its KV head is floor(h/(Hq/Hkv)). Let Q[t-7:t] be its latest8 real post-RoPE queries, and K[j] all currently surviving keys, including the current token. Then

    p_i(j) = softmax_j( Q_i K_j^T / sqrt(d) )
    score(j) = (1/8) sum_i p_i(j)

The FP32 softmax is independent per query. Average probabilities, not logits, and do not apply another softmax. The importance estimator uses all currently surviving keys for each historical query, matching the historical window selector semantics; it adds no historical-query causal mask. The actual generation attention remains exactly causal.

After attention, when surviving reasoning reaches1088, keep all prompt tokens (outside budget), the recent64 reasoning tokens (inside budget), and stable TopK960 among older reasoning by score. Ties prefer earlier original positions. Sort retained indices chronologically and physically compact all KV heads to the same positions within each layer/rollout. Compression repeats every64 decode steps; evicted KV never returns. Representative head is shared across the8 queries and active rollouts, but each rollout scores its own Q/K.

Execution inherits optimized W1's bounded double buffers, one-time prompt copy, reasoning-only fused Triton KV gather, and buffer swap. Copy Q only for the last8 decode steps before an event. This is exactly equivalent to keeping a rolling window every step because interval64 >= W8; GPU regression compares these implementations.

## Trial controls and checks

Same canonical MATH500 IDs0..19; two sampled rollouts/question per method, batch2, seed42, generation temperature0.6/top-p0.95, BF16/FlashAttention2, max total32768, two-token untimed warmup. W1 and W8 run on the same A10080GB within each shard. Even shards run W1→W8; odd shards W8→W1. Input count/hash must match canonical records; no missing or duplicate samples. Full model/tokenizer hashes, packages, commands, hardware and frozen source manifest are recorded.

New tests cover an independent probability-mean oracle, stable ties/protection, GQA, chronological query capture, skipped non-event Q copies, batch filtering, and CUDA parity against a continuously updated window (attention, selected IDs, scores, KV, original positions, private/global RNG and pair counts). Existing45 regression tests remain part of the GPU gate.

Collector independently checks exact selected/causal pair and budget formulas from realized lengths, per-sample EOS/length limits, input alignment, and complete outputs, then grades only the generated suffix with the existing math grader. Reports correct/40 and per-run correct/20, mean/median/P90 decode, mean prefill/total, decode tokens/s, output lengths, limit hits, peak memory and global pair sparsity. Mean wall time is per question batch of two rollouts.

W8 changes generation trajectories, so pair totals and text need not equal W1; each must satisfy the same budget law. Report wall-time changes together with output length and throughput. This small first20, single-seed trial cannot establish full-dataset accuracy improvement, and end-to-end timings alone cannot isolate the selector's overhead.

Completion artifacts: `summary.json`, `REPORT.md`, `w1_grades.jsonl`, `w8_grades.jsonl`, and `collection_status.json`. Check these and Slurm status before claiming completion; row count alone is insufficient.
