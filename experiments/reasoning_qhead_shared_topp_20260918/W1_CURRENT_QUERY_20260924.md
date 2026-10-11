# W1: current-query representative-head shared Top-p mask

Run root: `results/full_kv_w1_p095_smoke_20260924`.
This is a controlled query-window ablation. The inference source is byte-identical to the completed W16/W64 experiment; the selector parameter is `--query_window 1 --top_p 0.95`. Dedicated tests and a separate three-input smoke pipeline have been added. No full500 task is automatically submitted.

At layer l and current decode position t, choose representative Q head h(l,t) with the unchanged layer/input-seeded private generator. The only query used by the selector is the current post-RoPE vector. For the correct GQA KV head g(h):

\[
a_{l,t,j}=\operatorname{softmax}_{0\le u\le t}\left(\frac{q^{h(l,t)}_{l,t}\cdot k^{g(h(l,t))}_{l,j}}{\sqrt d}\right),\qquad 0\le j\le t.
\]

Let P be the prompt length and R_t={P,...,t} the non-prompt positions. For nonempty R_t, normalize the non-prompt part of the representative distribution:

\[
b_{l,t,j}=\frac{a_{l,t,j}}{\sum_{u\in R_t}a_{l,t,u}}.
\]

Sort b in descending order using the existing implementation. Keep the shortest prefix with cumulative mass at least0.95, and union it with all prompt positions. Sort final indices by original position and share this mask with every Q head in the layer. Actual attention is independently normalized for each head on its own query and GQA-mapped K/V. The mask is reconstructed every layer and step. No fixed percentage, additional recent64 protection, block mask, or persistent KV eviction is added.

The current query is derived from the latest input position, but its attention still depends on historical keys and values. Autoregressive generation models P(x_{t+1}|x_0,...,x_t), not merely P(x_{t+1}|x_t). W1 limits the number of selector queries, not the number of historical keys.

## Important clarification about W16/W64

The stored historical queries in the implemented W16/W64 selector are rescored against all keys currently in the full-history cache. The selector does not independently mask each old query to its own original position. This differs from a previously stated per-historical-query causal-softmax formula. No historical implementation or reported result is altered by this experiment. With W1, the sole query is current and all currently cached keys are legal for that query, so that historical-query distinction does not arise.

## Execution and validation

The query ring has capacity1. Dense prefill leaves its last query in that slot; the first decode update overwrites the slot before selection, as do all later updates. Added tests verify this ordering, direct current-query Top-p selection, re-selection of old KV, unchanged global RNG state, full-cache preservation, BF16 GQA attention against an independent reference, batch filtering, and history longer than1024.

Smoke uses canonical MATH500 inputs0,1,2, two full generations per input, Qwen3-4B BF16, temperature0.6, generation Top-p0.95, seed42, representative seed0, total length32768 and the same untimed two-token warmup as W16/W64. Per-step logging/debug output is disabled for timed generation. Accuracy, EOS/cap, lengths, synchronized prefill/decode/batch times, throughput, selected/causal pairs, pair sparsity and peak allocated memory are saved. Inputs and pair denominators are independently cross-checked. Same-three-input W16/W64 metrics are included in the final summary.

Status: see `pipeline_status.json`; final results: `smoke/summary.json` and `smoke/grades.jsonl` inside the run root.

## Completed smoke

11/11 tests passed; 3/3 complete inputs and 6/6 normal EOS. Correct 3/6. Mean decode 383.43s/question batch, weighted throughput 17.12 tokens/s, overall pair sparsity 76.50%. See [same-input comparison report](results/full_kv_w1_p095_smoke_20260924/smoke/REPORT.md). An output-filename assumption in the postprocessor was fixed without rerunning or modifying inference. No full500 submission was made.
