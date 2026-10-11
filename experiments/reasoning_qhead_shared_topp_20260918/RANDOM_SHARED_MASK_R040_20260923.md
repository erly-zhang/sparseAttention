# Full-history random shared mask: 40% reasoning retention

**2026-09-24 status:** The user stopped this original random40 smoke before its first question finished. The inference and automatic full500 pipeline are stopped; partial logs remain. The pipeline description below records the original plan, not an active submission. The replacement smoke is documented in [RANDOM_RECENT_GPU_SMOKE_20260924.md](RANDOM_RECENT_GPU_SMOKE_20260924.md).

User-approved variant, 2026-09-23. Independent method: `qhead_shared_mask_random`.
Run root: `results/full_kv_random_r040_20260923/`.

For a prompt of length P and current non-prompt history of length R, each layer
independently draws a uniform subset U of size ceil(0.4 R), without replacement,
from positions P,...,P+R-1. The actual attention set is {0,...,P-1} union U.
The same set is shared by every Q head in that layer. It is redrawn at each step.
The current non-prompt token is a candidate, not an additionally protected key.
Indices are sorted by original position. Persistent K/V is append-only; only a
temporary gather is used for attention. Prefill is dense. RoPE and GQA use the
existing tested attention path.

No historical queries are stored for selection. There is no W16/W64 window,
representative QK scoring, probability averaging, or Top-p selection. A fresh
nominal Q-head ID is sampled for each input/decode step and is the same across
all layers and both rollouts; adjacent draws can repeat. Because the mask is
uniform random, this nominal head does not influence its positions.

Private CPU torch.Generator streams plus the existing stable integer hash are
used, never Python hash or global RNG. Head stream: rep_seed=0, input idx,
decode step. Mask stream: seed42, input idx, decode step, layer, original rollout
ID. Stable dataset indices are tied to the frozen input manifest; prompts and
input_ids SHA256 are checked for every output. Batch filtering cannot renumber
surviving random streams. CPU and CUDA selection identities are tested.

Per-step sparsity is 1 - (P + ceil(0.4 R))/(P+R). Reasoning-only sparsity tends
to 60%; prompt-inclusive and whole-run sparsity are lower. Whole-run reporting
uses summed selected/causal pairs, including dense prefill, not averaged rates.
This ratio is user-specified: it does not exactly reproduce old Top-p counts.

Model/data: Qwen3-4B, same 500 MATH test inputs; batch=2 (two rollouts per input),
BF16, FlashAttention-2, seed42, generation temperature=.6, generation top_p=.95,
manual loop without top_k, max_tokens=32768 INCLUDING the prompt. Generation
Top-p is separate from attention selection, which is disabled here.

Validation: 14 tests covering the new mode and old shared Top-p mode, including
GPU kernel vs independent reference, full-history preservation beyond 1024,
short prompts, exact ceiling budgets, prompt protection, no duplicate positions,
uniform marginal frequencies, private RNG and survivor-batch reproducibility.
Frozen tests are rerun by the pipeline; results are in correctness.log/json.

Pipeline: correctness -> complete generation on inputs 0,1,2 with the full
32768 limit -> exact input/budget/pair/length validation -> automatic submission
of one 8-shard MATH500 array (one A100 per shard) -> dependent CPU grading.
Smoke correctness does not require correct mathematical answers: accuracy and
EOS/cap outcomes are recorded separately. A failed smoke blocks formal launch.
No existing job is cancelled, and no old source snapshot/result is overwritten.
Per-step diagnostic logging is disabled; independent pair counts are derived
from prompt lengths and actual generated lengths.

Track `pipeline_status.json`, `smoke/runner.log`, `smoke/summary.json`, and
`submission.json` under the run root. Formal workers write `random/shard_i/`;
final validation and grading write `random/summary.json` and `random/grades.jsonl`.
The report includes per-run/pooled accuracy, EOS outcomes, generated lengths,
synchronized prefill/decode/batch times, throughput, raw pair totals and sparsity,
peak allocated GPU memory, input alignment, and hardware records.
