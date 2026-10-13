# Q-only repeat with16 sparse layers

Status: independent evaluator, CPU gates and one-shot submission script prepared.
User approved the proposed resource budget by requesting submission on2026-09-29.
Awaiting remote validation and an authorized submission route; no job submitted.

User-requested semantics:

- Llama3.1-8B,32 query heads,8 GQA KV heads,128 head dimension.
- Sparse layers0,2,...,30; dense layers1,3,...,31 (zero-based).
- Prefill owner = original_position %32. Owner sees ALL causal history via
  its original GQA KV head. No KV filtering, eviction or position compression.
- Merge per-head query rows back into original token order: [B,N,128].
- Repeat each token's vector32 times: [B,N,4096], then ORIGINAL full o_proj.
  This is repetition, not summing heads, concatenating adjacent tokens or
  computing the missing query heads. No1/32 or1/sqrt(32) scaling.
- Dense decode in every layer; full KV. No MLP training or calibration.
- Frozen adapter, assembly and routing are hash-checked and reused unchanged.

Repeat equals x @ sum_h(W_O[h].T), with original bias applied once, under the
usual PyTorch Linear orientation. It does not recreate32 independent outputs.
16/32 sparse layers is50% of layers, not50% of attention pairs. With one of32
query heads active, ideal pair work is (16 +16/32)/32 =51.5625% of dense,
excluding routing, projections, duplication, FFNs and kernel differences.
Actual latency must be measured; Q-only uses masked SDPA in the frozen kernel,
while dense uses FA2. Do not describe it as a matched-kernel speed comparison.

Approved evaluation: same frozen50 KV Retrieval inputs,
Dense vs native16 vs repeat16,150 generations total;3-input smoke included.
One H10080GB,8CPU,128GB,max2h. Native16 isolates the output-assembly effect;
comparison with old21-layer native is confounded by layer schedule/count.
No evaluation tuning. Checkpoint/model/tokenizer/render/truncation and scoring
must remain pinned. Repeat/native must have identical analytical pair counts.
Dense regressions only: sparse predictions intentionally differ from old runs.

Before any submission: authorized login-node terminal (or
new explicit compute-node submission exception), complete runner/manifest and
one-shot submission guard, all CPU tests with no skips, actual script test-only.
Do not reuse earlier jobs, submission exceptions or approvals.
Smoke correctness and a measured runtime projection must pass before the rest;
quality degeneration must be reported separately from execution validity.

Local checks: python3 -m unittest discover -s experiments/qonly_repeat16_20260929 -v
Tensor tests require the pinned remote PyTorch/Transformers4.44 environment.
deploy.sh performs CPU validation and input freezing only, NEVER submission.
submit.py creates an exclusive intent, runs actual-script test-only, then submits
once. Any failure or ambiguous result requires reconciliation, not an automatic retry.
The compute-node exception flag may only be passed after explicit user permission.
