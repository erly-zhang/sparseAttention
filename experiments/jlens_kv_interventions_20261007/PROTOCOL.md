# KV Retrieval: causal J-lens pilot

Prepared on 2026-10-07. No model forward or Slurm submission is authorized by
this document itself. The user previously requested a concrete configuration
and algorithm report before approving a new experiment. The new pilot waits
for that confirmation; the old fitting/observer approvals remain untouched.

## Inputs and model

Use the existing case kv_retrieval-359, the complete 124,405-token InfiniteBench
input, Llama-3.1-8B-Instruct BF16, the frozen attention runtime, all 32 layers,
and the existing WikiText-fitted Dense J-lens. Keep calibrated representative
heads fixed. Sparse masks in downstream layers are recomputed as usual from
the intervened activations. Never truncate the original context.

Correct V: ca3b3e39-bf38-4633-974a-3a5d1bdc34cd (25 isolated tokens, 18 distinct
vocabulary IDs). A complete UUID has no single J-lens vector. Multi-token bag
steering retains token multiplicity but cannot represent order or K-to-V
binding. Ground-truth-dependent interventions are oracle diagnostics, not a
deployable retrieval method or a fresh unbiased benchmark score.

## Pilot matrix: 105 full-context runs

Three fresh no-intervention references: Dense, calibrated Token+Vertical+Slash,
and FlexPrefill. Activation interventions initially apply to calibrated V+S
only: a single block-output layer 16, 24, or 28, at either position 89362
(after the source V, in its saved neighborhood) or 124404 (answer-start input
position). Only one residual row is changed in each run.

For each layer/site pair, test four unit-norm steering directions at relative
perturbation norms 0.1, 0.3 and 1.0 of that row's residual norm: the first
correct-value token, the entire correct-value token bag, a wrong UUID bag
sharing the same first ca token, and a seeded isotropic random direction.
These 72 trials use h' = h + alpha ||h|| unit(direction).

For each layer/site pair, also test three Dense-donor interventions (18 trials):
(1) replace the projection onto the span of the 18 correct-value token
directions by its Dense counterpart; (2) add the Dense-minus-Sparse component
orthogonal to that span, rescaled to exactly the same perturbation norm as (1);
(3) replace the complete residual row with the saved Dense row. For (1), with
an orthonormal row basis B of the target directions, P=B^T B and
h'=h+P(h_dense-h). This preserves the orthogonal complement of h. It is a
target-token linear-span patch, not a replacement of the entire sparse
nonnegative J-space component. The donor comes from the verified full-context
Dense observer at the same original layer/position.

Twelve text positive/control runs insert a correct or incorrect key/value
reminder near the source V or immediately before the final Key question,
under each of the three methods. These change the prompt and its positions;
analyze them separately from activation interventions. Their purpose is to
check response to an explicit hint, including specificity against a wrong
value, not to attribute the original attention failure.

The effective direction dictionary folds the model's learned final RMSNorm
gain into unembedding: normalize rows of (W_U * gamma) J_l. This matches the
readout directions for this model up to a positive input-dependent RMS scale.
This gain convention is retained from our earlier decomposition.

## Generation and observations

Run one complete prefill per trial, then greedy Dense decode with full KV cache.
Do not inject during decode and do not feed gold answer tokens. Raise the common
generation cap to 128; also score the first 50 generated tokens with the same
official scorer to distinguish output-cap effects from intervention effects.
New baseline runs use the same 128-token cap. Compare their first historical
steps to the earlier evaluations; numerical/replay discrepancies must be
reported before interpreting rescue effects.

Save BF16 residuals at every layer for the existing sampled prefill positions,
and every generated input-token step. Modified text controls use their own
prefix/tail/uniform positions. Record before/after residuals at the injected
site. A block-output hook occurs after the current layer's KV cache is written:
it changes the residual consumed by later layers, not the existing current-
layer cache. Full residual patching therefore does not restore the whole
Dense cache. The first generated token is predicted at the final prompt
position; a decode row for generated token j predicts token j+1.

Read out all correct-value token IDs over the full vocabulary. Perform the
same nonnegative gradient-pursuit decomposition at k=16 and k=25 using all
128,256 unit token atoms at layers 16,24,28,31: selected source/query positions,
the first 32 decode positions, and the measured injection before/after pair.
Store supports, coefficients, reconstructions, remainders, and uncentered
reconstructed energy. This is not the paper's centered variance statistic.
Higher injected-token rank at the intervention site is expected by construction;
causal interpretation rests on downstream persistence and a specific answer
change relative to controls, not that local rank increase alone.

## Relation to the paper

The paper's Methods, Writing section uses token-direction steering and
lens-coordinate patching. This pilot reuses those intervention principles and
the same J-lens readout definition, adapted to Llama and long-context UUID
retrieval. The paper's two-concept coordinate swap is not exactly our
Dense-to-Sparse target-span replacement. Our 25-token bag is also not a single
concept with an unambiguous semantic direction. We cannot label it an exact
replication of the paper's task or layer ranges.

Source: https://transformer-circuits.pub/2026/workspace/index.html#methods

## Execution and limits

Each independent trial requests one H100-80, 8 CPUs, 128 GB host RAM and 8 hours
through GPU-shared/cis260309p/gpu. No array concurrency cap.
The three baseline jobs run first; all other trials depend on their successful
generation-prefix parity and analysis. A baseline mismatch stops that launch.
The pilot plan, inputs and new source code are hashed, and the original experiment files are
not changed. Atomic JSON progress is saved every 300 seconds; decode prefixes
are saved every 8 steps. A failed trial restarts from prefill; no KV-resume claim.

No full-context GPU hook compatibility/parity check has yet run. Those checks
belong to the approved pilot, not the CPU geometry checks. Single-case rescue
demonstrates a local causal effect but cannot estimate 50-sample task accuracy
or prove which sparse mask edges caused the original failure. A positive
oracle injection can bypass retrieval; even success need not establish normal
retrieval of the original key-bound value.

Scheduler update on 2026-10-07: the account no longer permits push QoS.
Use its permitted gpu QoS, keeping the H100 requirement and no array concurrency cap.
