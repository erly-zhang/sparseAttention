# Dense–Sparse prefill readouts and J-space decomposition

The user authorized visualizations and decomposition on 2026-10-06. No GPU budget is imposed. The analysis operates on the completed 1,400 prefills; it performs no new model forward, generation, gradient fitting, or parameter update.

## Inputs and coverage

One fixed Dense J-lens, fitted on 1,000 WikiText-103 sequences of 128 tokens; Llama-3.1-8B-Instruct readout weights and final RMSNorm gains; BF16 saved block-output residuals from seven methods, four InfiniteBench tasks, 50 complete long-context prompts per task. All 32 layers and every previously sampled position are included. These are sampled activations from full-context forwards, not every activation in the full contexts. Prompt token positions contain no generated response trajectory.

## Dictionary and algorithm

At layer l, construct token-indexed rows of `(lm_head.weight * final_RMSNorm.weight) @ J_l`, or identity transport at layer 31. Folding the learned RMSNorm gains into the effective unembedding preserves this model's readout directions. Its input-dependent scalar denominator does not change direction. Normalize atoms to unit length; the cone is unchanged and coefficients are reported in this unit-atom convention. Use all 128,256 vocabulary entries, without Top-10 candidate restriction or display filtering.

Greedily add the unselected atom with the largest positive residual correlation. Update all selected coefficients by one optimal directional gradient step, project to nonnegative coefficients, and backtrack if squared reconstruction error increases. Stop adding materially uncorrelated atoms. Record results at 16 and 25 iterations; active support can be smaller. This is a tested local implementation of the paper's main-text gradient-pursuit approximation, not its private exact code or a guaranteed global minimum. Do not replace it with unconstrained projection or NNLS without labeling the change.

Per layer use a seeded Haar orthogonal rotation of the complete dictionary as a control. It preserves dictionary size, all atom norms, and the atom Gram matrix. Compute it equivalently by rotating target activations, using the original dictionary. This is our control choice, not an assertion of identical random controls to the paper.

## Metrics and interpretation

Report residual energy reconstructed `1 - ||h-reconstruction||^2/||h||^2`, its excess above the rotated control, active support size, recall of Dense active atoms, and paired relative changes of independently fitted reconstruction and remainder. This energy measure is uncentered and is not labeled the paper's centered variance explained. The greedy decomposition is not unique; support changes may reflect correlated token alternatives.

As a separate diagnostic, compute the fraction of `Sparse-Dense` change energy in the linear span of Dense's active atoms using a Gram pseudoinverse. This is a signed orthogonal projection with an orthogonal complement, not a nonnegative J-space decomposition. Independent reconstruction and remainder differences need not yield additive energy fractions. Missing Dense active support makes recall undefined, not zero.

Aggregate positions within each prompt/group, then give prompts equal weight. Bootstrap prompts, never treat positions as independent samples. Historical task scores are association labels; no new task accuracy or response is measured. Retain replay first-token mismatch flags. Differences indicate prefill representation and readout drift, not proven causal loss of a reasoning step. Pattern variants are not budget matched. No Claude workspace-layer boundaries are assumed for Llama.

## Outputs and execution

English Times New Roman scientific PDF figures, an interactive paired token-slice viewer, complete sparse supports/coefficients in tensor files, per-prompt metrics, sample-weighted summaries and audit JSON. 128 H100 tasks (32 layers x 4 tasks), each processing all 50 prompts and all seven methods; no concurrent GPU cap. FP32 matrix arithmetic, TF32 disabled; only small saved activations and readout matrices are loaded, not a full inference model. An independent source verification and persistent controller run concurrently, retry transient failures, resume complete inputs, and assemble final figures only when all workers and source hashes pass.

Reference: https://transformer-circuits.pub/2026/workspace/index.html
