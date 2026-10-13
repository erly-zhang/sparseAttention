# Q-only residual readout calibration

Status: local collection/training/evaluation runner implemented, pending remote
CPU and GPU validation. No training or submission has occurred at packaging.
The user requested an MLP trial with Q-only on 2026-09-28 local time.
Original frozen experiments and their failure evidence must remain unchanged.

## Fixed semantics

- Only the query owner is routed: owner(i) = original_position(i) % 32.
- The owner query attends every legal past/current position j <= i through its
  original GQA-associated KV head. This does NOT mix all KV heads together.
- Preserve original RoPE, causal positions, full KV cache and dense decode.
- Preserve the existing 21 routed and 11 dense layers for the initial control.
- Output assembly is a separate variable: native selected for this approved pilot.
  Do not mix calibration
  features or checkpoints from different assembly settings.
- Q-only does not reconstruct missing query-head outputs or dense KV values.
  Its full-history support does not imply dense-equivalent representations.

## Saved MLP design

Follow ../../MLP_OUTPUT_CALIBRATION_PLAN_20260928.md:
final block -> original final RMSNorm -> residual adapter -> original LM head.
Hidden size4096, bottleneck128, SiLU, zero up-projection initialization;
1,052,800 trainable parameters including both biases. No extra normalization.
Base weights and LM head remain frozen; the student head operation retains its
gradient into the adapter. Dense baseline explicitly bypasses the adapter.
Apply it to the last valid prompt state AND every answer-prediction decode state.
This is readout correction, not layerwise attention or KV-cache repair.

calibrator.py implements the residual module and normalized hidden-MSE plus
KL(dense teacher || corrected student). The linear control uses the same rank
and biases, without SiLU; it is a low-rank affine residual, not full-rank OLS.
Use small feature microbatches because loss materializes full vocabulary logits.

## Approved first pilot

- Independently generated synthetic retrieval only:32 train +8 validation +8
  held-out prompts. Split by prompt, with disjoint seeds and key/value identities.
- Balance 4K and16K token-length bins and target depths. This is a pipeline pilot,
  NOT validation of the prior approximately130K-token benchmark configuration.
- Exclude all existing benchmark/diagnostic inputs from training and selection.
- Collect dense and actual Q-only prefill separately, with separate caches.
- Generate one dense-teacher answer trajectory (max128), then feed the identical
  answer prefix into each branch with dense single-token decode. Pair each state
  with the exact same next-token target, including EOS, never after EOS.
- Collect first-answer and decode states; freeze input IDs, prefix IDs, tokenizer,
  model revision, routing/assembly, dtype and collector source identities.
- Compare dense, uncorrected Q-only, rank128 linear residual, rank128 residual MLP.
- Optimize adapter only on train; use validation for checkpoint selection. Start
  alpha=beta=T=1; no hyperparameter sweep in this pilot. Weight prompts equally.
- Evaluate held-out free-running exact match, repetition, EOS/caps and latency,
  plus teacher-forced hidden error and vocabulary KL. Report teacher correctness.
- No claim of recovered quality from KL alone or from toy CPU tests.

User approved NEW budget:1 H10080GB,8CPU,128GB,max2h, complete and release
early. CPU correctness and two-prompt GPU pipeline smoke precede pilot collection;
remaining work must fit the measured-time budget gate, otherwise preserve and stop.
No automatic allocation retry or budget override. The old6h job authorization does
not authorize this training job. The user separately approved this pilot and
explicitly allowed submission from the authenticated compute-node terminal.

## Remaining gates

1. Scope fixed: native, Q-only, 48 synthetic prompts, approved resources above.
2. Execute tensor tests in the pinned PyTorch environment (local Python lacks torch).
3. Validate implemented paired collector, synthetic split manifest, trainer and
   inference integration; verify first-answer/decode coverage and dense bypass.
4. Test GQA/causal full-history support, separate full caches, feature provenance,
   train/validation/held-out exclusion, loss gradient and save/reload parity.
5. Freeze actual scripts and data identities; Slurm test-only; submit exactly once.
6. Require completed accounting, complete held-out outputs and independent scoring.

The inherited Q-only backend uses explicit masked SDPA. Do not silently replace
it with FA2: noncontiguous selected query positions require original-position
causality. Any kernel optimization needs its own numerical/causal parity tests.

## Implementation and validation

pilot.py implements prepare/check/run. Frozen inherited routing and native assembly
are loaded by absolute path with SHA pins; no old code or results are modified.
The48 dense teacher trajectories and Q-only replay trajectories retain separate
caches. Only train/validation features enter training/checkpoint selection.
Five epochs per adapter, AdamW lr0.001, no weight decay, microbatch8, gradient norm
cap1; prompt-balanced weighting. Select minimum validation loss among five epoch
checkpoints. Initial identity is separately reported as the uncorrected baseline.
Heldout free generation comprises32 records; dense predictions reuse the exact
heldout dense collection generations. GPU smoke uses train-00/01 only and verifies
identity-adapter free-generation parity; smoke collection features are reused.
Training is pure dense-output restoration, not ground-truth cross-entropy training.
Two-prompt measured projection includes all remaining collection/generation and
1800s training/load reserve; deadline6600s preserves the2h allocation boundary.
If a gate fails, preserve all artifacts; do not resubmit or expand automatically.
Local:4 stdlib tests passed;9 tensor/pipeline tests skipped because torch absent.
Remote validate_cpu.py requires all13 new tests and3 frozen routing tests to pass
with no skips. Neither syntax checks nor this README assert GPU correctness.
