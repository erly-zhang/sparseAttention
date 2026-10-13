# Monitoring and final audit gates

Use only authenticated Safari VSCodeTunnel psc-47216067; user handles auth/trust.
Never run training on development gl002, use SSH, redeploy, resubmit, or revive
previous repeat16/MLP jobs. Read submitted-job.txt before querying scheduler.
Verify hostname, squeue/sacct, slurm-ID.out, smoke-complete.json,
budget-projection.json, training-complete.json, complete.json and failure.json.
Existing job proceeds automatically after passing smoke/budget, no manual stage.
Budget rejection/failure: preserve and notify, no resource expansion/retry.

Completion requires all of:
- SlurmCOMPLETED0:0, correct singleH100/8CPU/128GB/max4h allocation.
- Source/archive/input hashes, actual48frozen IDs, no split/key/value overlap.
- Smoke2train prompts,6generations, exact identity-versus-repeat generated IDs,
 16prefill corrector calls and0decode calls; measured projection<13800s.
-40dense teacher feature files and16student feature files with correct256token
 positions, layer IDs, hashes, previous-selected checkpoint dependencies.
-96checkpoints (16x6 including identityepoch0),16selection histories independently
 rescored using saved validation features; minimum validation-only selection.
 Verify32train/8validation and noheldout training/features. Count16844800weights.
-24unique8heldout x3method results, token IDs/prediction/tokenizer agreement,
 independent exact-match, EOS/128caps and repetition audit, correct input hashes.
-Runtime full cache/per-step finite assertions, analytical Q-only paircounts,
 exactly16prefill MLP calls and0decode calls for mlp16; frozen base version check.
-Independent float64 KL recomputation from saved full vocabulary logits;
 verify checkpoint/logit/feature hashes and report dense teacher quality.

Report real generated examples,3way heldout accuracy, degeneration and caps.
Do not claim loss improvement is recovery or8heldout is benchmark evidence.
Monitoring requires desktop/tunnel access; quiet on unchanged/nonactionable
states, notify stage change/completion/failure/auth/approval. Remove monitor
after final audited report. Older monitors must not drive Safari concurrently.
