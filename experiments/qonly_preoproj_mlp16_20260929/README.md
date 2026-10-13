# Fresh Q-only/repeat16 pre-o_proj MLP pilot

User approved 2026-09-29: one H10080GB,8CPU,128GB,max4h and one
submission from authenticated gl002. No old checkpoint is loaded.

Frozen Llama3.1-8B BF16, original GQA and full KV cache. Sparse layers
0,2,...,30; dense odd layers. Q-only selects queries by original_position%32,
each assigned query sees all causal history in its original GQA KV group.
The merged128-vector is repeated into32 slots, without scaling.
Each sparse layer applies its own new residual SiLU MLP4096->128->4096
AFTER assembly, BEFORE original o_proj. Total16844800 new parameters.
MLPs activate only during prefill; cached decode uses original dense attention.

Fresh deterministic synthetic UUID retrieval:32train/8validation/8heldout,
balanced4K/16K. New seeds and keys/values; exclude previous synthetic IDs and
benchmark input hashes. No benchmark training, correctness filtering or tuning
on heldout. Sample256 token positions/prompt a priori:192 uniform plus last64.

Sequential calibration: collect dense pre-o_proj targets on40train/validation
prompts, then train each sparse layer in order. Previously selected MLPs are
active during collection for the next layer. Per-layer loss is mean relative
squared O error plus relative frozen-o_proj output error. FP32 training, BF16
inference cast. AdamWlr0.001,weight_decay0,batch64,gradientclip1,5epochs;
zero-up residual initialization. Select minimum validation loss among identity
epoch0 and5trained epochs. Save all96checkpoints and16selection histories.
Base weights frozen, not optimized. No final-postnorm calibrator is reused.

Two TRAIN prompts smoke: dense feature collection, sparse prefix collection,
6generations including uncorrected/identity-MLP token parity and actual small
five-epoch fit. Discard probe weights. Conservative measured projection includes
all40dense passes,640 full-depth sparse prefixes,24heldout generations,16fits,
factor1.5 plus1200s reserve. Gate <13800s in a4h allocation; no automatic retry.

Then finish40dense targets, sequential16MLP training and finally8heldout inputs
x dense/repeat16/mlp16=24generations, greedyseed42max128. No heldout features
are collected. Independent post-run audit is required; complete.json alone
is not a final result. See MONITOR.md. Eight heldout prompts are not a benchmark.
Loss improvement is not generation recovery; finalprompt vocabulary KL is not
layerwise KL. Q-only SDPA fallback and dense FA2 are not matched speed controls.

deploy.sh only creates a fresh remote directory, validates sources and11CPU
tests, and freezes real tokenizer IDs. submit.py performs actual-script test-only
and exactly one sbatch, with exclusive intent/job files. Never overwrite or
retry a failed/ambiguous run without reconciliation and new authorization.
