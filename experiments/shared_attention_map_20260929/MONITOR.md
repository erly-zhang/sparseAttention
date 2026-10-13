# Audit gates

Only inspect the job recorded in submitted-job.txt. Never rerun deploy.sh or
submit.py after submission intent. Test-only identifiers are not actual jobs.
User approved one H10080GB,8CPU,128GB,max30min and one v020 submission exception.
Only authenticated Safari VS Code Tunnel; user handles trust and authentication.

Confirm hostname, submitted-job.txt, squeue/sacct, Slurm log, budget-projection,
complete.json and failure.json. Do not automatically retry failures or allocate
more resources. No training or old MLP checkpoints. Do not alter other jobs.

Completion requires COMPLETED0:0 and six aligned results on3 frozen inputs.
Independently verify source and input hashes, regenerate tokenizer input IDs,
decode saved generated IDs, exact match against planted UUID, EOS/cap/repetition,
saved full-vocabulary logits and their hashes; recompute float64 KL against each
input's dense_full control. Check analytical full-map and causal pair counts,
all16 layer diagnostics, normalized rows, ownership counts and unchanged decode
call counts. Full-cache/per-step-finite/frozen-parameter assertions come from the
frozen runner, not a claimed independent GPU replay. Include actual predictions.

Dense_full is the matched explicit FP32-attention control in16 even layers,
not bitwise FA2 parity. Other16 layers and cached decode retain original FA2.
Shared map uses original GQA8 value heads, so four query heads share each V.
All32 full maps computed before row selection, no computational sparsity claim.
This is a three-input synthetic smoke, not a benchmark or speed comparison.
