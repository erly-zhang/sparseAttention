# Owner-Row Shared Top-p Mask Benchmark

New experiment, not a restart of array 47279144. User approved 2026-09-30:
4 independent H10080GB, 8CPU/128GB each, max24h each, total cap96GPUh;
one submission from the live user-authenticated compute-node terminal.

## Frozen Algorithm

Llama3.1-8B BF16, 32 query heads / 8 KV heads; modify even layers0..30 only
in prefill, preserve full causal KV and original dense decode, no training.
The number16 refers to modified layers, not the32 query heads. Assign query
row i to head i%32. Each head computes complete causal attention over all keys.
Extract its assigned rows, then take each row's minimal descending-probability
prefix with cumulative mass >=0.95 (include the threshold-crossing key).
Equal probabilities use ascending original key position. No sink/recent/self
positions are forcibly added, no top-k cap, no blockwise key normalization.

Stitch these boolean rows into one shared causal support M. Each head independently
computes softmax(own QK/sqrt(d), masked by M), multiplies its own original GQA V,
then concatenates head outputs through the unchanged o_proj. Probabilities and
outputs are NOT shared. Query blocks256 preserve the full key dimension.
Implementation recomputes QK in the mask-application pass and makes no speed claim.

## Exact Data Reuse and Fairness

prepare.py clones all200 original inputs and pinned official scorer/dataset files
from shared_attention_map_bench_20260929. No network fetch or re-sampling.
Original input manifest SHA359e51c5d4c92ca297cc1fd8866fc90edc456b38701d3dbbbaa18c923b3a9786.
50 each KV Retrieval/En.MC/En.QA/Code.Debug. Identical tokenizer/chat template,
intact context, position IDs, answers, sample assignments and caps50/40/40/5.
Manifest changes only methods and preparation provenance.

Rerun Dense and shared_topp on the same worker GPU for each input:400generations.
Dense is the same FP32 explicit blocked attention at16layers as previous control,
not all-FA2. Require all200 new Dense generated-ID sequences, decoded predictions,
stop reasons and scores equal the independently audited historical Dense outputs.
Any mismatch stops the worker and preserves the discrepancy.

## Gates and Submission

Use only authenticated Safari VS Code Tunnel. Recheck hostname/squeue and authorization.
No SSH workaround, no GPU test on the development V100. No install/deploy/submit retries.
Package source hashes,25 CPU tests and full retokenization must pass.
Run prepare.py once from the verified staging package, then install.py once,
then validate_cpu.py with GPUs hidden and OMP/MKL2. run.sbatch is the actual script.
Set AUTHORIZED_SUBMIT_HOST only to the just-verified authorized host, then execute
submit.py exactly once. It validates source/input/test hashes, runs sbatch --test-only,
records exclusive submission intent, submits the0-3%4 array and saves actual ID.
If intent exists without ID, investigate scheduler/logs before any retry.

Each worker tests GPU blocked/literal parity (exact masks) and top-p1 Dense parity,
then2longest paired smokes, cache/finite/causal/budget checks. It estimates remaining
work with worst observed per-method N-squared prefill and full-cap decode rates,
1.5margin plus1800sreserve, requiring <84600s. Continuation is automatic within the
same allocation; do not manually launch another stage. Budget rejection preserves
artifacts and stops. No automatic new allocation, resource expansion or retuning.

## Final Audit

Require all4Slurm COMPLETED0:0,400records,50unique identities/task/method and independent
audit.py PASS400. Check existing auditor/report before starting one CPU audit.
Recompute official scores, tokenization/EOS/caps/repetition and full-vocabulary
final-prompt KL in float64; verify hashes, historical Dense regressions and recorded
mask coverage/minimal-prefix/density plus own-QK normalization/pair counts.
Runtime full-cache/finite assertions are grounded in frozen runner, not GPU replay.
Report per-task scores, actual outputs, caps, repetition and selected mask density.
En.QA is F1. Code.Debug's5token cap limits degeneration detection; keep it unchanged.
Subset not full InfiniteBench; final-prompt KL not layerwise KL; no efficiency claim.
