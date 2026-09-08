# Deployment and source provenance

This directory contains the instance2 implementation of the two Shared6144 +
Residual4096 experiments, including correctness tests, smoke-gated scheduling,
separate profiling, merging, and the matched-input answer-coverage analysis.
It is a research-run snapshot, not a path-independent packaged application.

## Included dependencies

- `../probmean_compact_20260906`: probability-mean selector and runner.
- `../rep_residual_20260907`: protected selection and task-configuration helpers;
  the earlier proxy-residual experiment is also retained as a separate method.
- `../random_residual_20260908`: deterministic uniform complement sampling and
  the earlier Shared9216 + random1024 control.
- The parent experiments directory contains the matching compact attention
  kernel, benchmark installer, and unified InfiniteBench runner.

These sources were copied from the actual instance2 workspace. Core files are
checked against the SHA-256 inventory saved by the formal six-four experiment.
No model weights, task records, predictions, raw profiling arrays, credentials,
or full evaluation outputs are included in this publication.

## Before running

Review the constants in `schedule.py`, `report.py`, `compare_answer_coverage.py`,
and imported `../rep_residual_20260907/prepare.py`. They refer to the original
`/home/ubuntu/work` workspace, attentionmap Python environment, and `/local`
data/output paths. Restore data and adjust paths to new, independent output
directories before a new run. Do not point a new run at archived results.

Required external artifacts include the Llama-3.1-8B-Instruct checkpoint and
tokenizer, existing AE K=3 head-group JSON, the exact 497-record filtered KV
dataset and its task YAML/helpers, the original input reference metrics, the
Shared10240 comparison metrics, and the original three smoke input records.
Use the existing repository setup for PyTorch, Triton, Transformers, NumPy,
lm-eval, FlexPrefill, PyYAML, and requests. The original max generation budget
is 128 tokens; profiling is not a replacement for full generation.

The post-stop instance is CPU-only. Restore the GPU environment on authorized
hardware before running tests or inference. Merely importing or compiling
Python sources does not establish CUDA/Triton correctness.

## Validation record

`correctness.json` is the historical GPU test record, not evidence of a new test
run on the machine where this repository is cloned. The recorded thresholds
were relative L2 <= 0.005 and maximum absolute error <= 0.02; observed maxima
were 0.0020093 and 0.0096054. Regenerate the test record on the intended GPU
environment by running `test_correctness.py` before scheduling inference.
The scheduler then requires both aligned three-input full-generation smoke
runs to pass before formal generation. Inspect the scheduler GPU assignments
and all existing processes before starting it.

Historical `/local` paths embedded in archived JSON must be resolved against
the persistent backup root when analyzing old results. Some old calibration
symlinks also reference `/local`; the underlying layer files are preserved in
the backup's `attempts/calibration` directories.

## Coverage analysis

`compare_answer_coverage.py` reads existing per-layer records; it does not run
inference. Its own-trajectory comparison uses three exact matched input hashes,
while the B-Q/K shadow Shared10240 comparison holds B's hidden states fixed.
Keep these scopes separate from 497-input formal accuracy. The analysis output
and visualization data are deliberately not included in the code-only commit.
