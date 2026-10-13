# PSC repair result — 2026-09-25

Status: compatibility repaired and five-method H100 correctness passed; the full controlled experiment is incomplete. No index-3 real-model comparison, AE checkpoint, or groups JSON was generated.

## Repair and validation

Preserved tracked source at commit 8b21a6a18056eadc00b6248bb319a4af3c6ccc4d. All changes remain in this independent experiment directory.

1. Restored the unchanged original MInference kernel; used an isolated Triton 3.0 override instead of shared Triton 3.2. The original kernel compiled offline for SM90. No sparse algorithm was substituted.
2. Copied the original AutoBlock probmean/member-score modules to repair_02; replaced only JIT compile-time math.sqrt(D) with D ** 0.5. The values agree exactly for D=128. All four affected kernels compiled offline. Original repository modules remain untouched.
3. All five methods passed GPU correctness in job 47071290: dense, routing C, Shared6144+member2q4096, FlexPrefill, MInference. Checks cover full cache and future invariance; routing C additionally checks independent attention/repeat-project semantics, AutoBlock checks actual selector/masked-attention semantics with a small test budget. These fixtures do not establish 124k-context correctness or accuracy.
4. Warmup is uniformly 129 tokens to accommodate MInference last_q=64; the formal test input remains the same 123959 tokens. MInference emitted a nonfatal compiler scheduling diagnostic but returned successfully and passed numerical checks; long-context compilation is still unverified.

## Actual execution and partial artifacts

| Job | Terminal state | Exit | H100 allocation seconds |
|---|---|---|---:|
| 47069472 | FAILED | 1:0 | 76 |
| 47070792 | FAILED | 1:0 | 58 |
| 47071290 | FAILED | 1:0 | 653 |

Total 787 seconds (13m07s), leaving 2813 seconds (46m53s) of the authorized 3600-second H100 allocation budget. No experiment job remains running. The previous import-time CUDA-constant accounting caveat documented in RESULT.md still applies; 787 seconds is exact Slurm H100 allocation time, not a claim to quantify that earlier incidental activity.

Job 47071290 ran on PSC w009, H100-80GB, and reached real-model calibration. Three calibration samples produced all 96 raw layer-map files. Mean maps completed for layers 0–12 (13 files). Headers, shapes (32,1936,1936), dtype float16 and file lengths were checked without a full numerical scan. The collection process exceeded its internal 600-second stage timeout while performing mean-map I/O; Slurm itself did not reach its 54-minute wall limit. The original collector writes its manifest only on full completion, so that final manifest is absent. Do not treat the partial directory as complete.

The test sample (index 3) was not forwarded or used for calibration. No training checkpoint or AE groups exist. No 32-layer KL/L2/RMSE/cosine results or task accuracy are available.

## Measured budget bottleneck

During the same H100 allocation, the short cost probe used layer-0 mean maps from calibration only and discarded its temporary weights. CPU map I/O ran concurrently. Measured mean training step: 0.06422s; validation step: 0.03001s. Estimated 921 training + 103 validation steps per epoch: 62.24s.

Original patience=30 permits stopping no earlier than epoch31: estimated 1929s (32.16min). With the existing 25% planning margin and 900s replay reserve, the estimate is 3312s (55.20min), exceeding the remaining 2813s even before finishing mean-map aggregation. The 1000-epoch maximum extrapolates to about 17.29h, not a prediction that training will actually run that long. These short-probe estimates do not guarantee epoch time or stopping time.

No further job was submitted because the measured plan does not fit the remaining budget. We did not shorten the test, reduce epochs/patience, replace AE clustering, omit methods, or extend resources.

## Next feasible steps requiring a revised plan

Reuse the preserved 96 raw maps instead of repeating model loading/forwards. Complete only missing mean maps, validate the calibration-token fingerprint, and use node-local scratch for active weight/map I/O where capacity permits. This preserves algorithm semantics and addresses the observed storage bottleneck, but is not yet measured to make the entire experiment fit the remaining time.

If that remains insufficient, completing the unchanged five-method experiment requires an explicitly approved budget revision. A shorter AE training schedule or reduced map size would instead change the experiment and also requires approval. No such change has been made.

Evidence: repair_02/final_status.json, repair_02/slurm_terminal.psv, repair_02/partial_artifacts.json, repair_02/source_snapshot.tar.gz, run_repair02/correctness_*.json, run_repair02/ae_budget.json, and run_repair02/ae_collect.log. Existing RESULT.md and earlier outputs are retained as historical records.
