# Authorized KL continuation

User approved proceeding after the proposed cumulative two-hour cap, still one H100.

- Job47086997 failed at shell startup after2seconds because the Slurm job environment omitted the budget variable. No model calculation occurred. Previous logs/submission retained.
- Fixed by explicitly recording the approved7200-second cap and prior789-second allocation usage in export_fix/run.sbatch, and exporting the environment explicitly.
- Job47087068 submitted successfully. Latest observed state: PENDING(Priority). Requested one H100-80GB for6360seconds; maximum cumulative7149seconds,51seconds below the approved cap.
- The job will recover mean maps, run original AE/clustering, then compare dense, routingC, Shared6144+member2q4096, FlexPrefill and MInference on the same index3input. Two capture points,32layers,KL(dense||method),T=1.
- Output directory upon start: ../run_kl_resume. Job log: ../slurm-47087068.out. Source snapshot, repaired submission script and record: export_fix/.

No KL values have been generated as of this pending-state update. This file is a submission record; use Slurm and run_kl_resume/state.json for current execution state.

Final update: job47087068 COMPLETED0:0 in3250seconds. All320comparison records verified. See ../run_kl_resume/RESULT.md and slurm_terminal.psv.
