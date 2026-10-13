# PSC repair continuation

User renewed authorization: 自行进行修复并继续实验. Same five methods, fixed test index 3 and original AE procedure; cumulative allocation budget remains 3600 seconds on one H100. Earlier RESULT.md records the first attempt and is preserved as history.

- 47069472: FAILED 1:0, 76 seconds; original MInference kernel triggered Triton 3.2 LLVM assertion.
- 47070792: FAILED 1:0, 58 seconds; isolated Triton 3.0 compiled original MInference kernel offline, but AutoBlock JIT rejected math.sqrt(D).
- Repair 02: two original selector modules copied into repair_02. Only compile-time math.sqrt(D) expressions replaced by D ** 0.5 (same double value at D=128). Original tracked modules unchanged. All four affected kernels compiled offline for SM90 without CUDA visibility.
- 47071290: submitted, wall limit 3240 seconds (54 minutes), no requeue. Prior cumulative 134 seconds; maximum cumulative allocation 3374 seconds, leaving 226 seconds under original budget.
- All method warmups increased uniformly to 129 tokens, because MInference uses last_q=64. The fixed 123959-token test input is unchanged.

Output: run_repair02. Submission, source snapshot, source hashes, offline compilation logs and compatibility changes: repair_02. Current state is recorded by Slurm and run_repair02/state.json when the allocation starts. No full-model output comparison has been produced yet.

Note: pipeline initial state has a legacy max_wall_seconds=3300 label; authoritative job limit is repair_02/run.sbatch (3240 seconds), with pipeline internal deadline 3160 seconds and process timeout 3190 seconds.

Final update: 47071290 FAILED 1:0 after 653 seconds in collection I/O timeout. All five correctness checks passed. Cumulative H100 allocation 787 seconds. See REPAIR_RESULT.md and repair_02/final_status.json for measured budget blockage and preserved artifacts.
