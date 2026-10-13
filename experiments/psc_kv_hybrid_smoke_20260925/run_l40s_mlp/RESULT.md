# L40S smoke completed; generation quality failed

User authorized existing vscode-gpu allocation47082259 on gl002 (NVIDIA L40S48GB). The original queued H100 smoke47104554 was cancelled while PENDING, elapsed0seconds, to avoid duplicate work. Other jobs were preserved.

Successful step47082259.9: COMPLETED, exit0:0,62seconds. Fixed InfiniteBench KV Retrieval index3,123959 tokens, original model/tokenizer hashes verified. Hybrid prefill(C0–20,dense21–31), all32layers original dense decode, same full actual prefill KV cache. No KL calculated.

Results:
- Tiny32layer hybrid prefill→4step dense decode test passed.
- Real input generated128tokens with127dense decode calls at every layer, final full KV shapes[1,8,124086,128].
- No EOS reached; stop reason=max_new_tokens.
- Output begins “– 4.5 6d5.6d5...” and repeats. Correct answer:0612a86a-c593-4d21-8c2b-c424e2f94d90.
- Original InfiniteBench get_score_one_kv_retrieval score=0 for this single sample. Not a formal task accuracy estimate.
- Prefill18.289s, decode13.315s, total31.604s; peak allocated40.56GiB. Timing includes greedy selection/cache checks and excludes staging/loading/warmup. L40S timing must not be mixed with H100 timing as a controlled speed comparison.

Repairs:
1. Original GCC/CUDA module environment caused SIGILL during Torch import on gl002. Purging modules and using minimal Python LD_LIBRARY_PATH allowed unchanged Torch2.6+cu124/FA2.7.4/Transformers4.44 to load.
2. Full input hit OOM in original MLP temporary product(3.31GiB requested,3.12GiBfree). In an isolated inference-only adapter, reuse gate storage via gate.mul_(up) before down_proj. Same weights, projection shapes, formula, attention and cache. Tiny BF16 MLP output bitwise equality checked; this is not a claim to have run full-length original-vs-optimized equivalence, which would OOM here. Failed outputs/logs retained.
3. Model staged once and reused after verifying every file SHA. Input length/dtype and generation parameters were not reduced.

Resource accounting: H100 allocation remains4976seconds. L40S model attempts154seconds(OOM) and62seconds(success), plus small import/diagnostic steps recorded in vscode_l40s/slurm_steps.psv. Existing20-hour vscode allocation was not newly requested or counted as a fresh experiment allocation.

The earlier condition “if no problem, run full KV Retrieval” is not satisfied by these outputs. No500-sample evaluation was launched. The root cause of poor generation is not resolved; successful cache checks do not establish that all algorithmic/numerical issues are excluded. Prediction and references were not used to modify routing or tune a cutoff.
