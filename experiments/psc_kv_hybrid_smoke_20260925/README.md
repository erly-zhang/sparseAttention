> Latest: L40S smoke completed but generated repetitive, incorrect output. See [run_l40s_mlp/RESULT.md](run_l40s_mlp/RESULT.md). The queued H100 smoke was cancelled before running. Earlier submission notes below are historical.

# Hybrid generation smoke, no KL

User requested generation smoke instead of another KL comparison. Prior hybrid job47100018 had already completed; it was preserved and not rerun or cancelled.

Prefill: zero-based layers0–20 use C (position-mod-H grouped causal attention, scatter, explicit repetition, full original o_proj); layers21–31 use original dense attention. Decode: restore original dense attention forward for ALL32layers and continue from the full KV cache produced by this hybrid prefill. No dense prefill cache or hidden states are substituted. This is an explicit single-prefill/single-token-decode driver, not a claim of general chunked-prefill or batched generation support.

Same fixed InfiniteBench KV Retrieval index3,123959 saved token IDs; Llama-3.1-8B-Instruct BF16, batch1, seed42, greedy argmax, max128 new tokens. Stops at any model generation_config EOS ID or128-token cap. Prediction and raw token IDs are retained. This is a smoke test, not an official grader/accuracy run. No KL, layer-output tensor dumps, AE retraining, method sweep or new test input.

Before real-model generation, a tiny32-layer model verifies hybrid prefill and four dense decode steps, full-cache growth and all-layer decode calls. Real model receives a129-token warmup; actual generation uses a fresh cache. Only the final hidden position is passed through lm_head to avoid allocating prompt-length vocabulary logits. Timing excludes model staging/loading/warmup, and includes greedy token selection/cache validation; it is not a pure-kernel benchmark.

Submitted job47104554: one H100-80GB, accountcis260309p, partitionGPU-shared,30minute limit, no requeue. Prior cumulative H100 allocation4976seconds; maximum with this job6776seconds, under approved7200seconds. Current status comes from Slurm and run/state.json.

Expected artifacts: run/correctness.json, generation.json, prediction.txt, smoke.log, environment.json, model_provenance.json and state.json. Source snapshot, hashes, exact configuration, submission lock and prior allocation accounting are stored here. A successful pipeline means the generation loop completed; it does not mean the answer is correct or EOS was reached.
