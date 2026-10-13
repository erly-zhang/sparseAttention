# Hybrid C layers0–20 / original dense layers21–31

Added at the user's explicit request after inspecting the full-C index3 curves. Layer numbering is zero-based and inclusive:21 C layers followed by11 original dense-attention layers. The tail's bound original LlamaFlashAttention2 forward functions are restored verbatim. Every layer receives its own trajectory's hidden state; no dense hidden-state injection. Full original Q/K/V projections and prompt KV cache remain. The run is prefill-only, batch1, no padding or chunking, matching the prior comparison; this adapter does not add a generation/decode interface.

This is an exploratory, post-hoc experiment on the same InfiniteBench KV Retrieval index3 (123959 tokens), not an independent held-out evaluation. The cutoff was selected after observing that input's results; no cutoff sweep is included. A smaller KL would establish only closer output distributions for this input, not task accuracy.

## Execution

Slurm job47100018, one PSC H100-80GB, cis260309p / GPU-shared,8CPUs,128GB memory,30minute wall limit, no requeue. Prior cumulative H100 allocation4039seconds; this job's maximum makes5839seconds, below the approved7200seconds. Initial observed status:PENDING(Priority). Current state:Slurm and run/state.json after start.

The pipeline first runs a tiny32-layer GPU integration test covering the exact20/21 boundary, identical21-layer prefix and KV cache against all-C, original dense tail functions, own-prefix input to the first dense layer, complete cache, and causal future invariance. It then stages and rehashes the pinned real model, performs a separate timed prefill and diagnostic prefill, and exports the comparison.

Frozen reference outputs are reused read-only from ../psc_kv_layer_outputs_20260925/run_kl_resume/dense/tensors. The prior five methods and AE are not rerun. New metrics for the first21layers must agree with the saved all-C metrics within recorded numerical tolerances before the summary is accepted.

## Expected artifacts

- run/routing_c_l0_20_dense_l21_31/layers.jsonl:64 new records(32layers × attention-post-o_proj/block-output).
- run/comparison.csv and .jsonl: Dense, full-C and hybrid,192records.
- run/all_six_methods.csv and .jsonl: five original methods plus hybrid,384records.
- run/attention_post_o_proj.png and block_output.png: three-condition overlay, boundary at20.5.
- run/summary.json: whole32-layer, prefix21-layer and tail11-layer mean KL plus final-layer metrics.
- run/correctness.json, prefix_parity.json, reference_provenance.json, model_provenance.json, timing and state.

No results existed at submission. Source hashes/snapshot, prior allocation accounting, configuration and submission record are in this directory. Old experiment files are unchanged.
