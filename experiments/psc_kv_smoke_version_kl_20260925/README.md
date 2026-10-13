# Per-layer KL for the exact L40S smoke version

User requested another per-layer dense comparison after the hybrid smoke generated repetitive text. This diagnostic uses one fixed InfiniteBench KV Retrieval input(index3,123959tokens), BF16 Llama-3.1-8B-Instruct, seed42, batch1. Both dense and hybrid run on the same existing L40S allocation47082259(step12), with identical inference-only MLP buffer reuse from the successful smoke. No new allocation, full-task evaluation, decode KL or additional generation is included.

Dense: original attention in every layer. Hybrid: C in zero-based layers0–20 inclusive, original dense attention in21–31. Both follow their own end-to-end hidden-state trajectories. Both retain all prompt KV heads/tokens in the full32-layer cache. The final cache shape is asserted. Original position IDs and saved input IDs are reused. The model weight/tokenizer file SHA256 values are verified against pinned provenance before loading.

Two capture points: attention after full o_proj and before residual; full Transformer block output before final model norm. At each valid token, feature softmax temperature1 defines the distributions, then compute KL(dense||hybrid) in nats and average across tokens. All123959tokens are valid; no padding. RelativeL2,RMSE and token cosine use the same existing metric implementation and zero-norm handling. This is a feature-output diagnostic, not attention-probability KL or answer accuracy.

The cutoff was chosen after inspecting index3, so this remains post-hoc exploratory analysis. A change in output KL does not identify the cause of repetitive generation by itself.

Expected outputs in run/:all_layers.csv/jsonl(128records), dense_layers.jsonl, hybrid_smoke_layers.jsonl, summary.json, layerwise_kl.png, dense_tensors/(64reference tensors), provenance.json and state.json. Incremental *_partial.jsonl files are incomplete until state.json says complete. Source snapshot/hash and config.json live in this directory. Prior H100 KL and L40S smoke files remain unchanged.

Elapsed diagnostics include output tensor transfers, disk I/O and CPU metric calculations. They are not pure inference latency or a controlled H100–L40S speed comparison. Execution timeout1500seconds.
