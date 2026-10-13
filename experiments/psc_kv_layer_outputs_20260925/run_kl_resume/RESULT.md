# Completed PSC KL comparison

Job47087068: COMPLETED, exit0:0; elapsed3250 seconds (54m10s). Cumulative H100 allocation across all five attempts: 4039 seconds (67m19s), under approved7200seconds.

Verified320 unique records: five methods ×32 layers ×two capture points. All reuse input index3,123959 valid tokens, identical input SHA256. AE completed38epochs, best epoch8; original patience30.

KL is KL(dense||method), feature softmax T=1, averaged across valid tokens. Table values additionally average the32 layers. Each method used its own end-to-end trajectory.

| Method | Mean attention-output KL | Mean block-output KL |
|---|---:|---:|
| dense | 0 | 0 |
| routing_c | 0.011384423 | 5.1145698 |
| shared6144_member2q4096 | 9.5153579e-05 | 0.0035913847 |
| flexprefill | 6.1108262e-05 | 0.0023133647 |
| minference | 0.00027386661 | 0.015013236 |

Files: all_layers.csv, all_layers.jsonl, attention_post_o_proj.png, block_output.png. AE provenance, original checkpoint, groups, per-method timing and dense reference tensors are preserved in this output directory.

These are feature-distribution differences on a single fixed input, not task accuracy. Routing C shows substantially larger divergence on this input; this alone does not establish general task effectiveness. Earlier budget estimates were conservative: observed AE training1052.47seconds after scratch staging, versus the earlier short-probe estimate. No training configuration was reduced.
