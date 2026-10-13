# KL continuation prepared; not submitted

The existing total allocation cap remains 3600 seconds. Already used: 787 seconds; remaining: 2813 seconds. No new GPU work or allocation was performed while preparing this continuation. There are no real-test KL values to plot yet.

## Proposed approval

Increase the cumulative cap to 7200 seconds on one H100-80GB, including all earlier attempts. This proposal is NOT authorization. The prepared next job requests 6360 seconds (106 minutes), making the maximum cumulative allocation 7147 seconds, leaving 53 seconds under the proposed cap. Internal deadline6250 seconds; timeout6290 seconds; no requeue. AE convergence and long-context kernels are not guaranteed to finish within this cap.

Account cis260309p, partition GPU-shared, one H100-80GB, eight CPUs,128GB RAM. No full benchmark, additional input or additional GPU. The five algorithms and fixed input3(123959 tokens) remain as recorded in ../experiment_config.json. Original AE max1000 epochs/patience30/seed42/learning rate and K=3 clustering remain unchanged.

## Continuation

1. Reuse all96 completed raw maps from job47071290. Recompute averages in original sample order using float32 sums and float16 outputs, headwise, on compute-node scratch. This avoids rerunning the three calibration model forwards and never treats a partly written mean file as complete. Tiny CPU test confirms bitwise equality with the original averaging; nonfinite maps are rejected.
2. Persist the completed means and clearly marked recovered manifest in run_kl_resume/ae. Run original AE training and clustering, excluding test index3. Reserve30minutes for model staging, clustering and five-method replay. If budget or training completion fails, preserve checkpoint as incomplete and do not cluster it.
3. Stage unchanged model files on node-local scratch and verify their SHA256 against model_provenance.json. Run all five methods with their own end-to-end hidden-state trajectories, uniform warmup129 and full prompt cache.
4. Save dense references and calculate KL(dense||method), feature softmaxT=1, mean over all valid tokens, for attention post-o_proj before residual and full block outputs, each32layers. Also export L2/RMSE/cosine/pairs. Diagnostics are excluded from inference timing.

Expected outputs: ../run_kl_resume/all_layers.jsonl (320 rows), all_layers.csv, attention_post_o_proj.png, block_output.png, dense/tensors, per-method logs/timing, AE groups/checkpoint and provenance. These files do NOT exist yet.

Script: run.sbatch. Submission wrapper: submit.py, requiring an explicit approved cumulative cap, active-job check, atomic submission lock, source snapshot/hash and output-directory guard. It has not been invoked. Existing files/jobs/results remain preserved.

Alternative within the old cap: no reliable complete five-method plan currently fits the measured estimate plus replay allowance and outstanding I/O. Reducing AE training or omitting a method would change the experiment and has not been done.
