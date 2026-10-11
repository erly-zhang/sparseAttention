"""Freeze an independent W1 Top-p union recent64 smoke; never overwrite a run."""
import datetime,hashlib,json,shutil,sys
from pathlib import Path
E=Path(__file__).resolve().parents[1];root=Path(sys.argv[1]).resolve()
reference=E/'results/full_kv_w16_w64_20260922'
root.mkdir();(root/'source/scripts').mkdir(parents=True)
for name in ('ra_harness','tests'):
    shutil.copytree(E/name,root/'source'/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for name in ('prepare_w1_recent_smoke.py','run_w1_recent_smoke.py','check_w1_recent_smoke.py','w1_recent_smoke.sh'):
    shutil.copyfile(E/'scripts'/name,root/'source/scripts'/name)
shutil.copyfile(reference/'inputs.jsonl',root/'inputs.jsonl')
cfg=json.loads((E/'results/full_kv_w1_p095_smoke_20260924/config.json').read_text())
cfg.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    method='qhead_shared_mask_w1_recent_gpu',experiment='W1_topp095_union_recent64',
    recent_window=64,recent_semantics='union_after_topp_all_nonprompt',random_keep_ratio=None,
    gather_backend='triton_fused_reusable_variable_lengths',
    inference_change='Unchanged historical W1 selector, union recent64, optimized full-KV growth and ragged fused reusable gather',
    reference_root=str(reference),auto_full_submission=False)
(root/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
(root/'README.md').write_text('''# W1 Top-p0.95 union recent64

User-confirmed semantics: prompt union W1 conditional non-prompt Top-p0.95 union the last min(64,R) non-prompt positions. No fixed40% retention budget. Select afresh each layer and decode step, share the selected positions with all heads in that layer; each head computes its own attention weights. Representative head RNG and Top-p scoring/sort use the historical SharedMaskLayer implementation unchanged. W1 holds the current post-RoPE query only. Full KV is retained and previously unselected keys remain eligible. Recency is added AFTER selection across all non-prompt positions, not by reweighting/excluding recent positions first.

Use the random-recent implementation's full-cache growth, fused K/V gather and reusable buffers, adapted to different selected counts per rollout. Padding follows each valid sorted prefix; cache_seqlens excludes padding. Record actual pair sums plus Top-p-core and added-recent decode pair counts.

Run correctness/regression tests, then canonical MATH500 inputs0,1,2, two complete rollouts each. Model Qwen3-4B BF16, seed42, rep_seed0, sampling temperature0.6 and generation Top-p0.95, max total tokens32768, untimed2-token warmup. No per-step JSON during timing. Preserve exact input hashes and preprocessing. Separate synchronized prefill/decode timing, throughput, EOS, output lengths, memory and pair sparsity. Comparisons use same3inputs from W1, W16, W64 and random40recent64. No automatic full500 submission for this smoke.

Frozen source and inputs are tracked in manifest.json. Pipeline status is pipeline_status.json; final report smoke/REPORT.md, summary smoke/summary.json, grades smoke/grades.jsonl. Existing runs and selectors remain unchanged.
''')
manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*')) if p.is_file()}
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(root)
