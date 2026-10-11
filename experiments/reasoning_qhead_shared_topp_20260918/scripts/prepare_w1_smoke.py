"""Freeze W1 using byte-identical inference code from the completed W16/W64 experiment."""
import argparse,datetime,hashlib,json,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();root=a.root.resolve();E=Path(__file__).resolve().parents[1];ref=E/'results/full_kv_w16_w64_20260922'
manifest=json.loads((ref/'manifest.json').read_text())
for rel,h in manifest.items():assert hashlib.sha256((ref/rel).read_bytes()).hexdigest()==h,rel
root.mkdir();(root/'source/scripts').mkdir(parents=True)
for name in ('ra_harness','tests'):shutil.copytree(ref/'source'/name,root/'source'/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
shutil.copyfile(E/'tests/test_shared_mask_w1.py',root/'source/tests/test_shared_mask_w1.py')
for name in ('w1_smoke.sh','run_w1_smoke.py','check_w1_smoke.py','prepare_w1_smoke.py'):shutil.copyfile(E/'scripts'/name,root/'source/scripts'/name)
shutil.copyfile(ref/'inputs.jsonl',root/'inputs.jsonl')
cfg=json.loads((ref/'config.json').read_text());cfg.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),method='qhead_shared_mask_topp',experiment='full_history_shared_topp_W1',query_window=1,windows=[1],scope='Three complete smoke inputs 0,1,2, two rollouts per input, full max_tokens32768',auto_full_submission=False,reference_root=str(ref),query_semantics='Only the current post-RoPE query for selection; overwritten before every decode attention call',inference_change='Only query_window changed from 16/64 to 1; frozen inference files identical',sample_ids=[0,1,2])
for key in ('array_tasks','max_concurrent_gpus','walltime_per_worker_hours','num_shards'):cfg.pop(key,None)
(root/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
identity={rel:h for rel,h in manifest.items() if rel.startswith('source/ra_harness/')}
for rel,h in identity.items():assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==h,rel
(root/'inference_source_identity.json').write_text(json.dumps(dict(reference_root=str(ref),status='identical',files=identity),indent=2)+'\n')
(root/'README.md').write_text('''# W1 shared Top-p smoke

This experiment uses the exact frozen W16/W64 inference code with query_window=1. Each layer/step samples its representative Q head using the existing private RNG and selects over all current cached keys with the CURRENT post-RoPE query only. All prompt positions are protected. Non-prompt probabilities are renormalized over the non-prompt subset, sorted using the unchanged implementation, and the minimum prefix reaching Top-p0.95 is retained. The mask is shared across that layer's heads. Full persistent KV is retained, so unselected old positions remain eligible at later steps. No fixed 40% budget or added recent64 protection is introduced.

W1 is the selector query-buffer size, not the candidate key window. Current Q is updated before selection. During dense prefill the buffer stores the final prompt query; the first decode call overwrites it with its own current query before any sparse selection. Final attention still uses each head's own current query and normal GQA mapping.

Correction to an earlier explanation: the historical W16/W64 selector rescores all keys available at the CURRENT step using its stored query vectors. It does NOT apply a separate original-position causal mask to each historical query. The actual decode attention uses one current query over legal history. With W1, every candidate cached key is legal for the sole current query. Historical code/results are preserved and not silently corrected in this comparison.

Correctness tests specifically check one-slot overwrite, direct current-query Top-p selection, reselection of old KV, GQA attention against an independent reference, batch filtering, and histories beyond1024. The existing shared-mask regression tests are also retained. Only after correctness passes do canonical MATH500 inputs0,1,2 run to EOS or the existing32768 total-length cap, two rollouts each. Qwen3-4B BF16, temperature0.6, generation Top-p0.95, seed42, representative seed0 and untimed two-token warmup are unchanged. Per-step JSON/debug serialization is disabled during timed generation.

Track pipeline_status.json. Completed outputs and synchronized phase timings, input hashes, raw pair totals, memory and EOS outcomes appear under smoke/. The validator writes smoke/summary.json, grades.jsonl and smoke_gate.json. It compares timing/length/sparsity against the SAME three W16/W64 inputs. This smoke does not auto-submit a full500 run.
''')
manifest={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(root.rglob('*')) if f.is_file()}
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(root)
