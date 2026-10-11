"""Freeze ONLY the user-requested optimized random-mask three-input smoke."""
import argparse,datetime,hashlib,json,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
E=Path(__file__).resolve().parents[1];reference=E/'results/full_kv_random_r040_20260923';root=a.root.resolve()
manifest=json.loads((reference/'manifest.json').read_text())
for name in ('inputs.jsonl','config.json'):assert hashlib.sha256((reference/name).read_bytes()).hexdigest()==manifest[name]
root.mkdir();(root/'logs').mkdir()
for name in ('ra_harness','tests','scripts'):shutil.copytree(E/name,root/'source'/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
shutil.copyfile(reference/'inputs.jsonl',root/'inputs.jsonl')
cfg=json.loads((reference/'config.json').read_text());cfg.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    method='qhead_shared_mask_random_recent_gpu',recent_window=64,recent_inside_budget=True,
    random_selection='GPU 32-bit uniform primary score, independent secondary random key, position as final exact tie-break; TopK eligible old reasoning positions',
    seed_coordinates='mask: seed42/sample_idx/decode_step/layer/original_rollout, salt 0x475055524543; head rep_seed0/sample_idx/decode_step',
    protection='full prompt + min(64,ceil(.4*R)) most recent reasoning positions, INSIDE 40% budget',
    gather_backend='Triton K/V fused gather directly into reusable buffers; dynamic lengths/strides do not recompile per step',
    auto_full_submission=False,scope='three complete MATH500 inputs 0,1,2; full 32768 total length; no new full-run submission',
    comparison_caveat='three changes combined; compare length, EOS and throughput; uniform old-only random mask was user-stopped before first output')
(root/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
m={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((root/'source').rglob('*')) if f.is_file()}
for name in ('inputs.jsonl','config.json'):m[name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
(root/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(root)
