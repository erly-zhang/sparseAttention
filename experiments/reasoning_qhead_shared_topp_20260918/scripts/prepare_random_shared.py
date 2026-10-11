"""Freeze a separate random-mask run; reuse the exact 500 reference prompts."""
import argparse,datetime,hashlib,json,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--keep-ratio',type=float,required=True);a=p.parse_args()
assert 0<a.keep_ratio<=1
E=Path(__file__).resolve().parents[1];reference=E/'results/full_kv_w16_w64_20260922';root=a.root.resolve()
old=json.loads((reference/'manifest.json').read_text())
for name in ('inputs.jsonl','config.json'):
    assert hashlib.sha256((reference/name).read_bytes()).hexdigest()==old[name]
root.mkdir()
for name in ('ra_harness','tests','scripts'):
    shutil.copytree(E/name,root/'source'/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for name in ('logs','random'):(root/name).mkdir()
shutil.copyfile(reference/'inputs.jsonl',root/'inputs.jsonl')
config=json.loads((reference/'config.json').read_text())
config.pop('windows');config.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    method='qhead_shared_mask_random',selection_p=None,query_window=None,random_keep_ratio=a.keep_ratio,
    mask_seed=42,rep_seed=0,array_tasks=8,max_concurrent_gpus=8,smoke_sample_ids=[0,1,2],
    random_selection='uniform subset without replacement via private CPU randperm; sorted original positions',
    seed_coordinates='mask: seed42/sample_idx/decode_step/layer/original_rollout; head: rep_seed0/sample_idx/decode_step',
    representative_policy='sample-step Q-head ID shared across layers; fresh draws may repeat; nominal only, no query scoring',
    protection='all prompt tokens only; current reasoning token belongs to uniform candidate set',
    count_rule='prompt_length + ceil(random_keep_ratio * nonprompt_length)',
    reference_inputs=str(reference/'inputs.jsonl'),
    comparison_caveat='fixed reasoning retention ratio, not stepwise Top-p count matching; output lengths can change; no query scoring')
(root/'config.json').write_text(json.dumps(config,indent=2)+'\n')
manifest={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((root/'source').rglob('*')) if f.is_file()}
for name in ('inputs.jsonl','config.json'):manifest[name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(root)
