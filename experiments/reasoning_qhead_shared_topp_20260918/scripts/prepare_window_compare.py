"""Freeze one formal W16/W64 comparison after BOTH diagnostic smokes pass."""
import argparse,hashlib,importlib.metadata,json,os,shutil,sys,datetime
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);ap.add_argument('--smoke16',type=Path,required=True);ap.add_argument('--smoke64',type=Path,required=True);args=ap.parse_args()
E=Path(__file__).resolve().parents[1];root=args.root.resolve()
checks=[]
for window,smoke in ((16,args.smoke16.resolve()),(64,args.smoke64.resolve())):
    summary=json.loads((smoke/'summary.json').read_text())
    assert summary['query_window']==window and summary['samples']==3 and summary['rollouts']==6
    assert summary['full_history_continuity'].startswith('passed') and summary['pair_counter_crosscheck'].startswith('passed')
    assert summary['input_alignment'].startswith('passed') and summary['decode_seconds']>0
    assert json.loads((smoke/'exit.json').read_text())['exit_code']==0
    for rel,digest in json.loads((smoke/'manifest.json').read_text()).items():
        assert hashlib.sha256((smoke/rel).read_bytes()).hexdigest()==digest
        if rel.startswith('source/ra_harness/') or rel.startswith('source/tests/'):
            assert (smoke/rel).read_bytes()==(E/rel.removeprefix('source/')).read_bytes(),rel
    checks.append(dict(window=window,path=str(smoke),summary_sha256=hashlib.sha256((smoke/'summary.json').read_bytes()).hexdigest()))
root.mkdir()
for name in ('ra_harness','tests','scripts'):
    shutil.copytree(E/name,root/'source'/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for name in ('logs','w16','w64'):(root/name).mkdir()
sys.path[:0]=[str(root/'source/ra_harness/kvcompress/harness'),str(root/'source/ra_harness')]
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
from transformers import AutoTokenizer
from Utils.data_loader import load_data
from Utils.parser import parse_question,parse_ground_truth
import numpy as np
model='/scratch/yl258/jz648/Qwen3-4B';data='/project/yl258/jz648/Random-Attention/data'
examples=load_data('math','test',data);assert [e['idx'] for e in examples]==list(range(500))
tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
with (root/'inputs.jsonl').open('x') as stream:
    for e in examples:
        messages=[dict(role='user',content=parse_question(e,'math')+'\nPlease reason step by step, and put your final answer within \\boxed{}.')]
        ids=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True)['input_ids']
        row=dict(idx=e['idx'],input_tokens=len(ids),input_ids_sha256=hashlib.sha256(np.asarray(ids,dtype='<i8').tobytes()).hexdigest(),prompt_text=tok.decode(ids,skip_special_tokens=True),gold=parse_ground_truth(e,'math')[1])
        stream.write(json.dumps(row)+'\n')
geometry=json.loads((Path(model)/'config.json').read_text());geometry=geometry.get('text_config',geometry)
config=dict(model_geometry={k:geometry[k] for k in ('num_hidden_layers','num_attention_heads','num_key_value_heads','head_dim')},model_file_hashes={name:hashlib.sha256((Path(model)/name).read_bytes()).hexdigest() for name in ('config.json','generation_config.json','tokenizer_config.json')},created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),model=model,dataset='MATH500 math/test',dataset_sha256=hashlib.sha256((Path(data)/'math/test.jsonl').read_bytes()).hexdigest(),windows=[16,64],selection_p=.95,seed=42,rep_seed=0,batch_size=2,rollouts=2,max_tokens=32768,num_shards=8,array_tasks=16,max_concurrent_gpus=8,walltime_per_worker_hours=48,warmup_new_tokens=2,cache_semantics='full_history_temporary_mask',generation_temperature=.6,generation_top_p=.95,generation_top_k_applied=False,diagnostic_per_step_logging=False,packages={name:importlib.metadata.version(name) for name in ('torch','transformers','flash-attn','numpy')},timing='prefill forward and decode phase separately synchronized; decode includes selection/gather/attention/sampling/bookkeeping; excludes loading/warmup/metric serialization',comparison_caveat='different windows can change output length; report decode latency together with token throughput and pair totals')
(root/'config.json').write_text(json.dumps(config,indent=2)+'\n')
(root/'smoke_gate.json').write_text(json.dumps(dict(status='passed',checks=checks),indent=2)+'\n')
manifest={}
for p in sorted((root/'source').rglob('*')):
    if p.is_file():manifest[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
for name in ('inputs.jsonl','config.json','smoke_gate.json'):
    manifest[name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(root=str(root),inputs=500,status='frozen; ready for submission'),indent=2))
