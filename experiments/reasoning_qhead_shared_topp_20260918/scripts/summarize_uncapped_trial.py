"""Validate a completed bounded pilot; report actual pair totals, never smoke accuracy."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

parser=argparse.ArgumentParser()
parser.add_argument('trial', type=Path)
args=parser.parse_args()
out=args.trial.resolve()
config=json.loads((out/'config.json').read_text())
assert json.loads((out/'exit.json').read_text())['exit_code']==0
for rel,digest in json.loads((out/'manifest.json').read_text()).items():
    assert hashlib.sha256((out/rel).read_bytes()).hexdigest()==digest, rel
sys.path[:0]=[str(out/'source/ra_harness/kvcompress/harness'), str(out/'source/ra_harness')]
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
from transformers import AutoTokenizer
from Utils.data_loader import load_data
from Utils.parser import parse_question
import numpy as np

examples=load_data('math','test','/project/yl258/jz648/Random-Attention/data')
tokenizer=AutoTokenizer.from_pretrained(config['model'],local_files_only=True)
expected=set(config['sample_ids'])
metrics=[json.loads(l) for l in (out/'cache_metrics.jsonl').open()]
assert len(metrics)==len(expected) and {r['idx'] for r in metrics}==expected
prompt_text={}
for record in metrics:
    idx=record['idx']
    messages=[dict(role='user',content=parse_question(examples[idx],'math')+'\nPlease reason step by step, and put your final answer within \\boxed{}.')]
    ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True)['input_ids']
    assert len(ids)==record['input_tokens']
    assert hashlib.sha256(np.asarray(ids,dtype='<i8').tobytes()).hexdigest()==record['input_ids_sha256']
    prompt_text[idx]=tokenizer.decode(ids,skip_special_tokens=True)
    assert record['selected_pairs']<=record['causal_pairs']
for run in (0,1):
    records=[json.loads(l) for l in (out/f'run_{run}/completions_shard0.jsonl').open()]
    assert len(records)==len(expected) and {r['idx'] for r in records}==expected
    assert all(r['completion'].startswith(prompt_text[r['idx']]) for r in records)

n=0;kept=0;candidates=0;reason_kept=0;reason_candidates=0;min_mass=1.;max_visible=0
previous={}
trace_selected=0; trace_causal=0
model_config=json.loads((Path(config['model'])/'config.json').read_text())
model_config=model_config.get('text_config', model_config)
hq=model_config['num_attention_heads']; layers=model_config['num_hidden_layers']
for m in metrics:
    prefill=2*layers*hq*m['input_tokens']*(m['input_tokens']+1)//2
    trace_selected+=prefill;trace_causal+=prefill
excluded_regression_records=0
for line in (out/'qhead_selections.jsonl').open():
    r=json.loads(line)
    if r['sample_id'] not in expected:
        assert r['sample_id'] in (None, 5), 'unexpected sample ID'
        excluded_regression_records+=1
        continue
    n+=1
    assert r['implementation']=='row_safe_v2' and r['token_budget']==0 and r['p']==.95
    assert r['total_visible_tokens']<=r['cache_capacity']
    assert 0<=r['selected_reasoning_K']<=r['reasoning_candidates']
    if r['reasoning_candidates']: assert r['selected_reasoning_mass']>=.95-1e-5
    key=(r['sample_id'],r['rollout'],r['layer'])
    if key in previous:
        old=previous[key]
        assert r['step']==old['step']+config['eviction_interval'], 'duplicate or missing decode event'
        assert r['total_visible_tokens']==old['prompt_len']+old['selected_reasoning_K']+r['step']-old['step']
    else:
        assert r['step']==r['prompt_len']+1, 'missing first decode event'
    previous[key]=r
    trace_selected+=hq*r['total_visible_tokens'];trace_causal+=hq*r['step']
    min_mass=min(min_mass,r['selected_reasoning_mass'])
    candidates+=r['total_visible_tokens'];kept+=r['prompt_len']+r['selected_reasoning_K']
    reason_candidates+=r['reasoning_candidates'];reason_kept+=r['selected_reasoning_K']
    max_visible=max(max_visible,r['total_visible_tokens'])
selected=sum(r['selected_pairs'] for r in metrics);causal=sum(r['causal_pairs'] for r in metrics)
assert selected==trace_selected and causal==trace_causal, 'pair counters disagree with independent telemetry reconstruction'
result=dict(pair_counter_crosscheck='passed against per-event length reconstruction',scope='bounded integration smoke; not a formal accuracy evaluation',
            samples=len(expected),rollouts=2*len(expected),max_tokens=config['max_tokens'],
            eos_finished_rollouts=sum(sum(r['eos_reached']) for r in metrics) if all('eos_reached' in r for r in metrics) else None,
            input_alignment='passed/full for trial inputs',selection_records=n,excluded_regression_records=excluded_regression_records,
            selected_pairs=selected,causal_pairs=causal,global_pair_sparsity=1-selected/causal,
            event_overall_sparsity=1-kept/candidates,event_reasoning_sparsity=1-reason_kept/reason_candidates,
            min_selected_reasoning_mass=min_mass,max_visible_tokens=max(r['max_visible_tokens'] for r in metrics),
            peak_memory_allocated=max(r['peak_memory_allocated'] for r in metrics),
            batch_wall_seconds=sum(r['batch_wall_seconds'] for r in metrics),
            telemetry_length_continuity='passed',source_manifest='passed',
            timing_note='includes assertions and per-event JSON logging; not a speed benchmark')
with (out/'summary.json').open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
print(json.dumps(result,indent=2))
