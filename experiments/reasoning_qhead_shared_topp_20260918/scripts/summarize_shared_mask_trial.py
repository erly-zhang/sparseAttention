"""Validate completed full-history/shared-mask smoke outputs and reconstruct pairs."""
import argparse,hashlib,json,os,sys
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('trial',type=Path);args=parser.parse_args()
out=args.trial.resolve();config=json.loads((out/'config.json').read_text())
assert json.loads((out/'exit.json').read_text())['exit_code']==0
assert config['cache_semantics']=='full_history_temporary_mask'
for rel,digest in json.loads((out/'manifest.json').read_text()).items():
    assert hashlib.sha256((out/rel).read_bytes()).hexdigest()==digest,rel
sys.path[:0]=[str(out/'source/ra_harness/kvcompress/harness'),str(out/'source/ra_harness')]
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
from transformers import AutoTokenizer
from Utils.data_loader import load_data
from Utils.parser import parse_question
import numpy as np
examples=load_data('math','test','/project/yl258/jz648/Random-Attention/data')
tok=AutoTokenizer.from_pretrained(config['model'],local_files_only=True)
metrics=[json.loads(l) for l in (out/'cache_metrics.jsonl').open()]
expected=set(config['sample_ids']);assert len(metrics)==len(expected) and {r['idx'] for r in metrics}==expected
model_config=json.loads((Path(config['model'])/'config.json').read_text());model_config=model_config.get('text_config',model_config)
Hq=model_config['num_attention_heads'];layers=model_config['num_hidden_layers']
selected=0;causal=0;prompts={}
for r in metrics:
    assert r['cache_semantics']=='full_history_temporary_mask'
    assert r['query_window']==config.get('query_window',64)
    assert r['prefill_seconds']>0 and r['decode_seconds']>0
    assert r['prefill_selected_pairs']+r['decode_selected_pairs']==r['selected_pairs']
    assert r['prefill_causal_pairs']+r['decode_causal_pairs']==r['causal_pairs']
    assert r['decode_query_tokens']==sum(r['generated_tokens_per_rollout'])-2
    messages=[dict(role='user',content=parse_question(examples[r['idx']],'math')+'\nPlease reason step by step, and put your final answer within \\boxed{}.')]
    ids=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True)['input_ids']
    assert len(ids)==r['input_tokens']
    assert hashlib.sha256(np.asarray(ids,dtype='<i8').tobytes()).hexdigest()==r['input_ids_sha256']
    prompts[r['idx']]=tok.decode(ids,skip_special_tokens=True)
    for lengths,cumulative in zip(r['final_cache_seqlens'],r['final_cumulative_length']):
        assert all(n==cumulative for n in lengths),'persistent history was shortened'
    prefill=2*layers*Hq*len(ids)*(len(ids)+1)//2;selected+=prefill;causal+=prefill
for run in (0,1):
    records=[json.loads(l) for l in (out/f'run_{run}/completions_shard0.jsonl').open()]
    assert len(records)==len(expected) and {r['idx'] for r in records}==expected
    assert all(r['completion'].startswith(prompts[r['idx']]) for r in records)
previous={};records=0;min_mass=1.;max_full=0;max_selected=0
for line in (out/'qhead_selections.jsonl').open():
    r=json.loads(line);records+=1
    assert r['implementation']=='full_kv_mask_v1' and r['sample_id'] in expected
    assert r['query_window']==config.get('query_window',64)
    assert r['real_query_count']==min(r['query_window'],r['step'])
    assert r['full_cache_tokens']==r['step']==r['total_visible_tokens']
    assert r['reasoning_candidates']==r['step']-r['prompt_len']
    assert r['full_cache_tokens']<=r['cache_capacity']
    assert r['prompt_len']<=r['selected_tokens']<=r['full_cache_tokens']
    assert r['selected_reasoning_mass']>=config['p']-1e-5
    key=(r['sample_id'],r['rollout'],r['layer'])
    assert r['step']==previous.get(key,r['prompt_len'])+1,'duplicate or missing decode event'
    previous[key]=r['step'];selected+=Hq*r['selected_tokens'];causal+=Hq*r['full_cache_tokens']
    min_mass=min(min_mass,r['selected_reasoning_mass']);max_full=max(max_full,r['full_cache_tokens']);max_selected=max(max_selected,r['selected_tokens'])
assert selected==sum(r['selected_pairs'] for r in metrics)
assert causal==sum(r['causal_pairs'] for r in metrics)
result=dict(scope='bounded engineering smoke; no full-answer accuracy claim',cache_semantics='full_history_temporary_mask',
            samples=len(expected),rollouts=2*len(expected),max_tokens=config['max_tokens'],query_window=config.get('query_window',64),
            source_manifest='passed',input_alignment='passed/full for trial inputs',
            full_history_continuity='passed at every decode event and final cache',
            pair_counter_crosscheck='passed against per-event independent reconstruction',
            selection_records=records,selected_pairs=selected,causal_pairs=causal,global_pair_sparsity=1-selected/causal,
            min_selected_reasoning_mass=min_mass,max_full_cache_tokens=max_full,max_selected_tokens=max_selected,
            eos_finished_rollouts=sum(sum(r['eos_reached']) for r in metrics),
            prefill_seconds=sum(r['prefill_seconds'] for r in metrics),decode_seconds=sum(r['decode_seconds'] for r in metrics),
            decode_selected_pairs=sum(r['decode_selected_pairs'] for r in metrics),decode_causal_pairs=sum(r['decode_causal_pairs'] for r in metrics),
            batch_wall_seconds=sum(r['batch_wall_seconds'] for r in metrics),peak_memory_allocated=max(r['peak_memory_allocated'] for r in metrics))
with (out/'summary.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result,indent=2))
