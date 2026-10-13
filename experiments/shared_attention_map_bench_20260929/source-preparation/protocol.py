"""Frozen two-method, four-task matched-input evaluation contract."""
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import string

TASKS=('kv_retrieval','longbook_choice_eng','longbook_qa_eng','code_debug')
METHODS=('dense_full','shared_map')
LAYERS=tuple(range(0,32,2))
COUNT=50
WORKERS=4
SEED=2026092924
MAX_NEW={'kv_retrieval':50,'longbook_choice_eng':40,'longbook_qa_eng':40,'code_debug':5}
DATA_REV='90f0394333616266d9fe85824ceaf505093cbaa5'
RUN=Path('/ocean/projects/cis260309p/jzhang114/sparseAttention/experiments/shared_attention_map_bench_20260929')
PRIOR=RUN.parent/'psc_kl_quality_retry02_20260927'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def save(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)
        f.flush()
        os.fsync(f.fileno())


def official(directory):
    directory=Path(directory)
    tree=ast.parse((directory/'prompt.py').read_text())
    assignment=next(n for n in tree.body if isinstance(n,ast.Assign) and any(
        isinstance(t,ast.Name) and t.id=='yarn_mistral_templates' for t in n.targets))
    templates=ast.literal_eval(assignment.value)
    names={'normalize_answer','f1_score','qa_f1_score'}|{'get_score_one_'+t for t in TASKS}
    tree=ast.parse((directory/'compute_scores.py').read_text())
    funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in funcs}==names
    ns=dict(re=re,string=string,Counter=Counter)
    exec(compile(ast.Module(body=funcs,type_ignores=[]),'pinned_official_scorers','exec'),ns)
    tree=ast.parse((directory/'eval_utils.py').read_text())
    answer=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_answer')
    exec(compile(ast.Module(body=[answer],type_ignores=[]),'pinned_official_answers','exec'),ns)
    return templates,ns['get_answer'],{t:ns['get_score_one_'+t] for t in TASKS}


def render(row,task,templates):
    context=row.get('context',row.get('content'))
    assert isinstance(context,str) and context
    fields=dict(context=context,input=row.get('input',''),question=row.get('input',''))
    if task in ('longbook_choice_eng','code_debug'):
        assert len(row['options'])==4 and len(set(row['options']))==4
        fields.update({'OPTION_'+letter:value for letter,value in zip('ABCD',row['options'])})
    return templates[task].format(**fields)


def assign_shards(records):
    assert len(records)==COUNT*len(TASKS)
    loads=[0]*WORKERS
    sizes=[0]*WORKERS
    assignments={}
    for row in sorted(records,key=lambda r:(-r['input_tokens']**2,r['task'],r['source_index'])):
        shard=min((i for i in range(WORKERS) if sizes[i]<50),key=lambda i:(loads[i],i))
        assignments[row['key']]=shard
        loads[shard]+=row['input_tokens']**2
        sizes[shard]+=1
    return assignments,loads


def validate_inputs(manifest):
    rows=manifest['records']
    assert Counter(r['task'] for r in rows)==Counter({t:COUNT for t in TASKS})
    assert len({r['key'] for r in rows})==200
    assert len({(r['task'],r['source_index']) for r in rows})==200
    assert len({(r['task'],r['input_ids_sha256']) for r in rows})==200
    assert manifest['methods']==list(METHODS) and manifest['layers']==list(LAYERS)
    assert manifest['truncation']=='none; complete examples only'
    for r in rows:
        assert r['input_tokens']==r['before_truncation']>0
        assert r['input_tokens']+MAX_NEW[r['task']]<=manifest['model_context_limit']
        assert r['max_new_tokens']==MAX_NEW[r['task']]
        assert r['key']==r['task']+'-'+str(r['source_index'])
    assignments,loads=assign_shards(rows)
    assert all(r['shard']==assignments[r['key']] for r in rows)
    assert manifest['shard_costs']==loads


def validate_pair(rows,item,manifest_hash,config_hash):
    assert len(rows)==2 and {r['method'] for r in rows}==set(METHODS)
    for row in rows:
        for field in ('key','task','source_index','dataset_id','input_tokens','input_ids_sha256',
                      'position_ids_sha256','attention_mask_sha256','rendered_prompt_sha256',
                      'raw_row_sha256','answers','max_new_tokens','shard'):
            assert row[field]==item[field],field
        assert row['manifest_sha256']==manifest_hash and row['config_sha256']==config_hash
        assert row['finite_all_steps'] and row['full_kv_cache'] and row['dense_decode']
        ids=row['generated_ids']
        assert 1<=len(ids)<=item['max_new_tokens']
        assert row['stop_reason']==('eos' if ids[-1] in row['eos_ids'] else 'max_new_tokens')
        assert row['stop_reason']=='eos' or len(ids)==item['max_new_tokens']
        assert not any(i in row['eos_ids'] for i in ids[:-1])
        assert row['decode_calls']=={str(i):len(ids)-1 for i in LAYERS}
        n=item['input_tokens']
        assert set(row['prefill'])==set(map(str,LAYERS))
        for stat in row['prefill'].values():
            assert stat['mode']==row['method'] and stat['full_maps_computed']==32
            assert stat['score_elements']==32*n*n and stat['full_causal_pairs']==32*n*(n+1)//2
            assert stat['all_keys_per_block']==n and stat['max_row_sum_error']<=2e-5
            assert stat['rows_per_head']==[[n//32+int(h<n%32)] for h in range(32)]
    for field in ('eos_ids','node','gpu','torch','transformers','generation_seed'):
        assert rows[0][field]==rows[1][field],field


def repetition(ids):
    grams=[tuple(ids[i:i+4]) for i in range(max(0,len(ids)-3))]
    fraction=1-len(set(grams))/len(grams) if grams else 0.0
    tail=ids[-32:]
    period=next((p for p in range(1,9) if len(tail)==32 and all(tail[i]==tail[i%p] for i in range(32))),None)
    return dict(repeated_4gram_fraction=fraction,suffix_period=period,flagged=len(ids)>=32 and (period is not None or fraction>=0.8))
