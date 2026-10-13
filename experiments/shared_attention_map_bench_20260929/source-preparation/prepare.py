"""Download pinned public data, select intact inputs, and freeze once on CPU."""
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import struct
import urllib.request
from protocol import TASKS,METHODS,LAYERS,COUNT,SEED,MAX_NEW,DATA_REV,RUN,PRIOR
from protocol import sha,save,official,render,assign_shards,validate_inputs


def quality():
    path=PRIOR/'quality.py'
    assert sha(path)=='7c66db90d78a43ce19c876669e849414cf685389164c9819cb062cb91a1ffb61'
    spec=importlib.util.spec_from_file_location('bench_quality',path)
    q=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(q)
    q.verify_sources()
    prior=json.loads((PRIOR/'inputs/manifest.json').read_text())
    assert sha(PRIOR/'inputs/manifest.json')=='dc76a7a783ef48c5be21338b3c1c3cb0c8a816470bf7661a232da75057bd35e3'
    for name,value in prior['tokenizer_files'].items():
        assert sha(q.MODEL/name)==value
    return q,prior


def get_json(url):
    req=urllib.request.Request(url,headers={'User-Agent':'PSC-research-frozen-evaluation'})
    with urllib.request.urlopen(req,timeout=120) as response:
        return json.load(response)


def download(url,path,expected=None):
    part=path.with_suffix(path.suffix+'.download')
    with urllib.request.urlopen(url,timeout=180) as response,part.open('xb') as out:
        while chunk:=response.read(1024*1024):
            out.write(chunk)
    if expected is not None:
        assert sha(part)==expected,path
    assert not path.exists()
    part.rename(path)
    return sha(path)


def prepare():
    from transformers import AutoTokenizer
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'CPU preparation only'
    q,prior=quality()
    RUN.mkdir()
    for folder in ('data','official','inputs','source-preparation'):
        (RUN/folder).mkdir()
    source=Path(__file__).resolve().parent
    for name in ('prepare.py','protocol.py'):
        shutil.copyfile(source/name,RUN/'source-preparation'/name)
    repo=get_json('https://api.github.com/repos/OpenBMB/InfiniteBench/commits/main')
    commit=repo['sha']
    assert len(commit)==40 and all(c in '0123456789abcdef' for c in commit)
    official_hashes={}
    for name in ('prompt.py','compute_scores.py','eval_utils.py'):
        official_hashes[name]=download(f'https://raw.githubusercontent.com/OpenBMB/InfiniteBench/{commit}/src/{name}',RUN/'official'/name)
    tree=get_json(f'https://huggingface.co/api/datasets/xinrongzhang2022/InfiniteBench/tree/{DATA_REV}')
    save(RUN/'data/download-metadata.json',tree)
    by_name={r['path']:r for r in tree}
    datasets={}
    for task in TASKS:
        name=task+'.jsonl'
        expected=by_name[name]['lfs']['oid']
        assert len(expected)==64
        path=RUN/'data'/name
        cached=q.BASE/'data'/name if task=='kv_retrieval' else RUN.parent/'head_slot_kv_ablation_20260928/data'/name
        if cached.exists() and sha(cached)==expected:
            shutil.copyfile(cached,path)
        else:
            download(f'https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench/resolve/{DATA_REV}/{name}',path,expected)
        assert sha(path)==expected
        datasets[task]=dict(file=name,sha256=expected,revision=DATA_REV)
        print('DATA READY',task,path.stat().st_size,expected,flush=True)
    config=json.loads((q.MODEL/'config.json').read_text())
    limit=min(config['max_position_embeddings'],131072)
    tok=AutoTokenizer.from_pretrained(q.MODEL,local_files_only=True)
    templates,get_answer,scores=official(RUN/'official')
    # Sanity-check the downloaded scoring API before any model run.
    assert scores['code_debug']('B',['f','B'],'llama')==1
    assert scores['longbook_choice_eng']('C',['choice','C'],'llama')==1
    assert scores['longbook_qa_eng']('the red house',['red house'],'llama')==1
    records=[]
    for t,task in enumerate(TASKS):
        lines=(RUN/'data'/datasets[task]['file']).read_text().splitlines()
        indices=list(range(len(lines)))
        random.Random(SEED+t).shuffle(indices)
        chosen=[]
        scanned=[]
        for index in indices:
            row=json.loads(lines[index])
            prompt=render(row,task,templates)
            rendered=tok.apply_chat_template([dict(role='user',content=prompt)],tokenize=False,add_generation_prompt=True)
            ids=tok.encode(rendered,add_special_tokens=False)
            n=len(ids)
            eligible=n+MAX_NEW[task]<=limit
            scanned.append(dict(source_index=index,input_tokens=n,eligible=eligible))
            if not eligible:
                continue
            key=task+'-'+str(index)
            answers=get_answer(row,task)
            if isinstance(answers,str):
                answers=[answers]
            assert isinstance(answers,list) and answers
            paths={}
            for suffix,data in (('ids',struct.pack('<'+'q'*n,*ids)),
                                ('pos',struct.pack('<'+'q'*n,*range(n))),
                                ('mask',bytes([1])*n),('prompt',rendered.encode()),('raw',lines[index].encode())):
                p=RUN/'inputs'/(key+'.'+suffix)
                with p.open('xb') as f:
                    f.write(data)
                paths[suffix]=sha(p)
            chosen.append(dict(key=key,task=task,source_index=index,dataset_id=row.get('id'),
                input_tokens=n,before_truncation=n,input_ids_sha256=paths['ids'],
                position_ids_sha256=paths['pos'],attention_mask_sha256=paths['mask'],
                rendered_prompt_sha256=paths['prompt'],raw_row_sha256=paths['raw'],
                answers=answers,max_new_tokens=MAX_NEW[task]))
            print('INPUT',task,len(chosen),index,n,flush=True)
            if len(chosen)==COUNT:
                break
        save(RUN/'data'/(task+'-selection.json'),dict(seed=SEED+t,total_rows=len(lines),scanned=scanned,
             chosen=[r['source_index'] for r in chosen],rule='first50 within model limit in seeded random order; no truncation or correctness filtering'))
        assert len(chosen)==COUNT, f'{task}: only {len(chosen)} intact examples fit model context; do not truncate silently'
        records.extend(chosen)
    assignments,costs=assign_shards(records)
    for row in records:
        row['shard']=assignments[row['key']]
    manifest=dict(records=records,datasets=datasets,methods=list(METHODS),layers=list(LAYERS),
        model=str(q.MODEL),model_context_limit=limit,tokenizer_files=prior['tokenizer_files'],
        model_config_sha256=sha(q.MODEL/'config.json'),generation_config_sha256=sha(q.MODEL/'generation_config.json'),
        official_commit=commit,official_hashes=official_hashes,shard_costs=costs,
        preparation_sources={name:sha(source/name) for name in ('prepare.py','protocol.py')},
        truncation='none; complete examples only',seed=SEED,generation_seed=42,
        template='official yarn_mistral user content + original Llama chat template',
        limits='50 intact length-eligible examples per task; subset not full official benchmark',
        max_new_tokens=MAX_NEW,query_block_size=256,prefill_only=True,full_kv=True,dense_decode=True)
    validate_inputs(manifest)
    save(RUN/'inputs/manifest.json',manifest)
    print('FROZEN200',sha(RUN/'inputs/manifest.json'),flush=True)
    for task in TASKS:
        lengths=sorted(r['input_tokens'] for r in records if r['task']==task)
        print('LENGTHS',task,min(lengths),lengths[25],max(lengths),flush=True)


if __name__=='__main__':
    try:
        prepare()
    except Exception as exc:
        if RUN.exists() and not (RUN/'prepare-failure.json').exists():
            save(RUN/'prepare-failure.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
