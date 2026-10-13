"""CPU-only preparation: reference all 200 frozen inputs and select 3 disjoint calibrations."""
import json
import os
from pathlib import Path
import random
import numpy as np
from transformers import AutoTokenizer
from protocol import RUN,ORIGIN,ORIGIN_MANIFEST,ORIGIN_AUDIT,TASKS,MAX_NEW
from protocol import configuration,sha,save,official,render,verify_inputs


def dataset_row(path,index):
    with path.open() as f:
        for i,line in enumerate(f):
            if i==index:
                return line.rstrip('\n')
    raise IndexError(index)


def prepare():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('preparation requires GPUs hidden')
    c=configuration()
    if sha(ORIGIN/'inputs/manifest.json') != ORIGIN_MANIFEST or sha(ORIGIN/'independent-audit.json') != ORIGIN_AUDIT:
        raise ValueError('unexpected origin')
    source=json.loads((ORIGIN/'inputs/manifest.json').read_text())
    tok=AutoTokenizer.from_pretrained(c['model'],local_files_only=True)
    templates,get_answer,_=official(ORIGIN/'official')
    directory=RUN/'inputs'
    directory.mkdir(exist_ok=False)
    excluded={(r['task'],r['source_index']) for r in source['records']}
    calibration=[]
    # One disjoint KV, MC, and QA prompt; answers/correctness never guide selection.
    for ti,task in enumerate(TASKS[:3]):
        dataset=ORIGIN/'data'/source['datasets'][task]['file']
        if sha(dataset)!=source['datasets'][task]['sha256']:
            raise ValueError('dataset changed')
        with dataset.open() as f:
            count=sum(1 for _ in f)
        order=[i for i in range(count) if (task,i) not in excluded]
        random.Random(c['calibration']['seed']+ti).shuffle(order)
        scanned=[]
        for index in order:
            raw=dataset_row(dataset,index)
            row=json.loads(raw)
            prompt=tok.apply_chat_template([dict(role='user',content=render(row,task,templates))],
                tokenize=False,add_generation_prompt=True)
            ids=tok.encode(prompt,add_special_tokens=False)
            eligible=len(ids)+MAX_NEW[task]<=source['model_context_limit']
            scanned.append(dict(source_index=index,input_tokens=len(ids),eligible=eligible))
            if not eligible:
                continue
            key='calibration-'+task+'-'+str(index)
            for suffix,content in (('raw',raw),('prompt',prompt)):
                with (directory/(key+'.'+suffix)).open('x') as f:
                    f.write(content)
            np.asarray(ids,dtype='<i8').tofile(directory/(key+'.ids'))
            np.arange(len(ids),dtype='<i8').tofile(directory/(key+'.pos'))
            np.ones(len(ids),dtype='u1').tofile(directory/(key+'.mask'))
            answers=get_answer(row,task)
            item=dict(key=key,task=task,source_index=index,dataset_id=row.get('id',index),
                input_tokens=len(ids),before_truncation=len(ids),answers=answers if isinstance(answers,list) else [answers],
                max_new_tokens=MAX_NEW[task],role='calibration',selection_scan=scanned)
            for suffix,field in {'ids':'input_ids_sha256','pos':'position_ids_sha256','mask':'attention_mask_sha256',
                'prompt':'rendered_prompt_sha256','raw':'raw_row_sha256'}.items():
                item[field]=sha(directory/(key+'.'+suffix))
            calibration.append(item)
            break
        else:
            raise ValueError('no disjoint intact calibration sample for '+task)
    pilot=[]
    for ti,task in enumerate(TASKS):
        rows=sorted((r for r in source['records'] if r['task']==task),key=lambda r:r['source_index'])
        pilot.extend(r['key'] for r in random.Random(c['pilot']['seed']+ti).sample(rows,2))
    m=dict(config_sha256=sha(RUN/'config.json'),origin_manifest_sha256=ORIGIN_MANIFEST,
        origin_audit_sha256=ORIGIN_AUDIT,evaluation=source['records'],calibration=calibration,
        pilot_keys=pilot,evaluation_input_directory=str(ORIGIN/'inputs'),truncation='none',
        calibration_selection='seeded disjoint length-eligible; one per KV, MC, QA; no correctness filtering')
    save(directory/'manifest.json',m)
    verify_inputs()
    print('CPU PREPARATION PASS: 200 historical evaluation references, 8 pilot inputs, 3 disjoint calibrations',flush=True)


if __name__=='__main__':
    prepare()
