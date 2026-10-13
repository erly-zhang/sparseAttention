"""CPU-only input preparation. Never loads model weights or starts experiments."""
import ast
from collections import Counter
import json
import os
from pathlib import Path
import struct
import subprocess

from common import HERE,config,legacy,read,save,sha,positions_for


def evidence_positions(tokenizer,prompt,input_ids,answer,radius):
    """Locate the exact answer span as an analysis annotation, never a mask input.

    The benchmark answer is used only for post-hoc diagnostics. Fail closed on
    retokenization mismatch. No text or token IDs are changed.
    """
    start=prompt.find(answer)
    if start < 0:
        return [],'answer_not_found'
    encoded=tokenizer(prompt,add_special_tokens=False,return_offsets_mapping=True)
    if encoded['input_ids'] != input_ids:
        raise ValueError('Evidence annotation retokenized to different input IDs')
    hits=[i for i,(a,b) in enumerate(encoded['offset_mapping'])
          if a < start+len(answer) and b > start]
    selected=set()
    for i in hits:
        selected.update(range(max(0,i-radius),min(len(input_ids),i+radius+1)))
    return sorted(selected),'exact_answer_text_span'


def prepare():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Preparation must hide GPUs')
    if (HERE/'approval.json').exists():
        raise PermissionError('Cannot alter an approved preparation')
    c=config()
    if c['fit']['shards']!=32 or c['replay']['workers_per_method']!=10:
        raise ValueError('Configuration does not match prepared Slurm arrays')
    if c['fit']['source_layers']!=list(range(31)) or c['replay']['layers']!=list(range(32)):
        raise ValueError('Unexpected source or observation layers')
    old=legacy()
    old.preflight()
    historical_config,manifest=old.verify_inputs()
    old.load_heads(historical_config,manifest)
    if c['replay']['methods'] != historical_config['methods']:
        raise ValueError('Changed experimental methods')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(c['model'],local_files_only=True)
    import pyarrow.parquet as pq
    corpus=HERE/'corpus/wikitext-train-0.parquet'
    expected='74da360f23826045b3e6ac6375411fdb15f003030aa74f2596ed08b857cb9212'
    if sha(corpus) != expected:
        raise ValueError('WikiText shard does not match published LFS SHA256')
    prompts=[]
    seen=set()
    for batch in pq.ParquetFile(corpus).iter_batches(batch_size=2048,columns=['text']):
        for text in batch.column(0).to_pylist():
            if len(text.strip()) < c['fit']['min_chars'] or text in seen:
                continue
            ids=tokenizer(text,add_special_tokens=True,truncation=True,
                          max_length=c['fit']['max_seq_len'])['input_ids']
            if len(ids) != c['fit']['max_seq_len']:
                continue
            prompts.append(dict(text=text,input_ids=ids))
            seen.add(text)
            if len(prompts) == c['fit']['prompts']: break
        if len(prompts) == c['fit']['prompts']: break
    if len(prompts) != c['fit']['prompts']:
        raise ValueError('Insufficient independent fitting prompts')
    save(HERE/'fit-prompts.json',prompts)
    evaluation=[]
    input_files={}
    for parent in (Path(c['historical_run'])/'outputs').glob('calibration*'):
        if parent.is_dir():
            input_files.update({str(p):sha(p) for p in parent.rglob('*.json')})
    directory=old.ORIGIN/'inputs'
    for index,item in enumerate(manifest['evaluation']):
        ids=list(struct.unpack('<'+'q'*item['input_tokens'],
                               (directory/(item['key']+'.ids')).read_bytes()))
        prompt=(directory/(item['key']+'.prompt')).read_text()
        if prompt in seen:
            raise ValueError('Exact fit/evaluation text overlap')
        evidence,status=[], 'not_kv_retrieval'
        if item['task']=='kv_retrieval':
            evidence,status=evidence_positions(tokenizer,prompt,ids,item['answers'][0],
                                  c['replay']['kv_retrieval_evidence_radius'])
        positions=positions_for(len(ids),c['replay'],evidence)
        groups=dict(prefix=list(range(min(len(ids),c['replay']['prefix_positions']))),
                    tail=list(range(max(0,len(ids)-c['replay']['tail_positions']),len(ids))),
                    uniform=positions_for(len(ids),dict(prefix_positions=0,tail_positions=0,
                                  uniform_positions=c['replay']['uniform_positions'])),
                    kv_evidence=evidence)
        evaluation.append(dict(item,analysis_positions=positions,position_groups=groups,
                               evidence_status=status,analysis_shard=index % c['replay']['workers_per_method']))
        for suffix in ('ids','pos','mask','prompt','raw'):
            path=directory/(item['key']+'.'+suffix)
            input_files[str(path)]=sha(path)
        if index % 20 == 0: print('Prepared positions',index,flush=True)
    historical_records={}
    for path in sorted((Path(c['historical_run'])/'outputs').glob('formal-*/*.json')):
        record=read(path)
        if 'prediction' not in record: continue
        historical_records[(record['method'],record['key'])]=dict(path=str(path),sha256=sha(path),
                      score=record['score'],logits=str(path.parent/record['logits_file']),
                      logits_sha256=record['logits_sha256'])
    for item in evaluation:
        item['history']={method:historical_records[(method,item['key'])]
                         for method in c['replay']['methods']}
        for history in item['history'].values():
            input_files[history['path']]=history['sha256']
            input_files[history['logits']]=history['logits_sha256']
    eval_files=sorted(Path(c['jlens_repository']).glob('data/evaluations/*.json'))
    if sum(len(read(p)['items']) for p in eval_files) != c['validation']['items']:
        raise ValueError('Original evaluation count changed')
    input_files.update({str(p):sha(p) for p in eval_files})
    input_files[str(corpus)]=sha(corpus)
    input_files[str(HERE/'fit-prompts.json')]=sha(HERE/'fit-prompts.json')
    model_files={}
    for path in sorted(Path(c['model']).iterdir()):
        if path.is_file() and path.suffix in ('.json','.safetensors','.model'):
            stat=path.stat()
            model_files[str(path)]=dict(sha256=sha(path),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)
    print('Pinned model and tokenizer files',len(model_files),flush=True)
    save(HERE/'input-manifest.json',dict(evaluation=evaluation,files=input_files,
         model_files=model_files,
         historical_manifest_sha256=sha(Path(c['historical_run'])/'inputs/manifest.json'),
         fit_prompt_file='fit-prompts.json',fit_prompts=len(prompts),
         fit_selection='first 1000 distinct train-shard records >=600 characters and >=128 tokens',
         fit_eval_exact_text_overlap=0,tasks=dict(Counter(x['task'] for x in evaluation)),
         evidence_annotation='ground-truth text span, diagnostic only; never used by sparse masks',
         truncation=False))
    freeze()


def freeze():
    c=config()
    files=list(HERE.glob('*.py'))+list(HERE.glob('*.sh'))+list(HERE.glob('*.sbatch'))
    files += [HERE/'config.json',HERE/'input-manifest.json',HERE/'EXPERIMENT_PROTOCOL.md']
    files += list(Path(c['jlens_repository']).glob('jlens/*.py'))
    files += list(Path(c['jlens_repository']).glob('jlens/data/*'))
    prior=read(Path(c['historical_run'])/'source-lock.json')
    files += [Path(p) for p in prior['files']]
    files += [Path(c['historical_run'])/'source-lock.json']
    for p in HERE.glob('*.py'): ast.parse(p.read_text(),filename=str(p))
    for p in list(HERE.glob('*.sh'))+list(HERE.glob('*.sbatch')):
        subprocess.run(['bash','-n',str(p)],check=True)
    import importlib.metadata as metadata
    save(HERE/'source-lock.json',dict(files={str(p.resolve()):sha(p) for p in sorted(set(files))},
       versions={n:metadata.version(n) for n in ['torch','transformers','flash-attn','triton','numpy','pyarrow','pytest']},
       jlens_commit='581d398613e5602a5af361e1c34d3a92ea82ba8e',
       original_fitting_modified=False,model_execution=False,experiment_submitted=False))


if __name__=='__main__': prepare()
