"""Freeze the proposed experiment and verify identical historical inputs on CPU."""
import ast
import collections
import os
from pathlib import Path
import subprocess
from support import HERE,OLD,ORIGIN,legacy,read,save,sha


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert not (HERE/'approval.json').exists(), 'Do not alter an approved experiment'
    c=read(HERE/'config.json');old=legacy();prior,m=old.preflight()
    heads=old.load_heads(prior,m)
    assert c['selection']==prior['selection'] and c['model']==prior['model']
    assert c['layers']==prior['layers']==list(range(32)) and c['generation_seed']==prior['generation_seed']
    assert collections.Counter(x['task'] for x in m['evaluation'])==collections.Counter({t:50 for t in c['tasks']})
    source=dict(read(OLD/'source-lock.json')['files'])
    for name in ['config.json','inputs/manifest.json','outputs/calibration/representative-heads.json',
                 'outputs/calibration/complete.json','outputs/independent-audit.json']:
        path=OLD/name;source[str(path)]=sha(path)
    for item in m['evaluation']:
        for suffix in ['ids','pos','mask','prompt','raw']:
            path=ORIGIN/'inputs'/(item['key']+'.'+suffix);source[str(path)]=sha(path)
    references=[]
    for directory in sorted((OLD/'outputs').glob('formal-*')):
        for path in sorted(directory.glob('*.json')):
            if path.name=='complete.json':continue
            row=read(path)
            if 'method' not in row:continue
            item=next(x for x in m['evaluation'] if x['key']==row['key'])
            assert row['input_ids_sha256']==item['input_ids_sha256']
            assert row['max_new_tokens']==item['max_new_tokens'] and row['applied_layers']==list(range(32))
            source[str(path)]=sha(path)
            references.append(dict(path=str(path),method=row['method'],key=row['key'],task=row['task'],score=row['score']))
    assert len(references)==1400 and len({(r['key'],r['method']) for r in references})==1400
    workers=[]
    for method in c['methods']:
        for task in c['tasks']:
            keys=[x['key'] for x in m['evaluation'] if x['task']==task]
            for shard in range(10):
                workers.append(dict(index=len(workers),method=method,task=task,keys=keys[shard*5:(shard+1)*5]))
    sentinels=[next(r for r in references if r['method']==method and r['key']==c['sentinel_key']) for method in c['comparison_methods']]
    assert len(workers)==160 and sum(len(w['keys']) for w in workers)==800
    manifest=read(HERE.parent/'jlens_prefill_analysis_20261005/input-manifest.json')
    save(HERE/'prepared.json',dict(workers=workers,sentinels=sentinels,references=references,heads=heads,
        evaluation=m['evaluation'],source_sha256=source,model_files=manifest['model_files'],gpu_runs=0))
    for path in HERE.glob('*.py'):ast.parse(path.read_text(),filename=str(path))
    for path in list(HERE.glob('*.sh'))+list(HERE.glob('*.sbatch')):subprocess.run(['bash','-n',str(path)],check=True)
    files=list(HERE.glob('*.py'))+list(HERE.glob('*.sh'))+list(HERE.glob('*.sbatch'))
    files += [HERE/'config.json',HERE/'prepared.json',HERE/'PROTOCOL.md']
    save(HERE/'lock.json',dict(files={str(p):sha(p) for p in sorted(files)}))
    print('CPU PREPARATION PASS: 800 evaluations, 160 workers, 1400 paired historical references; no GPU jobs')

if __name__=='__main__':main()
