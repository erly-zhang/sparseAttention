import json,time
from pathlib import Path
from common import HERE,SOURCE,METHODS,TASKS,read,save,sha

def main():
    audit=read(SOURCE/'outputs/independent-audit.json')
    if audit['status']!='PASS' or audit['records']!=1400:raise ValueError('Incomplete source')
    original=read(SOURCE/'input-manifest.json');c=read(SOURCE/'config.json')
    records=[];started=time.time()
    for method in METHODS:
        for item in original['evaluation']:
            d=SOURCE/'outputs'/('replay-'+method+'-'+str(item['analysis_shard']))
            p=d/(item['key']+'.json');raw=p.read_bytes()
            import hashlib
            if hashlib.sha256(raw).hexdigest()!=read(d/'complete.json')['files'][p.name]:raise ValueError('Result changed')
            v=json.loads(raw);a=d/v['artifact_file'];s=a.stat()
            if v['positions']!=item['analysis_positions'] or v['lens_sha256']!=audit['lens_sha256']:raise ValueError('Alignment changed')
            records.append(dict(key=item['key'],task=item['task'],method=method,
                positions=v['positions'],position_groups=v['position_groups'],artifact=str(a),
                artifact_sha256=v['artifact_sha256'],artifact_bytes=s.st_size,artifact_mtime_ns=s.st_mtime_ns,
                result=str(p),result_sha256=hashlib.sha256(raw).hexdigest(),
                historical_score=v['historical_score'],history_first_token_equal=v['history_replay_match']['first_token_equal']))
        print('INDEXED',method,len(records),'seconds',round(time.time()-started,1),flush=True)
    paths=[SOURCE/'outputs/lens.pt',Path(c['model'])/'model-00004-of-00004.safetensors']
    metadata={str(p):dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,
        sha256=audit['lens_sha256'] if p==paths[0] else original['model_files'][str(p)]['sha256']) for p in paths}
    if sha(paths[0])!=audit['lens_sha256']:raise ValueError('Lens changed')
    save(HERE/'manifest.json',dict(source=str(SOURCE),source_audit=audit,model=c['model'],
        lens=str(paths[0]),lens_sha256=audit['lens_sha256'],metadata=metadata,
        methods=METHODS,tasks=TASKS,layers=list(range(32)),k=[16,25],records=records,
        algorithm='nonnegative gradient pursuit; one optimal directional step per added atom, backtracking for nonincrease',
        dictionary='unit-normalized rows of (lm_head.weight * final_RMSNorm.weight) @ J_l; J_31=I',
        random_control='one layer-specific Haar orthogonal rotation of the entire dictionary, same vocabulary size and identical atom Gram matrix',
        seed=42,artifact_hash_verification='Original audit reused; verify_sources.py independently rechecks all bytes before completion'))
    files=[p for p in HERE.iterdir() if p.suffix in ['.py','.sbatch','.sh']]+[HERE/'manifest.json',SOURCE/'outputs/independent-audit.json',SOURCE/'outputs/summary.json',SOURCE/'outputs/lens-validation.json']
    save(HERE/'lock.json',dict(files={str(p):sha(p) for p in files}))
    save(HERE/'authorization.json',dict(approved=True,user_instruction='进行可视化和分解，我想知道在这几个任务上，经过稀疏后的LLM在推理时与dense有什么区别',scope='Offline visualizations and J-space decomposition of all saved prefill observations; no new model forward or fitting',gpu_budget=None,lock_sha256=sha(HERE/'lock.json')))
    print('PREPARATION COMPLETE',len(records),flush=True)

if __name__=='__main__':main()
