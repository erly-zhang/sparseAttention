"""Resume from completed raw maps; unchanged AE and five end-to-end methods.
No submission is performed by this module. Requires an explicit cumulative cap.
"""
import hashlib,json,os,shutil,subprocess,sys,tempfile,time,traceback
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parents[1]; ROOT=HERE.parents[1]
RUN=HERE/'run_kl_resume'; PRIOR=int(os.environ.get('PSC_PRIOR_ALLOCATION_SECONDS','787'))
METHODS=('dense','routing_c','shared6144_member2q4096','flexprefill','minference')

def save(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')

def mean_heads(paths,destination,expected_shape=(32,1936,1936)):
    """Exact original sample order / float32 sum / float16 output, headwise I/O."""
    arrays=[np.load(p,mmap_mode='r') for p in paths]
    assert len(arrays)==3
    assert all(a.shape==expected_shape and a.dtype==np.float16 for a in arrays)
    part=destination.with_suffix('.partial.npy')
    result=np.lib.format.open_memmap(part,mode='w+',dtype=np.float16,shape=expected_shape)
    for h in range(expected_shape[0]):
        value=np.zeros(expected_shape[1:],dtype=np.float32)
        for a in arrays:
            head=a[h].astype(np.float32)
            if not np.isfinite(head).all(): raise ValueError('nonfinite calibration map')
            value+=head
        value/=len(arrays)
        result[h]=value.astype(np.float16)
    result.flush(); del result
    part.replace(destination)

def main():
    cap=int(os.environ.get('PSC_APPROVED_TOTAL_SECONDS','3600'))
    seconds=int(os.environ.get('PSC_PIPELINE_SECONDS','0'))
    assert 0<seconds<=cap-PRIOR-60, 'deadline exceeds approved remaining budget'
    assert os.environ.get('SLURM_JOB_ID'), 'compute allocation required'
    import torch
    assert torch.cuda.device_count()==1 and 'H100' in torch.cuda.get_device_name()
    RUN.mkdir(exist_ok=False); start=time.monotonic();deadline=start+seconds
    os.environ['PSC_RUN_DIR']=str(RUN)
    def invoke(name,args,limit=None):
        remaining=deadline-time.monotonic()
        if limit is not None:remaining=min(remaining,limit)
        if remaining<=0:raise TimeoutError('allocation budget exhausted')
        begin=time.monotonic()
        with (RUN/(name+'.log')).open('x') as f:
            subprocess.run(args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=remaining)
        with (RUN/'stages.jsonl').open('a') as f:
            f.write(json.dumps({'stage':name,'seconds':time.monotonic()-begin})+'\n')
    save(RUN/'state.json',{'status':'running','job_id':os.environ['SLURM_JOB_ID'],'approved_cumulative_seconds':cap,'prior_seconds':PRIOR})
    try:
        base=Path(os.environ.get('LOCAL') or os.environ.get('TMPDIR') or '/tmp')
        assert shutil.disk_usage(base).free>30*1024**3, 'need 30 GiB local space for mean maps and model'
        scratch=Path(tempfile.mkdtemp(prefix='psc-kv-'+os.environ['SLURM_JOB_ID']+'-',dir=base))
        ae=scratch/'ae';means=ae/'mean_attention_maps';means.mkdir(parents=True)
        # All raw maps are from job47071290, whose log confirms three completed forwards.
        sample_dirs=sorted((HERE/'ae/attention_maps').glob('sample_*'))
        assert len(sample_dirs)==3
        log=(HERE/'run_repair02/ae_collect.log').read_text()
        ids=['kv_retrieval-'+str(i) for i in range(3)]
        for i,name in enumerate(ids):
            assert 'Collected full attention maps for calibration %d/3: %s'%(i+1,name) in log
            assert sample_dirs[i].name.endswith(name)
        for layer in range(32):
            if time.monotonic()>deadline:raise TimeoutError('mean aggregation budget exhausted')
            mean_heads([d/('layer_%02d.npy'%layer) for d in sample_dirs],means/('layer_%02d.npy'%layer))
            print('recovered mean layer',layer,flush=True)
        fingerprint=hashlib.sha256()
        for i,name in enumerate(ids):
            tokens=np.frombuffer((HERE/('inputs/%d.ae_ids.i64le'%i)).read_bytes(),dtype='<i8')
            fingerprint.update(name.encode()+b'\0');fingerprint.update(','.join(str(int(t)) for t in tokens).encode()+b'\n')
        manifest={'schema_version':1,'source':'benchmark_specific_prompt_jsonl','calibration_format':'prompt_jsonl',
            'sample_ids':ids,'num_samples':3,'map_size':1936,'num_layers':32,'num_heads':32,'dtype':'float16',
            'input_truncation':'tokenizer_right_truncation_to_1936','input_fingerprint_sha256':fingerprint.hexdigest(),
            'evaluation_sample_usage':False,'recovered_from_job':'47071290','recovery':'original sum order, headwise; no repeated model forward'}
        save(ae/'attention_map_manifest.json',manifest)
        durable=RUN/'ae';shutil.copytree(ae,durable)
        common=[sys.executable,'-m','experiments.run_shareprefill_autoencoder_clustering']
        original=json.loads((HERE/'run_repair02/ae_config.json').read_text())['args']
        original[original.index('--artifact_dir')+1]=str(ae)
        available=deadline-time.monotonic()-1800
        estimate=json.loads((HERE/'run_repair02/ae_budget.json').read_text())['estimated_earliest_patience_seconds']*1.25
        if available<estimate:raise TimeoutError('AE estimate plus replay reserve exceeds remaining budget')
        try:
            invoke('ae_train',common+['train']+original,available)
        finally:
            # Preserve best checkpoint on any exit, but never cluster an incomplete training run.
            for p in ae.glob('*'):
                if p.is_file():shutil.copy2(p,durable/p.name)
        assert (ae/'autoencoder_training.json').is_file()
        invoke('ae_cluster',common+['cluster']+original,180)
        for p in ae.glob('*'):
            if p.is_file():shutil.copy2(p,durable/p.name)
        groups=durable/'shareprefill_ae_k3_head_groups.json';g=json.loads(groups.read_text())
        assert len(g['layers'])==32
        for layer in g['layers'].values():
            assert len(layer)==3 and sorted(h for c in layer for h in c['members'])==list(range(32))
            assert all(c['representative'] in c['members'] for c in layer)
        sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        save(RUN/'groups_provenance.json',{'origin':'PSC new AE K=3','calibration_indices':[0,1,2],
            'test_index_excluded':3,'groups_sha256':sha(groups),'checkpoint_sha256':sha(durable/'autoencoder_best.pt'),
            'grouping_independent_of_validation':False,'gqa_mapping':[h//4 for h in range(32)]})
        model_source=Path(json.loads((HERE/'model_provenance.json').read_text())['path'])
        local_model=scratch/'model'
        shutil.copytree(model_source,local_model,ignore=shutil.ignore_patterns('.cache'))
        provenance=json.loads((HERE/'model_provenance.json').read_text())
        copied={}
        for name,expected in provenance['files'].items():
            digest=hashlib.sha256()
            with (local_model/name).open('rb') as stream:
                for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
            assert digest.hexdigest()==expected['sha256'], 'staged model provenance mismatch: '+name
            copied[name]=digest.hexdigest()
        save(RUN/'staged_model_provenance.json',{'source':str(model_source),'staged_path':str(local_model),'files':copied})
        os.environ['PSC_MODEL_DIR']=str(local_model)
        for method in METHODS:
            invoke('compare_'+method,[sys.executable,str(HERE/'compare.py'),'--method',method,'--groups',str(groups)])
        rows=[]
        for method in METHODS:
            items=[json.loads(x) for x in (RUN/method/'layers.jsonl').read_text().splitlines()]
            assert {(x['layer'],x['point']) for x in items}=={(i,p) for i in range(32) for p in ('attention_post_o_proj','block_output')}
            assert len(items)==64;rows.extend(items)
        (RUN/'all_layers.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
        invoke('summarize',[sys.executable,str(HERE/'summarize.py')])
        save(RUN/'state.json',{'status':'complete','rows':320,'elapsed_seconds':time.monotonic()-start,'job_id':os.environ['SLURM_JOB_ID']})
    except BaseException:
        save(RUN/'state.json',{'status':'incomplete','traceback':traceback.format_exc(),'elapsed_seconds':time.monotonic()-start,'job_id':os.environ['SLURM_JOB_ID']})
        raise
if __name__=='__main__':main()
