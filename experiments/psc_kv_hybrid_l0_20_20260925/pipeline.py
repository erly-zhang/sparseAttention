"""Single authorized hybrid variant; existing dense/reference outputs reused read-only."""
import hashlib,json,os,shutil,subprocess,sys,tempfile,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
BASE=ROOT/'experiments/psc_kv_layer_outputs_20260925';RUN=HERE/'run'
START=time.monotonic();DEADLINE=START+1700

def sha(path):
    value=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):value.update(chunk)
    return value.hexdigest()

def save(name,obj): (RUN/name).write_text(json.dumps(obj,indent=2)+'\n')

def invoke(name,args):
    remain=DEADLINE-time.monotonic()
    if remain<=0:raise TimeoutError('hybrid stage deadline exhausted')
    start=time.monotonic()
    with (RUN/(name+'.log')).open('x') as f:subprocess.run(args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=remain)
    with (RUN/'stages.jsonl').open('a') as f:f.write(json.dumps({'stage':name,'seconds':time.monotonic()-start})+'\n')

def main():
    assert os.environ.get('SLURM_JOB_ID')
    RUN.mkdir(exist_ok=False);os.environ['PSC_RUN_DIR']=str(RUN)
    try:
        import torch
        assert torch.cuda.device_count()==1 and 'H100' in torch.cuda.get_device_name()
        save('state.json',{'status':'running','job_id':os.environ['SLURM_JOB_ID']})
        save('environment.json',{'job_id':os.environ['SLURM_JOB_ID'],'host':os.uname().nodename,
            'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'prior_allocated_seconds':4039,'cumulative_cap_seconds':7200})
        baseline=BASE/'run_kl_resume'
        assert json.loads((baseline/'state.json').read_text())['status']=='complete'
        for layer in range(32):
            for point in ('attention_post_o_proj','block_output'):
                assert (baseline/'dense/tensors'/('%s_%02d.pt'%(point,layer))).is_file()
        save('reference_provenance.json',{'baseline_job':'47087068','baseline_metrics_sha256':sha(baseline/'all_layers.jsonl'),
            'input_ids_sha256':sha(BASE/'inputs/3.ids.i64le'),'reference_dir':str(baseline/'dense/tensors'),
            'model_provenance_sha256':sha(BASE/'model_provenance.json'),'exploratory_posthoc':True,'cutoff_chosen_using_test_index3':True})
        invoke('correctness',[sys.executable,str(HERE/'correctness.py')])
        local=Path(os.environ.get('LOCAL') or os.environ.get('TMPDIR') or '/tmp')
        assert shutil.disk_usage(local).free>18*1024**3
        scratch=Path(tempfile.mkdtemp(prefix='psc-hybrid-'+os.environ['SLURM_JOB_ID']+'-',dir=local))
        prov=json.loads((BASE/'model_provenance.json').read_text());source=Path(prov['path']);target=scratch/'model';target.mkdir()
        copied={}
        for name,meta in prov['files'].items():
            if time.monotonic()>DEADLINE:raise TimeoutError('model staging deadline exhausted')
            shutil.copy2(source/name,target/name)
            copied[name]=sha(target/name);assert copied[name]==meta['sha256'],name
        save('model_provenance.json',{'source':str(source),'staged':str(target),'files':copied,'revision':prov['revision']})
        os.environ['PSC_MODEL_DIR']=str(target)
        invoke('compare',[sys.executable,str(HERE/'compare.py'),'--method','routing_c_l0_20_dense_l21_31'])
        invoke('summarize',[sys.executable,str(HERE/'summarize.py')])
        save('state.json',{'status':'complete','job_id':os.environ['SLURM_JOB_ID'],'new_rows':64,'comparison_rows':192,
            'elapsed_seconds':time.monotonic()-START,'exploratory_posthoc':True})
    except BaseException:
        save('state.json',{'status':'incomplete','job_id':os.environ['SLURM_JOB_ID'],'error':traceback.format_exc(),'elapsed_seconds':time.monotonic()-START})
        raise
if __name__=='__main__':main()
