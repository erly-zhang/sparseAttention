import hashlib,json,os,shutil,subprocess,sys,tempfile,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parents[1];BASE=HERE.parent/'psc_kv_layer_outputs_20260925';RUN=HERE/'run_l40s_mlp'
START=time.monotonic();DEADLINE=START+1700

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(n,o):(RUN/n).write_text(json.dumps(o,indent=2)+'\n')
def main():
    RUN.mkdir(exist_ok=False)
    save('state.json',{'status':'running','job_id':os.environ['SLURM_JOB_ID']})
    try:
        import torch
        assert torch.cuda.device_count()==1 and 'L40S' in torch.cuda.get_device_name()
        save('environment.json',{'host':os.uname().nodename,'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'job_id':os.environ['SLURM_JOB_ID']})
        local=Path(os.environ.get('LOCAL') or os.environ.get('TMPDIR') or '/tmp');assert shutil.disk_usage(local).free>18*1024**3
        dest=Path('/local/psc-hybrid-smoke-47082259-sbm8_zpy')
        assert dest.is_dir(), 'reuse verified local weights from preceding attempt'
        prov=json.loads((BASE/'model_provenance.json').read_text());copied={}
        for name,meta in prov['files'].items():
            if time.monotonic()>DEADLINE:raise TimeoutError('staging deadline exceeded')
            copied[name]=sha(dest/name);assert copied[name]==meta['sha256']
        save('model_provenance.json',{'revision':prov['revision'],'source':prov['path'],'staged':str(dest),'files':copied})
        os.environ['PSC_MODEL_DIR']=str(dest)
        with (RUN/'smoke.log').open('x') as log:
            subprocess.run([sys.executable,str(HERE/'vscode_l40s/smoke.py')],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=max(1,DEADLINE-time.monotonic()))
        save('state.json',{'status':'complete','job_id':os.environ['SLURM_JOB_ID'],'elapsed_seconds':time.monotonic()-START,'no_KL':True})
    except BaseException:
        save('state.json',{'status':'incomplete','job_id':os.environ['SLURM_JOB_ID'],'error':traceback.format_exc(),'elapsed_seconds':time.monotonic()-START});raise
if __name__=='__main__':main()
