"""Single allocation, strict wall-clock cap; no retry, no silent scope reduction."""
import hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
METHODS=('dense','routing_c','shared6144_member2q4096','flexprefill','minference')
START=time.monotonic();DEADLINE=START+3500

def save(name,obj): (HERE/'run'/name).write_text(json.dumps(obj,indent=2)+'\n')
def invoke(name,command,limit=None):
    remaining=DEADLINE-time.monotonic()
    if limit is not None:remaining=min(remaining,limit)
    if remaining<=0:raise TimeoutError('GPU budget exhausted')
    start=time.monotonic()
    with (HERE/'run'/f'{name}.log').open('x') as log:
        subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=remaining)
    with (HERE/'run/stages.jsonl').open('a') as f:f.write(json.dumps({'stage':name,'seconds':time.monotonic()-start})+'\n')

def main():
    (HERE/'run').mkdir(exist_ok=False)
    save('state.json',{'status':'running','job_id':os.environ.get('SLURM_JOB_ID'),'max_wall_seconds':3600})
    try:
        import torch
        assert torch.cuda.device_count()==1 and 'H100' in torch.cuda.get_device_name(0)
        assert torch.cuda.is_bf16_supported()
        save('gpu.json',{'name':torch.cuda.get_device_name(0),'torch':torch.__version__,
                        'host':os.uname().nodename,'job_id':os.environ['SLURM_JOB_ID']})
        for method in METHODS:
            invoke('correctness_'+method,[sys.executable,str(HERE/'correctness.py'),'--method',method],180)
        common=[sys.executable,'-m','experiments.run_shareprefill_autoencoder_clustering']
        args=['--model_name_or_path','/ocean/projects/cis260309p/jzhang114/prefill-routing-deps/models/Llama-3.1-8B-Instruct',
              '--data_path',str(HERE/'inputs/calibration.jsonl'),'--calibration_format','prompt_jsonl',
              '--artifact_dir',str(HERE/'ae'),'--head_selection_num_samples','3','--map_size','1936','--latent_dim','64',
              '--epochs','1000','--patience','30','--min_delta','1e-7','--learning_rate','0.001','--batch_size','1',
              '--groups','3','--seed','42','--device','cuda']
        save('ae_config.json',{'command_base':common,'args':args,'test_index_used':False,
                             'origin':'new PSC calibration 0,1,2','grouping_independent_of_validation':False})
        invoke('ae_collect',common+['collect']+args,600)
        invoke('ae_cost',[sys.executable,str(HERE/'ae_budget.py')],180)
        budget=json.loads((HERE/'run/ae_budget.json').read_text())
        available=DEADLINE-time.monotonic()-900
        # 900 seconds reserve is a planning allowance, not a runtime guarantee.
        # If even earliest permitted stopping is implausible, preserve remaining budget.
        if budget['estimated_earliest_patience_seconds']*1.25>available:
            save('state.json',{'status':'blocked_budget','stage':'original_AE_training',
                'elapsed_seconds':time.monotonic()-START,'training_budget_seconds':available,
                'estimated_earliest_seconds_with_margin':budget['estimated_earliest_patience_seconds']*1.25,
                'groups_created':False,'comparison_completed':False,'scope_changed':False,
                'job_id':os.environ['SLURM_JOB_ID']})
            return 42
        invoke('ae_train',common+['train']+args,available)
        invoke('ae_cluster',common+['cluster']+args,180)
        groups=HERE/'ae/shareprefill_ae_k3_head_groups.json'
        config=json.loads(groups.read_text())
        assert len(config['layers'])==32
        for layer in config['layers'].values():
            assert len(layer)==3
            assert sorted(h for g in layer for h in g['members'])==list(range(32))
            assert all(g['representative'] in g['members'] for g in layer)
        save('groups_provenance.json',{'origin':'PSC regenerated AE K=3','groups_sha256':hashlib.sha256(groups.read_bytes()).hexdigest(),
             'checkpoint_sha256':hashlib.sha256((HERE/'ae/autoencoder_best.pt').read_bytes()).hexdigest(),
             'calibration_indices':[0,1,2],'test_index_excluded':3,'seed':42,'grouping_independent_of_validation':False,
             'gqa_mapping':[h//4 for h in range(32)],'member_coverage_validated':True})
        for method in METHODS:
            invoke('compare_'+method,[sys.executable,str(HERE/'compare.py'),'--method',method,'--groups',str(groups)])
        rows=[]
        for method in METHODS:
            data=[json.loads(x) for x in (HERE/'run'/method/'layers.jsonl').read_text().splitlines()]
            assert len(data)==64 and len({(x['layer'],x['point']) for x in data})==64
            rows.extend(data)
        (HERE/'run/all_layers.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
        save('state.json',{'status':'complete','elapsed_seconds':time.monotonic()-START,'rows':len(rows),
                          'job_id':os.environ['SLURM_JOB_ID'],'scope_changed':False})
        return 0
    except BaseException as exc:
        save('state.json',{'status':'incomplete','error':str(exc),'traceback':traceback.format_exc(),
            'elapsed_seconds':time.monotonic()-START,'job_id':os.environ.get('SLURM_JOB_ID'),
            'scope_changed':False,'automatic_resubmit':False})
        raise
if __name__=='__main__':sys.exit(main())
