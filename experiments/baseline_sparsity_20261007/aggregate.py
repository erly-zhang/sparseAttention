"""Independently reconcile every head count before changing the report."""
from collections import defaultdict
import fcntl
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ORIGINAL=HERE.parent/'representative_patterns_fast_retry03_20261001'
TASKS=['kv_retrieval','longbook_choice_eng','longbook_qa_eng','code_debug']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)

def aggregate_if_ready():
    (HERE/'outputs').mkdir(exist_ok=True)
    with (HERE/'outputs/aggregate.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return False
        plan=json.loads((HERE/'work-plan.json').read_text())
        if any(not (HERE/'outputs'/('worker-'+str(p['worker']))/'complete.json').exists() for p in plan):return False
        auditfile=HERE/'outputs/independent-audit.json'
        if auditfile.exists() and (HERE/'report-update.json').exists():return True
        groups=defaultdict(list);seen=set();hashes={};heads=0;bitwise=0;first=0;maxerr=0.
        for job in plan:
            folder=HERE/'outputs'/('worker-'+str(job['worker']))
            complete=json.loads((folder/'complete.json').read_text())
            assert complete['status']=='PASS' and complete['samples']==len(job['keys'])
            assert complete['source_lock_sha256']==sha(HERE/'source-lock.json')
            for filename,digest in complete['files'].items():assert sha(folder/filename)==digest
            for key in job['keys']:
                p=folder/(key+'.json');r=json.loads(p.read_text());hashes[str(p)]=sha(p)
                assert (r['method'],key) not in seen;seen.add((r['method'],key))
                assert r['method']==job['method'] and r['task'] in TASKS
                assert r['original_config_sha256']==sha(ORIGINAL/'config.json')
                assert r['input_manifest_sha256']==sha(ORIGINAL/'inputs/manifest.json')
                assert sha(r['original_record_path'])==r['original_record_sha256']
                assert sha(folder/r['logits_file'])==r['logits_sha256']
                assert set(r['layers'])==set(map(str,range(32)))
                n=r['input_tokens'];denominator=n*(n+1)//2;layer_means=[]
                for layer in r['layers'].values():
                    assert layer['causal_pairs_per_head']==denominator and [h['head'] for h in layer['heads']]==list(range(32))
                    assert all(type(h['selected_pairs']) is int and 0<h['selected_pairs']<=denominator for h in layer['heads'])
                    mean=sum(h['selected_pairs'] for h in layer['heads'])/(denominator*32)
                    assert abs(mean-layer['mask_density'])<1e-14
                    layer_means.append(mean);heads+=32
                density=statistics.mean(layer_means);assert abs(density-r['mask_density'])<1e-14
                groups[(r['method'],r['task'])].append(density)
                bitwise+=r['historical_logits_bitwise_equal'];first+=r['historical_first_token_equal'];maxerr=max(maxerr,r['historical_logits_max_abs_error'])
        assert len(seen)==400 and heads==409600
        rows=[]
        for method in ['flexprefill','minference']:
            for task in TASKS:
                values=groups[(method,task)];assert len(values)==50
                density=statistics.mean(values)
                rows.append(dict(method=method,task=task,samples=50,layers=32,heads=32,
                    mask_density_percent=density*100,mask_sparsity_percent=(1-density)*100,
                    sample_density_min=min(values)*100,sample_density_max=max(values)*100))
        for method in ['flexprefill','minference']:
            smoke=json.loads((HERE/'outputs'/('smoke-'+method)/'complete.json').read_text());assert smoke['status']=='PASS'
            records=[json.loads(p.read_text()) for p in (HERE/'outputs'/('smoke-'+method)).glob('*.json') if p.name not in ['complete.json','start.json','progress.json','synthetic-parity.json','failure.json']]
            assert len(records)==1 and records[0]['observer_full_model_bitwise_equal'] is True
        audit=dict(status='PASS',samples=400,head_counts_verified=heads,rows=rows,source_record_sha256=hashes,
            source_lock_sha256=sha(HERE/'source-lock.json'),definition=json.loads((HERE/'config.json').read_text())['definition'],
            scope='same-configuration prefill replay on the original 200 full inputs for each baseline; historical masks not saved',
            observer_full_model_bitwise_equal={'flexprefill':True,'minference':True},
            historical_logits_bitwise_equal_samples=bitwise,historical_first_token_equal_samples=first,historical_logits_max_abs_error=maxerr)
        save(auditfile,audit)
        subprocess.run([sys.executable,str(HERE/'update_report.py')],check=True)
        return True

if __name__=='__main__':print('READY',aggregate_if_ready())
