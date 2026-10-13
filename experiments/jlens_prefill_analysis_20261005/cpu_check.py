"""Final preparation audit, CPU tests and proof of approval gates."""
from collections import Counter
import os
from pathlib import Path
import subprocess
import sys

from common import HERE,config,read,save,sha,verify_preparation,legacy


def check():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='': raise ValueError('Hide GPUs for CPU audit')
    if (HERE/'approval.json').exists(): raise PermissionError('CPU audit expects pending approval')
    manifest=verify_preparation()
    c=config()
    if len(manifest['evaluation'])!=200 or manifest['fit_prompts']!=1000:
        raise ValueError('Wrong input count')
    if len(manifest.get('model_files',{}))<1: raise ValueError('Missing model provenance')
    old=legacy()
    prior,historical_manifest=old.verify_inputs()
    heads=old.load_heads(prior,historical_manifest)
    if set(heads)!= {'coverage','random'}: raise ValueError('Representative heads missing')
    for shard in range(10):
        subset=[r for r in manifest['evaluation'] if r['analysis_shard']==shard]
        if Counter(r['task'] for r in subset)!=Counter({t:5 for t in manifest['tasks']}):
            raise ValueError('Task-unbalanced replay worker')
    test=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(HERE),'-p','test_*.py','-v'],
                        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    (HERE/'analysis-cpu-tests.log').write_text(test.stdout)
    if test.returncode: raise RuntimeError(test.stdout)
    print(test.stdout,flush=True)
    blocked=[]
    for script,args in [('fit_lens.py',['fit','--shard','0']),
                        ('replay.py',['--method','dense_full','--shard','0']),('submit.py',[])]:
        result=subprocess.run([sys.executable,str(HERE/script),*args],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        if result.returncode==0 or 'Awaiting user approval' not in result.stdout:
            raise ValueError('Approval gate failed: '+script)
        blocked.append(script)
    if (HERE/'outputs').exists() or (HERE/'submitted.json').exists():
        raise ValueError('Unexpected experiment outputs before approval')
    original_log=HERE/'original-jlens-cpu-tests.log'
    if '32 passed' not in original_log.read_text(): raise ValueError('Original tests incomplete')
    lengths=[r['input_tokens'] for r in manifest['evaluation']]
    positions=[len(r['analysis_positions']) for r in manifest['evaluation']]
    save(HERE/'cpu-validation.json',dict(status='PASS',new_cpu_tests=9,original_jlens_cpu_tests=32,
        tests_log_sha256=sha(HERE/'analysis-cpu-tests.log'),original_tests_log_sha256=sha(original_log),
        approval_gate_proofs=blocked,model_checkpoint_loaded=False,gpu_tests_run=False,
        experiment_submitted=False,fit_prompts=1000,fit_tokens_per_prompt=128,
        evaluation_inputs=200,methods=7,planned_prefill_records=1400,
        input_length_min=min(lengths),input_length_max=max(lengths),
        sampled_position_min=min(positions),sampled_position_max=max(positions),
        evidence_annotations=Counter(r['evidence_status'] for r in manifest['evaluation']),
        representative_heads_loaded=True,
        fixed_bos_token=128000,source_lock_sha256=sha(HERE/'source-lock.json'),
        input_manifest_sha256=sha(HERE/'input-manifest.json'),config_sha256=sha(HERE/'config.json')))
    print('CPU PREPARATION PASS; experiments remain unapproved',flush=True)


if __name__=='__main__': check()
