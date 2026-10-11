"""Separate two-method, one-input shape diagnostic; no formal jobs touched."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from experiments.learned_mask_eval_20260914 import launch as old
from experiments.learned_mask_eval_20260914.gpu_preflight import allocation

ROOT = old.ROOT
OUT = ROOT/'experiments/outputs/learned_mask_shape_20260916'


def main():
    access = allocation()
    OUT.mkdir(exist_ok=False)
    old.save(OUT/'allocation.json',access)
    summaries = {}
    for variant in ('per_cluster','full_mask'):
        formal = ROOT/'experiments/outputs/learned_mask_eval_20260914/formal'/variant
        command = list(old.read(formal/'command.json')['command'])
        command[command.index('-m')+1] = 'experiments.learned_mask_shape_20260916'
        destination = OUT/variant
        destination.mkdir()
        command[command.index('--output_dir')+1] = str(destination/'generation')
        command[command.index('--limit')+1] = '1'
        env = dict(os.environ, LEARNED_MASK_SMOKE='0',
                   LEARNED_MASK_TRAINING_AUDIT=str(formal/'training_audit.json'),
                   MASK_SHAPE_OUTPUT=str(destination/'masks'))
        old.save(destination/'command.json',dict(command=command,diagnostic_only=True,
            calibration=False,selected=old.read(formal/'command.json')['selected']))
        with (destination/'runner.log').open('x') as log:
            subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        files=sorted((destination/'masks').glob('layer*.json'))
        assert len(files)==32
        records=[r for f in files for r in old.read(f)]
        metrics=old.read_rows(destination/'generation'/f'nnonly_{variant}_compact10240/kv_retrieval/online_metrics.jsonl')
        assert len(metrics)==1
        ref=old.read_rows(old.REFERENCE)
        key=lambda x:(x['input_tokens'],x['input_ids_sha256'])
        assert sum(key(x)==key(metrics[0]) for x in ref)==1
        members=[r for r in records if not r['is_representative']]
        result=dict(input_tokens=key(metrics[0])[0],input_ids_sha256=key(metrics[0])[1],
            layers=32,heads=32,tiles_per_layer=8,sampled_member_rows=len(members),
            comparator='original Shared10240 selection on each NN runs own identical Q/K',
            diagnostic_timing_not_formal=True)
        for name in ['shared_overlap','new_keys','member_overlap']:
            values=[r[name] for r in members]
            result[name]=dict(mean=float(np.mean(values)),median=float(np.median(values)),
                              minimum=float(min(values)),maximum=float(max(values)))
        for method in ['final','shared']:
            result[method]={name:float(np.mean([r[method][name] for r in members]))
                            for name in ['runs','mean_run_length','mean_position','first_quarter','last_quarter']}
        result['per_layer']=[dict(layer=l,overlap=float(np.mean([r['shared_overlap'] for r in members if r['layer']==l])),
            member_overlap=float(np.mean([r['member_overlap'] for r in members if r['layer']==l]))) for l in range(32)]
        old.save(destination/'shape_summary.json',result)
        summaries[variant]=result
        print('SHAPE_SUMMARY',variant,json.dumps(result),flush=True)
    assert summaries['per_cluster']['input_ids_sha256']==summaries['full_mask']['input_ids_sha256']
    old.save(OUT/'summary.json',summaries)
    print('SHAPE_COMPLETE',flush=True)


if __name__=='__main__':
    code,error=1,None
    try:
        main()
        code=0
    except BaseException as exc:
        error=repr(exc)
        raise
    finally:
        if OUT.exists():
            old.save(OUT/'exit.json',dict(returncode=code,error=error,time=time.time()))
