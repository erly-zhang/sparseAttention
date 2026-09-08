"""Read-only compact snapshot for this experiment only."""
import json
from pathlib import Path
import subprocess

ROOT=Path('/local/results/rep_residual_20260907')
print(subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv'],text=True))
print(subprocess.check_output(['df','-h','/home/ubuntu/work','/local'],text=True))
print('PROCESSES')
for line in subprocess.check_output(['ps','-eo','pid,ppid,etime,args'],text=True).splitlines():
    if 'rep_residual_20260907' in line and 'inspect.py' not in line:print(line)
print('STATUS')
path=ROOT/'scheduler_status.jsonl'
if path.exists():
    for line in path.read_text().splitlines()[-12:]:print(line)
print('PROGRESS')
for log in sorted((ROOT/'attempts').glob('*/*/attempt*/run.log')):
    lines=log.read_text(errors='replace').splitlines()
    layers=[s for s in lines if s.startswith('CALIBRATION layer=')]
    metrics=list(log.parent.glob('*/kv_retrieval/online_metrics.jsonl'))
    count=sum(sum(1 for s in p.read_text().splitlines() if s.strip()) for p in metrics)
    errors=[s for s in lines if any(w in s for w in ('Traceback','OutOfMemoryError','RuntimeError','Error:'))]
    print(json.dumps({'attempt':str(log.parent),'calibration_layers':len(layers),'metrics':count,
                      'summary':bool(list(log.parent.glob('*/kv_retrieval/summary.json'))),
                      'last_layer':layers[-1] if layers else None,'errors':errors[-3:]}))
if (ROOT/'policies.json').exists():
    p=json.loads((ROOT/'policies.json').read_text())
    print('POLICIES',json.dumps({k:v['enabled'] for k,v in p['policies'].items()}))
