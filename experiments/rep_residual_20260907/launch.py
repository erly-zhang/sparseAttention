"""Start only this experiment scheduler; never stop existing processes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

HERE=Path(__file__).resolve().parent
ROOT=Path('/local/results/rep_residual_20260907')
PY='/home/ubuntu/miniconda3/envs/attentionmap/bin/python'
parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['calibration','fixed','both'],default='calibration')
args=parser.parse_args()
active=subprocess.check_output(['ps','-eo','pid,args'],text=True)
needle=str(HERE/'schedule.py')
if any(needle in line for line in active.splitlines()):raise RuntimeError('Existing residual scheduler found')
logpath=ROOT/f'scheduler_{args.stage}_{int(time.time())}.log'
with logpath.open('w') as log:
    child=subprocess.Popen([PY,str(HERE/'schedule.py'),'--stage',args.stage],cwd='/home/ubuntu/work',
        env={**os.environ,'PYTHONPATH':'/home/ubuntu/work'},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'pid':child.pid,'log':str(logpath),'stage':args.stage}))
