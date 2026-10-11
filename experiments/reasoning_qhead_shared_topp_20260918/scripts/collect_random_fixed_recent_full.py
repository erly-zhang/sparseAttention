"""Fail closed on incomplete workers, then validate/grade and enrich complete outputs."""
import datetime,hashlib,json,os,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve()
def status(phase,**kw):
 row=dict(phase=phase,timestamp=datetime.datetime.now().astimezone().isoformat(),**kw)
 (root/'collection_status.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row),flush=True)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
try:
 status('checking_workers')
 missing=[]
 for i in range(8):
  d=root/f'random/shard_{i}';p=d/'exit.json'
  if not p.exists() or json.loads(p.read_text()).get('exit_code')!=0:missing.append(str(d))
 if missing:
  status('incomplete_or_failed',missing_successful_workers=missing);raise SystemExit(1)
 for rel,h in json.loads((root/'manifest.json').read_text()).items():assert sha(root/rel)==h,rel
 for i in range(8):assert json.loads((root/f'random/shard_{i}/preflight.json').read_text())['status']=='passed'
 inputs={str(p.relative_to(root)):sha(p) for p in sorted((root/'random').glob('shard_*/**/*')) if p.is_file() and p.name in ('cache_metrics.jsonl','completions.jsonl','exit.json','hardware.json','preflight.json')}
 proof=root/'collector_input_hashes.json'
 if proof.exists():assert json.loads(proof.read_text())==inputs,'Inputs changed since an earlier collection attempt'
 else:proof.write_text(json.dumps(inputs,indent=2)+'\n')
 status('validating_and_grading')
 if not (root/'random/summary.json').exists():
  subprocess.run([sys.executable,str(root/'source/scripts/check_random_fixed_recent_smoke.py'),str(root),'--phase','formal'],check=True)
 else:
  s=json.loads((root/'random/summary.json').read_text());assert s['status']=='completed' and s['questions']==500 and s['input_alignment']=='passed/full'
 from summarize_fixed_recent_metrics import summarize
 s=summarize(root,root/'random','formal')
 # Keep scheduler accounting alongside inference metrics; collector uses no GPU.
 sub=json.loads((root/'submission.json').read_text())
 cmd=['sacct','-j',str(sub['array_job_id']),'--parsable2','--format=JobIDRaw,JobName,State,ExitCode,Submit,Start,End,Elapsed,ElapsedRaw,AllocCPUS,ReqMem,MaxRSS,NodeList,AllocTRES']
 r=subprocess.run(cmd,text=True,capture_output=True)
 (root/'scheduler_accounting.txt').write_text(r.stdout);(root/'scheduler_accounting_error.txt').write_text(r.stderr)
 status('completed',input_alignment=s['input_alignment'],questions=500,rollouts=1000,accuracy=s['accuracy'],scheduler_accounting_exit_code=r.returncode,report=str(root/'random/REPORT.md'))
except Exception as exc:
 status('failed',error=f'{type(exc).__name__}: {exc}');raise
