"""Fetch pinned public benchmark assets; GPQA is intentionally separate/access-gated."""
import concurrent.futures,hashlib,json,urllib.request,time
from pathlib import Path
E=Path(__file__).resolve().parents[1];root=E/'datasets/compact_multibench_20260926'
V='bfa2692112c45da13e637b3eca03ee6516f9e0bb';H='0fe84c3912ea0c4d4a78037083943e8f0c4dd505'
def fetch(url,path):
 path.parent.mkdir(parents=True,exist_ok=True)
 if not path.exists():
  temp=path.with_suffix(path.suffix+'.part')
  for attempt in range(3):
   try:
    with urllib.request.urlopen(url,timeout=60) as res,temp.open('wb') as out:
     while block:=res.read(1024*1024):out.write(block)
    temp.replace(path);break
   except Exception:
    if attempt==2:raise
    time.sleep(2)
 h=hashlib.sha256()
 with path.open('rb') as f:
  while b:=f.read(1024*1024):h.update(b)
 rec=dict(url=url,path=str(path.relative_to(root)),bytes=path.stat().st_size,sha256=h.hexdigest())
 print(json.dumps(rec),flush=True);return rec
jobs=[]
for task in ['aime25','aime26','hmmt','livecodebench']:
 jobs.append((f'https://raw.githubusercontent.com/terarachang/VaSE/{V}/eval/reasoning_tasks/data/{task}/test.jsonl',root/task/'test.jsonl'))
for f in ['livecodebench/download_tests.py','hmmt/convert_hmmt25.py']:
 jobs.append((f'https://raw.githubusercontent.com/terarachang/VaSE/{V}/eval/reasoning_tasks/data/{f}',root/'raw'/f))
for i in range(1,7):
 name='test.jsonl' if i==1 else f'test{i}.jsonl'
 jobs.append((f'https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/{H}/{name}',root/'raw/lcb'/name))
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:records=list(pool.map(lambda x:fetch(*x),jobs))
(root/'download_manifest.json').write_text(json.dumps(dict(vase_commit=V,lcb_revision=H,files=records),indent=2)+'\n')
