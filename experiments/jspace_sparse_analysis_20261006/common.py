"""Offline post-processing of the completed, frozen prefill observations."""
from pathlib import Path
import hashlib, json, os

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'jlens_prefill_analysis_20261005'
METHODS = ['dense_full','coverage_token_vertical','coverage_token_vertical_slash',
           'random_token_vertical','random_token_vertical_slash','flexprefill','minference']
TASKS = ['kv_retrieval','longbook_choice_eng','longbook_qa_eng','code_debug']
LABELS = ['Dense','Calibrated V','Calibrated V+S','Random V','Random V+S','FlexPrefill','MInference']
TASK_LABELS = ['KV Retrieval','En.MC','En.QA','Code.Debug']

def read(p): return json.loads(Path(p).read_text())
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+'.tmp.'+str(os.getpid()))
    with t.open('w') as f:
        json.dump(v,f,ensure_ascii=False,allow_nan=False);f.flush();os.fsync(f.fileno())
    t.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def verify():
    lock=read(HERE/'lock.json')
    for p,h in lock['files'].items():
        if sha(p)!=h:raise ValueError('Changed bound input/code: '+p)
    if read(SOURCE/'outputs/independent-audit.json')['status']!='PASS':raise ValueError('Source audit failed')
    return read(HERE/'manifest.json')
