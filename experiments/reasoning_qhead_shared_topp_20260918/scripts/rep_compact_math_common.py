import hashlib,json
from pathlib import Path

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  while block:=f.read(8*1024*1024):h.update(block)
 return h.hexdigest()

def rows(path):
 with Path(path).open() as f:return [json.loads(line) for line in f if line.strip()]

def verify(root,weights=False,tests=False):
 root=Path(root)
 for rel,h in json.loads((root/'manifest.json').read_text()).items():assert sha(root/rel)==h,rel
 cfg=json.loads((root/'config.json').read_text())
 assert cfg['method']=='qhead_shared_representative_compact_recent_gpu' and cfg['compaction_interval']==64
 if weights:
  for rel,h in json.loads((root/'model_assets_sha256.json').read_text()).items():assert sha(Path(cfg['model'])/rel)==h,rel
 if tests:
  for rel,h in json.loads((root/'test_assets_sha256.json').read_text()).items():assert sha(root/'data/livecodebench/livecodebench_tests'/rel)==h,rel
 return cfg
