"""Run after the USER authenticates and accepts the official gated model terms."""
import json
import os
from pathlib import Path
os.environ.setdefault('HF_HOME','/ocean/projects/cis260309p/jzhang114/prefill-routing-deps/hf-cache')
from huggingface_hub import HfApi, snapshot_download
from audit import model_audit

root = Path('/ocean/projects/cis260309p/jzhang114/prefill-routing-deps/models/Llama-3.1-8B-Instruct')
repo = 'meta-llama/Llama-3.1-8B-Instruct'
info = HfApi().model_info(repo)
path = snapshot_download(repo, revision=info.sha, local_dir=root,
                         allow_patterns=['*.json','*.safetensors','*.model','*.jinja','LICENSE','USE_POLICY.md'])
record = model_audit(path)
record.update(repo=repo,revision=info.sha)
(Path(__file__).parent/'records/pretrained_model_provenance.json').write_text(json.dumps(record,indent=2)+'\n')
print('MODEL_READY',path,'REVISION',info.sha)
