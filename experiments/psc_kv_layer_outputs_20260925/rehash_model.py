from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
old=json.loads((root/'model_provenance_previous_verified.json').read_text());model=Path(old['path'])
files={}
for name,expected in old['files'].items():
    p=model/name;h=hashlib.sha256()
    with p.open('rb') as f:
        for data in iter(lambda:f.read(8*1024*1024),b''):h.update(data)
    actual={'bytes':p.stat().st_size,'sha256':h.hexdigest()}
    if actual!=expected:raise ValueError('checkpoint identity changed: '+name)
    files[name]=actual
old.update(files=files,verified_again_for='PSC new controlled experiment')
(root/'model_provenance.json').write_text(json.dumps(old,indent=2)+'\n')
print('All model/tokenizer files rehashed and match pinned provenance')
