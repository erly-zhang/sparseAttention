"""Recover excluded calibration records without using formal evaluation for fitting."""
import hashlib
import json
from pathlib import Path
import shutil

import requests
import yaml

ROOT=Path('/local/results/rep_residual_20260907')
DATA=Path('/local/experiment-data/infinitebench_benchmark_specific_calibration/filtered_data')
SOURCE='https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench/resolve/main/kv_retrieval.jsonl?download=true'


def digest(row):
    return hashlib.sha256(json.dumps(row,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def configuration(dest, records):
    data=dest/'filtered_data'; config=dest/'task_configs'
    data.mkdir(parents=True,exist_ok=True); config.mkdir(exist_ok=True)
    content=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records)
    path=data/'kv_retrieval.jsonl'
    if path.exists() and path.read_text()!=content:
        raise RuntimeError(f'Refusing to overwrite different data: {path}')
    path.write_text(content)
    node=yaml.compose((DATA.parent/'task_configs/kv_retrieval.yaml').read_text())
    hits=[]
    def visit(n):
        if isinstance(n,yaml.ScalarNode) and n.value==str(DATA):
            n.value=str(data); hits.append(1)
        elif isinstance(n,yaml.MappingNode):
            for k,v in n.value: visit(k);visit(v)
        elif isinstance(n,yaml.SequenceNode):
            for x in n.value: visit(x)
    visit(node); assert len(hits)==1
    (config/'kv_retrieval.yaml').write_text(yaml.serialize(node))
    for f in (DATA.parent/'task_configs').glob('*.py'):
        shutil.copy2(f,config/f.name)
    return config


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    evaluation=[json.loads(x) for x in (DATA/'kv_retrieval.jsonl').read_text().splitlines() if x.strip()]
    assert len(evaluation)==497
    cached=ROOT/'calibration_source_first4.json'
    if cached.exists():
        records=json.loads(cached.read_text())
    else:
        records=[]
        with requests.get(SOURCE,stream=True,timeout=120) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.strip(): records.append(json.loads(line))
                if len(records)==4: break
        assert len(records)==4
        cached.write_text(json.dumps(records,ensure_ascii=False))
    # Exact original record 3 must be the current formal record 0.
    assert records[3]==evaluation[0], 'Upstream dataset no longer matches formal dataset'
    hashes={digest(r) for r in evaluation}
    assert len(hashes)==497
    assert not any(digest(r) in hashes for r in records[:3])
    for i in range(3): configuration(ROOT/'data'/f'calibration_{i}',records[i:i+1])
    configuration(ROOT/'data'/'smoke_0_3',evaluation[:3])
    for a,b in ((0,125),(125,250),(250,375),(375,497)):
        configuration(ROOT/'data'/f'formal_{a}_{b}',evaluation[a:b])
    manifest={'source':SOURCE,'source_record3_equals_formal_record0':True,
              'fit_source_indices':[0,1],'gate_source_indices':[2],
              'calibration_record_sha256':[digest(x) for x in records[:3]],
              'formal_count':497,'formal_record_sha256':[digest(x) for x in evaluation],
              'calibration_evaluation_disjoint':True}
    (ROOT/'data_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({k:v for k,v in manifest.items() if k!='formal_record_sha256'},indent=2))


if __name__=='__main__': main()
