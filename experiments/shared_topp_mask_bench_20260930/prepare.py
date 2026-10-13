"""Clone the audited benchmark inputs byte-for-byte; no new download or sample."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
from protocol import RUN,PRIOR,ORIGIN,ORIGIN_MANIFEST,ORIGIN_AUDIT,METHODS
from protocol import sha,save,validate_reuse


def quality():
    path=PRIOR/'quality.py'
    assert sha(path)=='7c66db90d78a43ce19c876669e849414cf685389164c9819cb062cb91a1ffb61'
    spec=importlib.util.spec_from_file_location('bench_quality',path)
    q=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(q)
    q.verify_sources()
    prior=json.loads((PRIOR/'inputs/manifest.json').read_text())
    assert sha(PRIOR/'inputs/manifest.json')=='dc76a7a783ef48c5be21338b3c1c3cb0c8a816470bf7661a232da75057bd35e3'
    for name,value in prior['tokenizer_files'].items():
        assert sha(q.MODEL/name)==value
    return q,prior


def clone_inputs(source,destination):
    source,destination=Path(source),Path(destination)
    assert sha(source/'inputs/manifest.json')==ORIGIN_MANIFEST
    assert sha(source/'independent-audit.json')==ORIGIN_AUDIT
    origin=json.loads((source/'inputs/manifest.json').read_text())
    stage=Path(__file__).resolve().parent
    destination.mkdir()
    (destination/'inputs').mkdir()
    (destination/'source-preparation').mkdir()
    for folder in ('data','official'):
        shutil.copytree(source/folder,destination/folder)
    for item in origin['records']:
        for suffix,field in {'ids':'input_ids_sha256','pos':'position_ids_sha256',
            'mask':'attention_mask_sha256','prompt':'rendered_prompt_sha256','raw':'raw_row_sha256'}.items():
            src=source/'inputs'/(item['key']+'.'+suffix)
            assert sha(src)==item[field]
            dst=destination/'inputs'/src.name
            shutil.copyfile(src,dst)
            assert sha(dst)==item[field]
    shutil.copyfile(source/'inputs/manifest.json',destination/'origin-manifest.json')
    shutil.copyfile(source/'independent-audit.json',destination/'origin-audit.json')
    for name in ('prepare.py','protocol.py'):
        shutil.copyfile(stage/name,destination/'source-preparation'/name)
    manifest=dict(origin,methods=list(METHODS),
        preparation_sources={name:sha(stage/name) for name in ('prepare.py','protocol.py')},
        reused_input_manifest_sha256=ORIGIN_MANIFEST,origin_audit_sha256=ORIGIN_AUDIT)
    validate_reuse(manifest,origin)
    save(destination/'inputs/manifest.json',manifest)
    return manifest


if __name__=='__main__':
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert not RUN.exists(), 'Exclusive preparation: never overwrite or resume implicitly'
    quality()
    try:
        clone_inputs(ORIGIN,RUN)
        print('EXACT INPUT CLONE200',sha(RUN/'inputs/manifest.json'),flush=True)
    except Exception as exc:
        if RUN.exists() and not (RUN/'prepare-failure.json').exists():
            save(RUN/'prepare-failure.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
