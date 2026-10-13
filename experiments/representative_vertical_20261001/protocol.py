"""Source-bound configuration and input validation. No GPU/model work on import."""
import hashlib
import json
import os
from pathlib import Path
from experiments.shared_topp_mask_bench_20260930.protocol import (
    TASKS, MAX_NEW, official, render, repetition, sha, digest, save,
)

RUN = Path(__file__).resolve().parent
ROOT = RUN.parent.parent
ORIGIN = RUN.parent/'shared_topp_mask_bench_20260930'
PRIOR = RUN.parent/'psc_kl_quality_retry02_20260927'
ORIGIN_MANIFEST = '0bfa5d19678ff39255e7d2e122a71ed51720647d3fbc81f508f3f24b1cb0d4c5'
ORIGIN_AUDIT = 'e7faa4c658cbf5a2aae8f842534a6a2fa2eefb4bfaa152de17d09e5563492c8d'


def configuration():
    c = json.loads((RUN/'config.json').read_text())
    s = c['selection']
    if c['methods'] != ['dense_full','representative_autoblock','representative_token','representative_token_vertical']:
        raise ValueError('unsupported method matrix')
    if s['query_tile_size'] != 128 or s['token_chunk_size'] not in (16,32):
        raise ValueError('existing kernel requires Q128 and K16/32')
    if not s['logical_key_block_sizes'] or any(x not in (32,64,128) for x in s['logical_key_block_sizes']):
        raise ValueError('invalid logical key blocks')
    if s['target_token_top_p'] is not None or s['target_token_budget'] < 128:
        raise ValueError('this frozen revision uses fixed Top-k >=128 to ensure nonempty causal rows')
    if len(s['probe_weights']) != 4 or any(x<0 for x in s['probe_weights']) or abs(sum(s['probe_weights'])-1)>1e-6:
        raise ValueError('invalid probe weights')
    if not 0 < s['vertical_gamma'] <= 1 or s['vertical_queries'] < 1:
        raise ValueError('invalid vertical selection')
    if s['vertical_max_tokens'] is not None and s['vertical_max_tokens'] < 1:
        raise ValueError('invalid optional vertical cap')
    if c['training'] or not all(c[k] for k in ('prefill_only','full_kv_cache','dense_decode')):
        raise ValueError('prefill-only inference contract')
    if len(set(c['layers'])) != len(c['layers']) or any(x not in range(32) for x in c['layers']):
        raise ValueError('invalid selected layers')
    if c['calibration']['samples'] != 3 or c['pilot']['samples_per_task'] != 2:
        raise ValueError('this pilot revision freezes 3 disjoint calibration + 8 evaluation samples')
    for profile,resources in (('pilot',(1,8,128,4)),('formal',(4,8,128,24))):
        if tuple(c[profile][k] for k in ('gpus','cpus','memory_gb','walltime_hours')) != resources or c[profile]['gpu']!='h100-80':
            raise ValueError('resource configuration must match the prepared Slurm scripts')
    if s['token_force_sink'] or s['token_force_diagonal']:
        raise ValueError('unbounded original token-only selector does not implement forced token protection')
    return c


def verify_sources():
    lock = json.loads((RUN/'source-lock.json').read_text())
    for path, expected in lock['files'].items():
        if sha(path) != expected:
            raise ValueError('source changed: '+path)
    return lock


def verify_inputs():
    c = configuration()
    if sha(ORIGIN/'inputs/manifest.json') != ORIGIN_MANIFEST or sha(ORIGIN/'independent-audit.json') != ORIGIN_AUDIT:
        raise ValueError('historical manifest/audit changed')
    original = json.loads((ORIGIN/'inputs/manifest.json').read_text())
    m = json.loads((RUN/'inputs/manifest.json').read_text())
    if m['evaluation'] != original['records'] or m['origin_manifest_sha256'] != ORIGIN_MANIFEST:
        raise ValueError('evaluation data changed')
    if m['config_sha256'] != sha(RUN/'config.json'):
        raise ValueError('configuration changed since preparation')
    evaluation = m['evaluation']
    eval_ids = {(r['task'],r['source_index']) for r in evaluation}
    eval_hashes = {r['input_ids_sha256'] for r in evaluation}
    if len(m['calibration']) != 3 or len({r['key'] for r in m['calibration']}) != 3:
        raise ValueError('exactly 3 unique calibration samples required')
    for item in m['calibration']:
        if (item['task'],item['source_index']) in eval_ids or item['input_ids_sha256'] in eval_hashes:
            raise ValueError('calibration/evaluation leakage')
    pilot = [r for r in evaluation if r['key'] in m['pilot_keys']]
    if len(pilot) != 8 or any(sum(r['task']==t for r in pilot)!=2 for t in TASKS):
        raise ValueError('invalid pilot subset')
    fields = {'ids':'input_ids_sha256','pos':'position_ids_sha256','mask':'attention_mask_sha256',
        'prompt':'rendered_prompt_sha256','raw':'raw_row_sha256'}
    for role,items in (('evaluation',evaluation),('calibration',m['calibration'])):
        directory = ORIGIN/'inputs' if role=='evaluation' else RUN/'inputs'
        for item in items:
            if item['input_tokens'] != item['before_truncation'] or item['input_tokens']+item['max_new_tokens'] > original['model_context_limit']:
                raise ValueError('truncated or overlength example')
            for suffix,field in fields.items():
                if sha(directory/(item['key']+'.'+suffix)) != item[field]:
                    raise ValueError('input bytes changed: '+item['key']+'.'+suffix)
    model = Path(c['model'])
    for field,name in (('model_config_sha256','config.json'),('generation_config_sha256','generation_config.json')):
        if sha(model/name) != original[field]:
            raise ValueError('model configuration changed')
    for name,expected in original['tokenizer_files'].items():
        if sha(model/name) != expected:
            raise ValueError('tokenizer changed')
    for name,expected in original['official_hashes'].items():
        if sha(ORIGIN/'official'/name) != expected:
            raise ValueError('official scorer changed')
    return c,m


def require_approval(profile):
    p = RUN/'approval.json'
    if not p.exists():
        raise PermissionError('Awaiting explicit user approval; no GPU/model execution or submission allowed')
    a = json.loads(p.read_text())
    if a.get('approved') is not True or a.get('profile') != profile or a.get('config_sha256') != sha(RUN/'config.json'):
        raise PermissionError('approval does not match this configuration/profile')
    if a.get('input_manifest_sha256') != sha(RUN/'inputs/manifest.json') or a.get('source_lock_sha256') != sha(RUN/'source-lock.json'):
        raise PermissionError('approval does not match prepared inputs/sources')
    return a
