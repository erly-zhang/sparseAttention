"""Independent CPU preparation for the explicitly authorized speed pilot."""
import hashlib
import json
from pathlib import Path
import random
import struct
import sys

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent / 'jlens_prefill_analysis_20261005'
sys.path.insert(0, str(ANALYSIS))
from common import config, legacy, read, save, sha


def main():
    from transformers import AutoTokenizer
    c = config()
    old = legacy()
    historical = read(Path(c['historical_run']) / 'inputs/manifest.json')
    tokenizer = AutoTokenizer.from_pretrained(c['model'], local_files_only=True)
    excluded = {(x['task'], x['source_index']) for x in historical['evaluation']}
    full = []
    for x in historical['calibration']:
        if (x['task'], x['source_index']) in excluded:
            raise ValueError('Calibration/evaluation overlap')
        directory = Path(c['historical_run']) / 'inputs'
        text = (directory / (x['key'] + '.prompt')).read_text()
        ids_path = directory / (x['key'] + '.ids')
        if sha(ids_path) != x['input_ids_sha256']:
            raise ValueError('Changed calibration input')
        ids = list(struct.unpack('<' + 'q' * x['input_tokens'], ids_path.read_bytes()))
        if tokenizer.encode(text, add_special_tokens=False) != ids:
            raise ValueError('Retokenization changed')
        full.append(dict(key=x['key'], task=x['task'], source_index=x['source_index'],
                         text=text, input_ids=ids, original_tokens=len(ids),
                         input_ids_sha256=x['input_ids_sha256']))
    origin_manifest = read(old.ORIGIN / 'inputs/manifest.json')
    templates, _, _ = old.official(old.ORIGIN / 'official')
    from protocol import render
    spec = origin_manifest['datasets']['code_debug']
    dataset = old.ORIGIN / 'data' / spec['file']
    if sha(dataset) != spec['sha256']:
        raise ValueError('Changed Code.Debug dataset')
    rows = dataset.read_text().splitlines()
    order = [i for i in range(len(rows)) if ('code_debug', i) not in excluded]
    random.Random(20261006).shuffle(order)
    for i in order:
        row = json.loads(rows[i])
        text = tokenizer.apply_chat_template(
            [dict(role='user', content=render(row, 'code_debug', templates))],
            tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(text, add_special_tokens=False)
        if 65536 <= len(ids) <= 131032:
            full.append(dict(key='speed-code_debug-' + str(i), task='code_debug',
                             source_index=i, text=text, input_ids=ids,
                             original_tokens=len(ids), input_ids_sha256=
                             hashlib.sha256(struct.pack('<'+'q'*len(ids), *ids)).hexdigest()))
            break
    if len(full) != 4:
        raise ValueError('Expected four independent long prompts')
    wiki = read(ANALYSIS / 'fit-prompts.json')[:2]
    payload = dict(wikitext=wiki, infinitebench=full)
    save(HERE / 'inputs.json', payload)
    save(HERE / 'config.json', dict(
        model=c['model'], source_layers=list(range(31)), target_layer=31,
        skip_first=16, short_tokens=128, short_dim_batch=32,
        long_dim_batch=1, long_backward_batches=4,
        long_input_policy='complete original token IDs, no truncation',
        decision_ratio=2.0,
        decision_rule='Use full-context InfiniteBench only if all tasks are feasible and projected per-prompt time is <=2x measured WikiText median; otherwise retain WikiText. Projections are not completed full fits.',
        checkpoint_test='non-reentrant activation checkpointing, same estimator, no dropout',
        snapshot_seconds=300, fit_prompts_planned=1000,
        authorization='User explicitly requested a speed pilot and selected complete InfiniteBench contexts; no approval for the formal analysis pipeline.',
        formal_pipeline_approved=False))
    locked = [HERE/'prepare.py', HERE/'benchmark.py', HERE/'run.sbatch', HERE/'config.json', HERE/'inputs.json',
              ANALYSIS/'common.py', ANALYSIS/'progress.py']
    locked += list(Path(c['jlens_repository']).glob('jlens/*.py'))
    save(HERE/'authorization.json', dict(approved=True, scope='speed pilot only',
        user_instruction='先比较 WikiText 与完整 InfiniteBench 拟合速度；差距不大用 InfiniteBench，否则 WikiText',
        files={str(p):sha(p) for p in locked}, model=c['model']))
    print(json.dumps(dict(status='PASS', evaluation_overlap=0,
        long_inputs=[{k:x[k] for k in ('key','task','source_index','original_tokens')} for x in full],
        wikitext_prompts=len(wiki), complete_context=True), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
