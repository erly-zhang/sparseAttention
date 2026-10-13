"""CPU-only preparation; never submits or loads an inference model."""
from pathlib import Path
import ast
import hashlib
import json
import os
import re
import subprocess

HERE = Path(__file__).resolve().parent


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Hide GPUs during preparation'
    assert not (HERE / 'approval.json').exists(), 'Cannot change an approved plan'
    c = read(HERE / 'config.json')
    source = (HERE / c['source_experiment']).resolve()
    analysis = (HERE / c['source_analysis']).resolve()
    manifest = read(source / 'input-manifest.json')
    item, = [i for i in manifest['evaluation'] if i['key'] == c['sample']]
    source_hashes = {}
    for p, expected in read(source / 'source-lock.json')['files'].items():
        assert sha(p) == expected, p
        source_hashes[p] = expected
    import numpy as np
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(read(source / 'config.json')['model'], local_files_only=True)
    inputs = {}
    for suffix in ['prompt', 'ids', 'pos', 'mask']:
        path, = [Path(p) for p in manifest['files'] if p.endswith(c['sample'] + '.' + suffix)]
        assert sha(path) == manifest['files'][str(path)]
        source_hashes[str(path)] = manifest['files'][str(path)]
        inputs[suffix] = str(path)
    prompt = Path(inputs['prompt']).read_text()
    ids = np.fromfile(inputs['ids'], dtype='<i8').tolist()
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    assert encoded['input_ids'] == ids
    assert len(ids) == item['input_tokens']
    answer = item['answers'][0]
    assert prompt.count(answer) == 1
    start = prompt.index(answer)
    end = start + len(answer)
    hits = [i for i, (a,b) in enumerate(encoded['offset_mapping']) if a < end and b > start]
    after = max(item['position_groups']['kv_evidence'])
    assert after >= hits[-1] and after < len(ids) - 1
    uuid = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
    pairs = re.findall('"(' + uuid + ')"\\s*:\\s*"(' + uuid + ')"', prompt)
    queried_key, = [k for k,v in pairs if v == answer]
    wrong = next(v for _,v in pairs if v != answer and v.startswith(answer[:2]))
    target_ids = tokenizer(answer, add_special_tokens=False)['input_ids']
    wrong_ids = tokenizer(wrong, add_special_tokens=False)['input_ids']
    artifacts = {}
    for method in c['methods']:
        meta = source / 'outputs' / f"replay-{method}-{item['analysis_shard']}" / (item['key'] + '.json')
        r = read(meta)
        p = meta.parent / r['artifact_file']
        assert sha(p) == r['artifact_sha256']
        source_hashes[str(p)] = r['artifact_sha256']
        source_hashes[str(meta)] = sha(meta)
        artifacts[method] = str(p)
    lens = source / 'outputs/lens.pt'
    assert sha(lens) == read(source / 'outputs/lens-provenance.json')['lens_sha256']
    source_hashes[str(lens)] = sha(lens)
    plan = []
    def add(method, kind, **extra):
        plan.append(dict(index=len(plan), method=method, kind=kind, **extra))
    for method in c['methods']:
        add(method, 'baseline')
    for layer in c['layers']:
        for site in c['sites']:
            for kind in c['steering_conditions']:
                for alpha in c['steering_strengths']:
                    add(c['activation_method'], kind, layer=layer, site=site, alpha=alpha)
            for kind in c['patch_conditions']:
                add(c['activation_method'], kind, layer=layer, site=site, alpha=1.0)
    text_inputs = {}
    for method in c['methods']:
        for site in c['sites']:
            for target, value in [('correct', answer), ('wrong', wrong)]:
                add(method, 'text_' + target, site=site)
                name = site + '-' + target
                if name not in text_inputs:
                    offset = prompt.rfind('\nKey:') if site == 'prompt_end' else prompt.index('"', end) + 1
                    assert offset > 0
                    hint = '\nRetrieval note: the value for key "' + queried_key + '" is "' + value + '".\n'
                    modified = prompt[:offset] + hint + prompt[offset:]
                    new_ids = tokenizer(modified, add_special_tokens=False)['input_ids']
                    assert len(new_ids) + c['generation']['max_new_tokens'] <= 131072
                    out = HERE / 'inputs' / (name + '.ids')
                    out.parent.mkdir(exist_ok=True)
                    np.asarray(new_ids, dtype='<i8').tofile(out)
                    (HERE / 'inputs' / (name + '.prompt')).write_text(modified)
                    text_inputs[name] = dict(ids=str(out), input_tokens=len(new_ids), sha256=sha(out), insertion_char_offset=offset)
    assert len(plan) == 105
    prepared = dict(item=item, inputs=inputs, sites={'source_after':after,'prompt_end':len(ids)-1},
                    source_value_positions=hits, source_after_token=tokenizer.decode([ids[after]]),
                    queried_key=queried_key, correct_value=answer, wrong_value=wrong,
                    target_ids=target_ids, wrong_ids=wrong_ids, artifacts=artifacts,
                    lens=str(lens), model=read(source / 'config.json')['model'],
                    text_inputs=text_inputs, source_sha256=source_hashes,
                    model_files=manifest['model_files'], oracle_intervention=True,
                    plan=plan, prepared_without_model_forward=True)
    save(HERE / 'prepared.json', prepared)
    files = list(HERE.glob('*.py')) + list(HERE.glob('*.sh')) + list(HERE.glob('*.sbatch'))
    files += [HERE / 'config.json', HERE / 'prepared.json', HERE / 'PROTOCOL.md']
    for p in HERE.glob('*.py'):
        ast.parse(p.read_text(), filename=str(p))
    for p in list(HERE.glob('*.sh')) + list(HERE.glob('*.sbatch')):
        subprocess.run(['bash','-n',str(p)],check=True)
    save(HERE / 'lock.json',dict(files={str(p):sha(p) for p in files}))
    print('CPU PREPARATION PASS', len(plan), 'trials; source', after, 'end', len(ids)-1, 'NO JOBS SUBMITTED')


if __name__ == '__main__':
    main()
