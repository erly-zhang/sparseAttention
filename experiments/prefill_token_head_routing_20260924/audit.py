"""Read-only, dependency-light audit of canonical InfiniteBench sources/assets."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[2]
CANONICAL = ROOT/'experiments/run_shareprefill_ae3_infinitebench.py'
LLAMA = ROOT/'experiments/run_llama_ae8_autoblock_formal.py'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def constant(path, name):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


def model_audit(model_dir):
    path = Path(model_dir).resolve(strict=True)
    config = json.loads((path/'config.json').read_text())
    if config['model_type'] != 'llama':
        raise ValueError('latest experiment selection is Llama-3.1-8B-Instruct')
    shapes = {}
    # Read safetensors headers only, never instantiate the model here.
    for shard in sorted(path.glob('*.safetensors')):
        with shard.open('rb') as stream:
            size = struct.unpack('<Q', stream.read(8))[0]
            if size > 100_000_000:
                raise ValueError('invalid safetensors header')
            metadata = json.loads(stream.read(size))
        for name, item in metadata.items():
            if '.self_attn.' in name:
                shapes[name] = {'shape': item['shape'], 'dtype': item['dtype']}
    files = {}
    for item in sorted(path.iterdir()):
        if item.is_file() and (item.suffix in ('.json', '.safetensors', '.model') or item.name.endswith('.jinja')):
            files[item.name] = {'bytes': item.stat().st_size, 'sha256': digest(item)}
    h = config['num_attention_heads']; kv = config['num_key_value_heads']
    d = config.get('head_dim', config['hidden_size']//h)
    for layer in range(config['num_hidden_layers']):
        for name, want in [('q_proj', [h*d, config['hidden_size']]), ('k_proj', [kv*d, config['hidden_size']]),
                           ('v_proj', [kv*d, config['hidden_size']]), ('o_proj', [config['hidden_size'], h*d])]:
            key = f'model.layers.{layer}.self_attn.{name}.weight'
            if shapes.get(key, {}).get('shape') != want:
                raise ValueError(f'missing/mismatched projection {key}')
    return {'path': str(path), 'config': config, 'attention_shapes': shapes,
            'files': files, 'q_to_kv_mapping': [i//(h//kv) for i in range(h)],
            'projection_shapes_verified': True}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--out', required=True); p.add_argument('--model')
    args = p.parse_args()
    tasks = constant(CANONICAL, 'TASKS')
    counts = constant(LLAMA, 'TASK_COUNTS')
    files = [CANONICAL, LLAMA, ROOT/'experiments/aggregate_infinitebench_methods.py',
             ROOT/'experiments/benchmark_shareprefill_ae3.py', ROOT/'README.md']
    record = {'git_head': subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'], text=True).strip(),
              'selected_model': 'Llama-3.1-8B-Instruct', 'benchmark': 'InfiniteBench',
              'source_sha256': {str(f.relative_to(ROOT)): digest(f) for f in files},
              'tasks': tasks, 'source_expected_counts': counts, 'source_expected_total': sum(counts.values()),
              'max_new_tokens': constant(CANONICAL, 'INFINITEBENCH_MAX_NEW_TOKENS'),
              'actual_dataset_verified': False, 'pretrained_model_verified': False,
              'decode': 'dense', 'conditions': ['dense','q_only','q_kv'],
              'generation': {'do_sample': False, 'batch_size': 1, 'seed': 42, 'chat': True,
                             'max_length': 131072, 'repetitions': 1},
              'truncation': 'canonical middle truncation to max_length - task generation budget; left padding',
              'missing': ['local checkpoint/tokenizer', 'FlexPrefill task YAMLs, utils and dataset',
                          'lm_eval environment', 'original input reference manifests'],
              'formal_status': 'not approved; not submitted'}
    if args.model:
        record['model'] = model_audit(args.model)
        record['pretrained_model_verified'] = True
    Path(args.out).write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))

if __name__ == '__main__':
    main()
