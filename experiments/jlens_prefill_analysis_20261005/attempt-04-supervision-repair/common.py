"""Provenance, approval and legacy integration. Importing does not load a model."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with temp.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def config():
    return read(HERE / 'config.json')


def verify_preparation():
    lock = read(HERE / 'source-lock.json')
    for path, expected in lock['files'].items():
        if sha(path) != expected:
            raise ValueError('Prepared file changed: ' + path)
    manifest = read(HERE / 'input-manifest.json')
    for path, expected in manifest.get('model_files',{}).items():
        stat=Path(path).stat()
        if stat.st_size != expected['bytes'] or stat.st_mtime_ns != expected['mtime_ns']:
            raise ValueError('Frozen model/tokenizer file metadata changed: '+path)
    for path, expected in manifest['files'].items():
        if sha(path) != expected:
            raise ValueError('Input changed: ' + path)
    return manifest


def require_approval(*,allocation=True):
    # This gate precedes model loading, GPU smoke tests and sbatch submission.
    path = HERE / 'approval.json'
    if not path.exists():
        raise PermissionError('Awaiting user approval of the reported J-lens experiment')
    approval = read(path)
    if approval.get('approved') is not True:
        raise PermissionError('Experiment not approved')
    for field, filename in [('config_sha256','config.json'),
                            ('source_lock_sha256','source-lock.json'),
                            ('input_manifest_sha256','input-manifest.json')]:
        if approval.get(field) != sha(HERE / filename):
            raise PermissionError('Approval does not match ' + filename)
    verify_preparation()
    if allocation and 'SLURM_JOB_ID' not in os.environ:
        raise PermissionError('Model execution requires the approved Slurm allocation')
    return approval


def legacy():
    name = '_jlens_historical_run'
    if name not in sys.modules:
        directory = Path(config()['historical_run'])
        sys.path.insert(0, str(directory))
        spec = importlib.util.spec_from_file_location(name, directory / 'run.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def atomic_torch_save(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    torch.save(value, temp)
    os.replace(temp, path)


def runtime():
    import torch
    import transformers
    if transformers.__version__ != '4.44.0':
        raise ValueError('Must keep the frozen Transformers 4.44.0 runtime')
    if torch.cuda.device_count() != 1 or 'H100' not in torch.cuda.get_device_name():
        raise ValueError('Expected one H100')
    torch.set_num_threads(config()['resources']['cpus'])
    torch.manual_seed(config()['seed'])
    torch.cuda.manual_seed_all(config()['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = config()['model']
    tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        name, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2').cuda().eval().requires_grad_(False)
    if (model.config.num_hidden_layers, model.config.hidden_size) != (32,4096):
        raise ValueError('Unexpected model dimensions')
    return model, tokenizer


class Llama44LensModel:
    """LensModel protocol adapter; uses unchanged norm, LM head and text decoder.

    Unlike jlens.from_hf, this does not call get_text_config(), which was added
    after the frozen 4.44 runtime. It does not alter the Jacobian estimator.
    """
    def __init__(self, hf_model, tokenizer):
        self.hf = hf_model
        self.tokenizer = tokenizer
        self.layers = hf_model.model.layers
        self.n_layers = hf_model.config.num_hidden_layers
        self.d_model = hf_model.config.hidden_size

    def encode(self, text, *, max_length=128):
        return self.tokenizer(text, return_tensors='pt', truncation=True,
                              max_length=max_length).input_ids.to(self.hf.device)

    def forward(self, input_ids):
        return self.hf.model(input_ids=input_ids, use_cache=False, return_dict=True)

    def unembed(self, residual):
        weight = self.hf.lm_head.weight
        return self.hf.lm_head(self.hf.model.norm(residual.to(weight)))


def positions_for(length, options, evidence=()):
    import numpy as np
    if length < 1:
        raise ValueError('Empty input')
    result = set(range(min(length, options['prefix_positions'])))
    result.update(range(max(0,length-options['tail_positions']), length))
    result.update(int(x) for x in np.linspace(0,length-1,options['uniform_positions']))
    result.update(evidence)
    if any(x < 0 or x >= length for x in result):
        raise ValueError('Position outside original input')
    return sorted(result)
