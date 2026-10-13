"""Q-only prefill with alternating sparse layers; no learned correction."""
import hashlib
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SPARSE_LAYERS = tuple(range(0, 32, 2))
DENSE_LAYERS = tuple(range(1, 32, 2))
METHODS = ('dense', 'native16', 'repeat16')
MODES = {'native16': 'native', 'repeat16': 'repeat'}
ASSEMBLY_SHA = '3d49d9cc58117081000e22886f891b07cf91681cbc68ef9ae9043d244587531d'
ADAPTER_SHA = 'a14e7a1b11462f9cbfb02e43bb979ef7fc3277df21f186e9a9b4c43b10231397'
ROUTING_SHA = '8abd4dc5bd21a69d6af7c2ef6632c3cd545fe1a0a51b8e2c79dd1e270ee652dd'


def load_pinned(name, path, expected):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f'Frozen dependency changed: {path}')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dependencies(root=ROOT):
    root = Path(root)
    prior = root / 'experiments/head_slot_kv_ablation_20260928'
    assembly = load_pinned('repeat16_assembly', prior / 'assembly.py', ASSEMBLY_SHA)
    # The frozen adapter imports assembly by its original module name.
    old = sys.modules.get('assembly')
    try:
        sys.modules['assembly'] = assembly
        adapter = load_pinned('repeat16_adapter', prior / 'adapter.py', ADAPTER_SHA)
    finally:
        if old is None:
            sys.modules.pop('assembly', None)
        else:
            sys.modules['assembly'] = old
    route = root / 'experiments/prefill_token_head_routing_20260924/routing.py'
    if not route.exists():
        route = root / 'artifacts/psc_kl_quality/source_audit/experiments/prefill_token_head_routing_20260924/routing.py'
    return assembly, adapter, load_pinned('repeat16_routing', route, ROUTING_SHA)


def install(model, method, *, backend='flash_attention_2', root=ROOT):
    if method not in METHODS:
        raise ValueError(f'Unknown method: {method}')
    cfg = model.config
    if (cfg.num_hidden_layers, cfg.num_attention_heads,
            cfg.num_key_value_heads, cfg.hidden_size) != (32, 32, 8, 4096):
        raise ValueError('Production configuration requires Llama3.1-8B dimensions')
    if method == 'dense':
        return None
    _, adapter, routing = dependencies(root)
    return adapter.install(model, MODES[method], routing.routed_attention,
                           dense_layers=DENSE_LAYERS, backend=backend, kv_mode='q_only')


def validate_counts(method, counts, input_tokens):
    if method not in METHODS or input_tokens < 1:
        raise ValueError('Invalid method or sequence length')
    counts = {int(k): v for k, v in counts.items()}
    expected = {} if method == 'dense' else {
        layer: input_tokens * (input_tokens + 1) // 2 for layer in SPARSE_LAYERS}
    if counts != expected:
        raise ValueError('Wrong routed layers or Q-only causal pair count')


def protocol():
    return dict(methods=list(METHODS), sparse_layers=list(SPARSE_LAYERS),
        dense_layers=list(DENSE_LAYERS), layer_numbering='zero-based',
        kv_mode='q_only', owner='original_position % 32',
        output='merge query rows by original position; repeat each 128-d vector into32 slots',
        scale=1.0, prefill_only=True, full_kv_cache=True, dense_decode=True,
        learned_adapter=False, original_o_proj=True,
        backend='Q-only uses masked SDPA fallback; dense uses FA2',
        warning='16/32 sparse layers is not 50% attention-pair sparsity or a promised2x speedup')
