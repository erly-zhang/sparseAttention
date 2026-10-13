"""Isolated adapter to the repository's canonical InfiniteBench runner.
Dependencies and model are installed; real-data end-to-end benchmark NOT yet validated.
Current review revision has NOT been executed. All experiment runs require user approval.
Run with the canonical arguments; --method is dense, q_only, or q_kv.
Preserves canonical FA2 dense prefill/decode; B uses masked SDPA, C grouped FA2.
"""
from pathlib import Path
import sys
import json
import hashlib
from functools import wraps
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    from smoke import require_allocation
    require_allocation()
    import torch
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError('original InfiniteBench bf16 protocol requires native bf16 GPU')
    import experiments.run_shareprefill_ae3_infinitebench as canonical
    from generation_bridge import GenerationBridge
    from audit import model_audit
    canonical.METHODS = ('dense', 'q_only', 'q_kv')
    # No SharePrefill grouping/calibration is used for position-modulo routing.
    original_parse = canonical.parse_args
    state = {}
    def parse():
        args = original_parse()
        if args.batch_size != 1 or args.dense_layers:
            raise ValueError('batch=1 and routing on every layer required')
        if (args.max_length, args.seed, args.chat, args.limit, args.calibration_scope) != (
                131072, 42, True, -1, 'benchmark_specific'):
            raise ValueError('formal autoblock protocol requires max_length=131072, seed=42, chat, full task, benchmark_specific')
        if any(getattr(args, name, None) for name in (
                'group_config', 'dump_selector_details', 'profile_member_mask_fidelity',
                'oracle_residual_dump_dir', 'ranked_probability_dump_dir',
                'oracle_residual_tokens', 'watched_key_ranges', 'target_token_top_p',
                'target_top_p_start_layer')):
            raise ValueError('AutoBlock selector/profiler options do not apply to routing')
        args.record_sparsity = True
        if args.profile_one_token:
            raise ValueError('profile-only generation is not an evaluation')
        for name in ('task_config_dir', 'flexprefill_root'):
            if not getattr(args, name) or not Path(getattr(args, name)).is_dir():
                raise ValueError(f'explicit existing --{name} required; old machine defaults are not usable')
        if args.create_reference or not args.reference_metrics:
            raise ValueError('A/B/C all require original autoblock --reference_metrics; do not create a new baseline')
        reference = Path(args.reference_metrics).resolve()
        if not reference.is_file():
            raise ValueError('original autoblock input reference is missing')
        if Path(__file__).resolve().parent in reference.parents:
            raise ValueError('reference must come from original autoblock results, not this experiment')
        out = Path(args.output_dir).resolve()
        if Path(__file__).resolve().parent not in out.parents:
            raise ValueError('output must be inside this independent experiment directory')
        cell = out / args.method / args.task
        cell.mkdir(parents=True, exist_ok=False)  # Refuse overwriting/resuming any cell.
        provenance = model_audit(args.model)
        pinned = json.loads((Path(__file__).parent/'records/pretrained_model_provenance.json').read_text())
        if provenance['files'] != pinned['files']:
            raise ValueError('model/tokenizer files differ from the audited Llama-3.1-8B-Instruct checkpoint')
        (cell/'model_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
        (cell/'run_config.json').write_text(json.dumps(vars(args), indent=2)+'\n')
        (cell/'source_sha256.json').write_text(json.dumps({
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
            list(Path(__file__).parent.glob('*.py')) + [Path(canonical.__file__)]}, indent=2)+'\n')
        (cell/'reference_provenance.json').write_text(json.dumps({
            'path': str(reference), 'sha256': hashlib.sha256(reference.read_bytes()).hexdigest(),
            'task_config_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted(Path(args.task_config_dir).rglob('*.yaml'))},
        }, indent=2)+'\n')
        state.update(args=args, cell=cell)
        return args
    canonical.parse_args = parse
    BaseHFLM = canonical.HFLM
    class RoutedHFLM(BaseHFLM):
        def __init__(self, **kwargs):
            if kwargs.get('_attn_implementation') != 'flash_attention_2':
                raise ValueError('canonical FA2 backend required; no silent fallback')
            super().__init__(**kwargs)
            state['tokenizer'] = self.tokenizer
    canonical.HFLM = RoutedHFLM
    def install(model, method, **kwargs):
        if model.config.model_type != 'llama':
            raise ValueError('selected experiment model is Llama-3.1-8B-Instruct')
        from aligned_recorder import PairStats
        patch = SimpleNamespace(selector=SimpleNamespace(stats=PairStats()))
        def on_record(row):
            n = row['valid_prompt_tokens']
            stats = patch.selector.stats
            stats.selected_token_pairs += sum(layer['eligible_pairs'] for layer in row['prefill_pairs_by_layer'])
            stats.causal_token_pairs += len(model.model.layers) * model.config.num_attention_heads * n * (n + 1) // 2
        bridge = GenerationBridge(model, method, state['cell']/'routing_generations.jsonl',
                                  tokenizer=state['tokenizer'], retain_input_ids=False,
                                  defer_output=True, on_record=on_record, event_timing=True)
        state['bridge'] = bridge
        return model, patch, {'method': method, 'routing': 'original_position_mod_num_query_heads',
            'output': 'original_dense' if method == 'dense' else 'scatter_then_explicit_repeat_full_o_proj', 'full_kv_cache': True,
            'full_qkv_projections': True, 'dense_decode': True, 'dense_backend': 'flash_attention_2',
            'routed_prefill_backend': 'sdpa_explicit_mask' if method == 'q_only' else 'flash_attention_2'}
    canonical.install_benchmark_method = install
    # Canonical warmup happens before recorder.install; remove warmup rows then.
    from aligned_recorder import make_recorder
    canonical.GenerationMetricsRecorder = make_recorder(
        canonical.GenerationMetricsRecorder, lambda: state['bridge'])
    original_evaluate = canonical.lm_eval.simple_evaluate
    @wraps(original_evaluate)
    def evaluate(*args, **kwargs):
        kwargs['log_samples'] = True
        return original_evaluate(*args, **kwargs)
    canonical.lm_eval.simple_evaluate = evaluate
    try:
        canonical.main()
    finally:
        if 'bridge' in state:
            state['bridge'].close()
    # Predictions and identity remain in canonical lm_eval samples; routing sidecar
    # supplies exact token IDs, cache dimensions, EOS/cap/criteria and timing.

if __name__ == '__main__':
    main()
