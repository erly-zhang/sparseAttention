"""Replay the 200 frozen FlexPrefill prompts and record actual per-head branches."""
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent / 'representative_patterns_fast_retry03_20261001'
sys.path.insert(0, str(ORIGINAL))
from protocol import sha, save
from observer import HeadObserver


def verify_observer_sources():
    lock = json.loads((HERE / 'source-lock.json').read_text())
    for name, digest in lock.items():
        assert sha(name) == digest, name
    return sha(HERE / 'source-lock.json')


def run():
    source_hash = verify_observer_sources()
    from run import preflight, runtime_quality, forward_prompt
    from adapter import install
    c, manifest = preflight()
    submitted = HERE / 'submitted.json'
    for _ in range(30):
        if submitted.exists():
            break
        time.sleep(1)
    job = os.environ['SLURM_ARRAY_JOB_ID']
    assert json.loads(submitted.read_text())['job_id'] == job
    shard = int(os.environ['SLURM_ARRAY_TASK_ID'])
    plan = next(x for x in json.loads((ORIGINAL / 'work-plan.json').read_text())
                if x['method'] == 'flexprefill' and x['shard'] == shard)
    items = sorted([x for x in manifest['evaluation'] if x['key'] in plan['keys']],
                   key=lambda x: (-x['input_tokens']**2, x['key']))
    out = HERE / 'outputs' / ('worker-' + str(shard))
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    metadata = {'shard': shard, 'job': os.environ['SLURM_JOB_ID'],
                'node': os.uname().nodename, 'source_lock_sha256': source_hash,
                'original_source_lock_sha256': sha(ORIGINAL / 'source-lock.json'),
                'original_config_sha256': sha(ORIGINAL / 'config.json'),
                'input_manifest_sha256': sha(ORIGINAL / 'inputs/manifest.json'),
                'baseline_options': c['baselines']['flexprefill']}
    save(out / 'start.json', metadata)
    try:
        import torch
        import numpy as np
        from transformers import AutoModelForCausalLM
        assert torch.cuda.device_count() == 1 and 'H100' in torch.cuda.get_device_name()
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        sys.path.insert(0, '/ocean/projects/cis260309p/jzhang114/prefill-routing-deps/FlexPrefill')
        flex = importlib.import_module('flex_prefill.ops.flex_prefill_attention')
        baselines = importlib.import_module('baselines')
        model = AutoModelForCausalLM.from_pretrained(c['model'], local_files_only=True,
            torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2').cuda().eval().requires_grad_(False)
        runtime_quality().install_mlp(model)
        versions = {name: p._version for name, p in model.named_parameters()}
        for item in items:
            torch.manual_seed(c['generation_seed'])
            torch.cuda.manual_seed_all(c['generation_seed'])
            state = install(model, 'flexprefill', layers=c['baseline_layers'], options=c['baselines'])
            original_record = ORIGINAL / 'outputs' / ('formal-' + str(plan['worker'])) / (item['key'] + '-flexprefill.json')
            old = json.loads(original_record.read_text())
            logits_file = original_record.parent / old['logits_file']
            assert sha(logits_file) == old['logits_sha256']
            sample_start = time.monotonic()
            try:
                with HeadObserver(flex, baselines) as observation:
                    logits, cache, ids = forward_prompt(model, item, state, lambda: None)
                assert set(observation.records) == set(range(32))
                actual, expected = logits.cpu().numpy(), np.load(logits_file, allow_pickle=False)
                exact = bool(np.array_equal(actual, expected))
                error = float(np.abs(actual - expected).max())
                assert exact, 'Original FlexPrefill logits mismatch: max_abs_error=' + str(error)
                assert versions == {name: p._version for name, p in model.named_parameters()}
                row = dict(metadata, key=item['key'], task=item['task'], input_tokens=len(ids),
                    original_record_sha256=sha(original_record), original_logits_sha256=old['logits_sha256'],
                    logits_bitwise_equal=exact, logits_max_abs_error=error,
                    layers={str(k): v for k, v in observation.records.items()},
                    elapsed_seconds=time.monotonic() - sample_start)
                save(out / (item['key'] + '.json'), row)
                print('OBSERVED', item['key'], 'logits_equal', exact,
                      'QA_heads', sum(v['query_aware_count'] for v in observation.records.values()), flush=True)
            finally:
                state['close']()
            del logits, cache, observation, actual, expected
            gc.collect()
            torch.cuda.empty_cache()
        save(out / 'complete.json', dict(metadata, records=len(items), status='complete',
            frozen_base_unchanged=True, elapsed_seconds=time.monotonic() - start,
            files={f.name: sha(f) for f in out.glob('*.json')}))
        del model
        gc.collect()
        torch.cuda.empty_cache()
        from usage_summary import aggregate_if_ready
        aggregate_if_ready()
    except Exception as error:
        save(out / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise


if __name__ == '__main__':
    run()
