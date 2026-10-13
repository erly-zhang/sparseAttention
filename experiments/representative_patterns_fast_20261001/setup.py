"""Freeze and validate the prepared source/config/environment entirely on CPU."""
import ast
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from protocol import RUN,ROOT,PRIOR,ORIGIN,configuration,sha,save,verify_inputs


def setup():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('setup requires GPUs hidden')
    if (RUN/'approval.json').exists():
        raise PermissionError('cannot reconfigure an approved experiment')
    c=configuration()
    from planning import work_plan
    m=json.loads((RUN/'inputs/manifest.json').read_text())
    save(RUN/'work-plan.json',work_plan(c,m))
    from baseline_reuse import inspect_baselines
    save(RUN/'baseline-reuse-audit.json',inspect_baselines())
    files=sorted(RUN.glob('*.py'))+[RUN/'work-plan.json',RUN/'baseline-reuse-audit.json',RUN/'config.json',RUN/'env.sh',RUN/'run-calibration.sbatch',RUN/'run-aggregate.sbatch',RUN/'run-formal.sbatch',RUN/'run-audit.sbatch',RUN/'README.md']
    files += [ROOT/'experiments'/name for name in ('token_compacted_sparse.py','jsd_grouped_sparse.py',
        'run_shared_layer_mask_experiment.py','benchmark_shareprefill_ae3.py')]
    files += [ORIGIN/name for name in ('attention.py','adapter.py','protocol.py','inputs/manifest.json','independent-audit.json')]
    files += [PRIOR/name for name in ('env.sh','quality.py','source_pins.json')]
    pins=json.loads((PRIOR/'source_pins.json').read_text())
    files += [Path(name) for name in pins]
    files += [ROOT/'experiments/psc_kv_layer_outputs_20260925/environment.sh']
    from baselines import FLEX
    files += list((FLEX/'flex_prefill').rglob('*.py'))
    overlay=PRIOR/'overlay/minference'
    files += list(overlay.rglob('*.py')) + [Path(c['baselines']['minference']['pattern_file'])]
    files += list(overlay.glob('*.so'))
    for path in RUN.glob('*.py'):
        ast.parse(path.read_text(),filename=str(path))
    for path in (RUN/'env.sh',RUN/'run-calibration.sbatch',RUN/'run-aggregate.sbatch',RUN/'run-formal.sbatch',RUN/'run-audit.sbatch'):
        subprocess.run(['bash','-n',str(path)],check=True)
    versions={name:metadata.version(name) for name in ('torch','transformers','triton','flash-attn','numpy')}
    lock=dict(files={str(p.resolve()):sha(p) for p in sorted(set(files))},versions=versions,
        repository='https://github.com/erly-zhang/sparseAttention',branch='autoblock',
        base_commit='8b21a6a18056eadc00b6248bb319a4af3c6ccc4d',
        original_fix='fixed TopK sorted=True before variable causal keep-count truncation',
        config_sha256=sha(RUN/'config.json'),training=False)
    save(RUN/'source-lock.json',lock)
    tests=unittest.TestLoader().discover(str(RUN),pattern='test_*.py').countTestCases()
    result=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(RUN),'-p','test_*.py','-v'],
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    with (RUN/'cpu-tests.log').open('x') as f:
        f.write(result.stdout)
    print(result.stdout,flush=True)
    if result.returncode:
        raise RuntimeError('CPU validation failed; preserve log')
    from run import preflight
    _,m=preflight()
    save(RUN/'cpu-validation.json',dict(status='passed',tests=tests,source_lock_sha256=sha(RUN/'source-lock.json'),
        input_manifest_sha256=sha(RUN/'inputs/manifest.json'),test_log_sha256=sha(RUN/'cpu-tests.log'),
        versions=versions,syntax_checks='Python AST and bash -n',gpu_visibility='',
        formal_workers=len(c['methods'])*c['formal']['shards'],evaluation_inputs=len(m['evaluation']),methods=len(c['methods']),planned_generation_records=len(m['evaluation'])*len(c['methods']),calibration_inputs=len(m['calibration']),
        gpu_tests_run=False,model_checkpoint_loaded=False,experiment_submitted=False))
    print('CPU SETUP COMPLETE: approval remains pending',flush=True)


if __name__=='__main__':
    setup()
