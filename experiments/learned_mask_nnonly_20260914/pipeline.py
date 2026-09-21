"""Fixed historical AE, four independent NN-only jobs. Never fit an AE."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path('/project/yl258/jz648/sparseAttention')
PACKAGE = 'experiments.learned_mask_nnonly_20260914'
SOURCE = Path(__file__).parent
OUTPUT = ROOT / 'experiments/outputs/learned_mask_nnonly_20260914'
TEACHER = ROOT / 'experiments/outputs/learned_mask_20260914_stage1/teacher'
GROUPS = ROOT / 'experiments/data/kv_randomscore_20260910/calibration/shareprefill_ae_k3_head_groups.json'
REFERENCE = ROOT / 'experiments/data/kv_randomscore_20260910/references/original_autoblock/online_metrics.jsonl'
VARIANTS = ('per_head', 'per_cluster', 'per_layer', 'full_mask')
PROTOCOL = 'leave_one_input_out_fixed_groups'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def source_hashes():
    return {p.name: sha(p) for p in SOURCE.iterdir() if p.suffix in {'.py', '.sbatch'}}


def dependencies():
    return {str(p): sha(p) for p in (GROUPS, REFERENCE, TEACHER/'records.json',
            TEACHER/'capture_verified.json', ROOT/'experiments/token_compacted_sparse.py')}


def metadata():
    from .model import validate_groups
    proof = json.loads((TEACHER/'capture_verified.json').read_text())
    records = json.loads((TEACHER/'records.json').read_text())
    groups = json.loads(GROUPS.read_text())
    if proof.get('records') != 768 or proof.get('debug') is not False or len(records) != 768:
        raise ValueError('Complete real 768-record capture required')
    if proof.get('teacher_state') != 'dense_post_rope' or proof.get('formal_input_overlap') != 0:
        raise ValueError('Wrong teacher convention or formal overlap')
    if validate_groups(groups['layers']) != 32 or len(groups['layers']) != 32:
        raise ValueError('Expected 32-layer, 32-head Llama grouping')
    if any(len(g) != 3 for g in groups['layers'].values()):
        raise ValueError('Historical K=3 required at every layer')
    families = sorted({e['family_id'] for e in records})
    if families != ['kv_retrieval-0', 'kv_retrieval-1', 'kv_retrieval-2']:
        raise ValueError('Wrong calibration families')
    if sorted(groups.get('calibration_sample_ids', [])) != families:
        raise ValueError('Historical grouping calibration provenance mismatch')
    reserved = [json.loads(s)['input_ids_sha256'] for s in REFERENCE.read_text().splitlines() if s.strip()]
    if len(reserved) != 497 or len(set(reserved)) != 497:
        raise ValueError('Expected 497 unique reference identities')
    identities = []
    for family in families:
        subset = [e for e in records if e['family_id'] == family]
        identity = {e['input_ids_sha256'] for e in subset}
        if len(subset) != 256 or len(identity) != 1 or identity & set(reserved):
            raise ValueError('Incomplete family or formal-reference overlap')
        identities.extend(identity)
        starts = None
        for layer in range(32):
            tiles = sorted(e['start'] for e in subset if e['layer'] == layer)
            if len(tiles) != 8 or len(set(tiles)) != 8 or (starts is not None and tiles != starts):
                raise ValueError('Eight aligned unique tiles per layer required')
            starts = tiles
    if len(set(identities)) != 3:
        raise ValueError('Duplicate input identities')
    return records, groups, families


def prepare():
    import torch
    from .data import validate_manifest, validate_group_provenance, load_record
    from .model import Predictor, teacher_loss, final_mask
    from .train import cases
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Prepare/debug in a valid, idle developer GPU allocation')
    OUTPUT.mkdir(parents=True, exist_ok=False)
    begin = time.time()
    print('PREPARE_START', socket.gethostname(), os.environ['SLURM_JOB_ID'], flush=True)
    records, groups, families = metadata()
    # Reuse exact network/kernel debug, but rerun changed split and training tests.
    old = ROOT/'experiments/outputs/learned_mask_20260914_preflight/preflight_passed.json'
    old_proof = json.loads(old.read_text())
    if old_proof.get('status') != 'passed' or old_proof['source_sha256']['model.py'] != sha(SOURCE/'model.py'):
        raise ValueError('Network differs from the GPU/kernel-tested snapshot')
    expected_kernel = old_proof['dependency_sha256']['experiments/token_compacted_sparse.py']
    if expected_kernel != sha(ROOT/'experiments/token_compacted_sparse.py'):
        raise ValueError('Compact kernel changed since GPU numerical validation')
    subprocess.run([sys.executable, '-m', 'unittest', PACKAGE+'.test_model'], check=True, cwd=ROOT)
    print('UNIT_TESTS_PASSED', flush=True)
    plans = []
    for fold, family in enumerate(families):
        manifest = dict(schema='learned_mask_dense_teacher_v1', protocol=PROTOCOL,
            validation_family=family, grouping_scope='historical_fixed_all_calibration',
            grouping_independent_of_validation=False, group_fit_family_ids=families,
            groups_sha256=sha(GROUPS), protected_reference=str(REFERENCE),
            protected_reference_sha256=sha(REFERENCE), splits={
                'train': [e for e in records if e['family_id'] != family],
                'validation': [e for e in records if e['family_id'] == family]})
        path = OUTPUT/f'fold_{fold}.json'
        path.write_text(json.dumps(manifest, indent=2))
        checked = validate_manifest(path)
        validate_group_provenance(checked, GROUPS)
        plans.append(dict(fold=fold, validation_family=family, manifest=str(path), manifest_sha256=sha(path)))
        print('FOLD_AUDIT_PASSED', fold, flush=True)
    # One real tile tests all member-head losses, budgets and four optimizers.
    entry = min((e for e in records if e['layer'] == 0), key=lambda e: e['start'])
    record = load_record(entry, 'cuda')
    counts = {}
    for variant in VARIANTS:
        torch.manual_seed(42)
        net = Predictor({'0': groups['layers']['0']}, variant).cuda()
        optimizer = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.01)
        steps = 0
        for head, q, k, p, protect, core, full, condition, fixed in cases(net, record):
            scores = net(q, k, p, condition, 0, head)
            end = record['start'] + len(record['q'])
            loss = teacher_loss(scores, record['teacher_p'][head], fixed, end)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite real-data NN loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in net.parameters()):
                raise ValueError('Nonfinite real-data gradient')
            optimizer.step()
            mask = final_mask(scores.detach(), core, full, protect, end, net.config, variant)
            if int(mask.sum()) != min(10240, end) or not mask[protect].all():
                raise ValueError('Budget/protection violation')
            steps += 1
        if steps != 29:
            raise ValueError('All 29 member heads must be tested')
        counts[variant] = steps
        print('REAL_NN_DEBUG_PASSED', variant, steps, flush=True)
        del net, optimizer
    proof = dict(status='passed', timestamp=time.time(), started=begin,
        source_sha256=source_hashes(), dependency_sha256=dependencies(), plans=plans,
        real_member_steps=counts, reused_gpu_kernel_proof=str(old),
        grouping_independent_of_validation=False, records=768,
        grouping_scope='historical_fixed_all_calibration',
        purpose='NN-only training preflight, not generation smoke or benchmark accuracy')
    (OUTPUT/'preflight_passed.json').write_text(json.dumps(proof, indent=2))
    print('NNONLY_PREFLIGHT_PASSED', flush=True)


def check_proof():
    proof = json.loads((OUTPUT/'preflight_passed.json').read_text())
    if proof.get('status') != 'passed' or proof['source_sha256'] != source_hashes():
        raise ValueError('Exact-source NN-only preflight required')
    if proof['dependency_sha256'] != dependencies():
        raise ValueError('Dependencies changed')
    for plan in proof['plans']:
        if sha(plan['manifest']) != plan['manifest_sha256']:
            raise ValueError('Fold manifest changed')
    return proof


def submit():
    if not socket.gethostname().split('.')[0].startswith('login'):
        raise RuntimeError('Submit only from an authenticated login node')
    proof = check_proof()
    submissions = OUTPUT/'submissions'
    submissions.mkdir(exist_ok=False)
    queue = subprocess.check_output(['squeue', '-u', 'jz648', '-h', '-o', '%j'], text=True)
    if any(('nnonly-'+v) in queue.splitlines() for v in VARIANTS):
        raise RuntimeError('A matching NN-only job is already queued')
    for variant in VARIANTS:
        if (OUTPUT/variant).exists():
            raise RuntimeError('Existing result: inspect before retry')
        command = ['sbatch', '--parsable', '--job-name=nnonly-'+variant,
                   str(SOURCE/'formal.sbatch'), variant]
        job = subprocess.check_output(command, text=True).strip()
        marker = submissions/(variant+'.json')
        with marker.open('x') as stream:
            json.dump(dict(job=job, variant=variant, host=socket.gethostname(),
                timestamp=time.time(), preflight_timestamp=proof['timestamp'], command=command), stream, indent=2)
        print('SUBMITTED', variant, job, flush=True)


def train_variant(variant):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm GPU allocation required')
    proof = check_proof()
    path = OUTPUT/variant
    path.mkdir(exist_ok=False)
    begin, code, error = time.time(), 0, None
    config = dict(variant=variant, job_id=os.environ['SLURM_JOB_ID'], source_sha256=source_hashes(),
        grouping_scope='historical_fixed_all_calibration', grouping_independent_of_validation=False,
        groups_path=str(GROUPS), groups_sha256=sha(GROUPS), ae_training=False,
        reused_teacher=str(TEACHER), teacher_records=768, nn_epochs=5,
        nn_lr=.001, nn_weight_decay=.01, seed=42, resource_cap='one A100, six hours',
        purpose='offline NN training; no formal 497-input generation')
    (path/'config.json').write_text(json.dumps(config, indent=2))
    results = []
    try:
        for plan in proof['plans']:
            fold = plan['fold']
            destination = path/f'fold{fold}'
            command = [sys.executable, '-u', '-m', PACKAGE+'.train', '--manifest', plan['manifest'],
                '--groups', str(GROUPS), '--output', str(destination), '--variant', variant,
                '--device', 'cuda', '--epochs', '5', '--learning-rate', '.001', '--seed', '42']
            print('NN_FOLD_START', variant, fold, flush=True)
            with (path/f'fold{fold}.log').open('x') as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            result_path = destination/variant/'validation.json'
            result = json.loads(result_path.read_text())
            result.pop('rows')
            result.update(fold=fold, validation_family=plan['validation_family'],
                validation_path=str(result_path), checkpoint=str(destination/variant/'best.pt'))
            results.append(result)
            print('NN_FOLD_FINISHED', variant, fold, result['mean_cross_entropy'], flush=True)
        import math
        if not all(math.isfinite(r['mean_cross_entropy']) for r in results):
            raise ValueError('Nonfinite fold loss')
        selected = min(results, key=lambda r: (r['mean_cross_entropy'], r['fold']))
        (path/'summary.json').write_text(json.dumps(dict(status='completed', variant=variant,
            results=results, selected_fold=selected['fold'], checkpoint=selected['checkpoint'],
            checkpoint_sha256=sha(selected['checkpoint']), groups_path=str(GROUPS),
            groups_sha256=sha(GROUPS), criterion='minimum validation mean_cross_entropy; fold-id tie-break',
            grouping_independent_of_validation=False, independent_test=False), indent=2))
    except BaseException as exc:
        code, error = 1, repr(exc)
        raise
    finally:
        (path/'exit.json').write_text(json.dumps(dict(returncode=code, error=error,
            start_time=begin, end_time=time.time()), indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'submit', *VARIANTS))
    action = parser.parse_args().action
    if action == 'prepare':
        prepare()
    elif action == 'submit':
        submit()
    else:
        train_variant(action)


if __name__ == '__main__':
    main()
