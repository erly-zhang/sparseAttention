"""Fail-closed checkpoint choice and immutable training provenance."""
from dataclasses import asdict
import json
import math
from pathlib import Path
import subprocess
import torch
from experiments.learned_mask_nnonly_20260914 import pipeline as training
from experiments.learned_mask_nnonly_20260914.model import Config, Predictor, VARIANTS


def read(path):
    return json.loads(Path(path).read_text())


def audit_training():
    training.check_proof()
    _, groups, _ = training.metadata()
    selected, jobs = {}, []
    for variant in VARIANTS:
        folder = training.OUTPUT/variant
        config, summary, exit_info = [read(folder/f) for f in ('config.json', 'summary.json', 'exit.json')]
        assert exit_info['returncode'] == 0 and exit_info['error'] is None
        assert summary['status'] == 'completed' and summary['variant'] == variant
        assert config['source_sha256'] == training.source_hashes()
        job = str(config['job_id'])
        submission = read(training.OUTPUT/'submissions'/f'{variant}.json')
        assert str(submission['job']).split(';')[0] == job
        jobs.append(job)
        losses = []
        for fold in range(3):
            directory = folder/f'fold{fold}'
            provenance = read(directory/'provenance.json')
            manifest_path = training.OUTPUT/f'fold_{fold}.json'
            manifest = read(manifest_path)
            result = read(directory/variant/'validation.json')
            assert provenance['manifest_sha256'] == training.sha(manifest_path)
            assert provenance['groups_sha256'] == training.sha(training.GROUPS)
            assert provenance['config'] == asdict(Config())
            assert result['variant'] == variant and result['cases'] == 7424
            assert result['grouping_independent_of_validation'] is False
            assert len(result['history']) == 5
            assert all(h['updates'] == 14848 and math.isfinite(h['train_loss'])
                       and math.isfinite(h['validation_loss']) for h in result['history'])
            expected = {(e['input_ids_sha256'], e['layer'], e['start'], h)
                        for e in manifest['splits']['validation']
                        for g in groups['layers'][str(e['layer'])]
                        for h in g['members'] if h != g['representative']}
            keys = [(r['input_ids_sha256'], r['layer'], r['start'], r['head']) for r in result['rows']]
            assert len(keys) == len(set(keys)) == 7424 and set(keys) == expected
            loss = result['mean_cross_entropy']
            assert math.isfinite(loss)
            assert loss == next(r['mean_cross_entropy'] for r in summary['results'] if r['fold'] == fold)
            assert (directory/variant/'best.pt').is_file()
            losses.append(loss)
        fold = min(range(3), key=lambda i: (losses[i], i))
        checkpoint = folder/f'fold{fold}'/variant/'best.pt'
        assert fold == summary['selected_fold'] and checkpoint == Path(summary['checkpoint'])
        assert summary['checkpoint_sha256'] == training.sha(checkpoint)
        assert summary['groups_sha256'] == training.sha(training.GROUPS)
        selected[variant] = dict(fold=fold, losses=losses, checkpoint=str(checkpoint),
            checkpoint_sha256=training.sha(checkpoint), groups_sha256=training.sha(training.GROUPS),
            config_sha256=training.sha(folder/'config.json'), job_id=job)
    accounting = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', ','.join(jobs),
                        '--format=JobID,State,ExitCode'], text=True)
    states = {row.split('|')[0]: row.split('|')[1:3] for row in accounting.splitlines() if row.strip()}
    assert all(states.get(j) == ['COMPLETED', '0:0'] for j in jobs), states
    return dict(status='passed', selected=selected, accounting=accounting,
                grouping_independent_of_validation=False, config=asdict(Config()),
                training_source_sha256=training.source_hashes(), dependency_sha256=training.dependencies())


def load(variant, proof, device):
    assert proof['status'] == 'passed' and proof['config'] == asdict(Config())
    assert proof['training_source_sha256'] == training.source_hashes()
    item = proof['selected'][variant]
    assert training.sha(item['checkpoint']) == item['checkpoint_sha256']
    assert training.sha(training.GROUPS) == item['groups_sha256']
    groups = read(training.GROUPS)['layers']
    model = Predictor(groups, variant, Config()).to(device=device, dtype=torch.float32)
    model.load_state_dict(torch.load(item['checkpoint'], map_location=device, weights_only=True), strict=True)
    model.eval().requires_grad_(False)
    return model
