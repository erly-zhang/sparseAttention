"""Freeze the authorized Full-KV raw-QK v2 K2048 HMMT completion run."""
import ast
import datetime
import hashlib
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zoneinfo import ZoneInfo

E = Path(__file__).resolve().parents[1]
BASE = E/'results/hmmt_w1_raw_qk_fast_full_hybrid_k1024_20261012_packed32'
ROOT = E/'results/hmmt_full_w1_raw_qk_fast_k2048_20261014'


def read(p): return json.loads(p.read_text())
def write(p, value): p.write_text(json.dumps(value, indent=2)+'\n')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert not ROOT.exists(), 'Never overwrite a run'
    ROOT.mkdir()
    for folder in ('source', 'inputs', 'data'):
        files = [p for p in (BASE/folder).rglob('*') if p.is_file()
                 and '__pycache__' not in p.parts and p.suffix != '.pyc']
        for parent in {(ROOT/folder/p.relative_to(BASE/folder)).parent for p in files}:
            parent.mkdir(parents=True, exist_ok=True)
        def copy(p):
            with p.open('rb') as src, (ROOT/folder/p.relative_to(BASE/folder)).open('xb') as dst:
                shutil.copyfileobj(src, dst)
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(copy, files))
    for name in ('model_assets_sha256.json', 'prompt_protection_stats.json'):
        shutil.copyfile(BASE/name, ROOT/name)
    (ROOT/'logs').mkdir()
    (ROOT/'reference_previous').mkdir()
    cfg = read(BASE/'config.json')
    cell = dict(cfg['cells'][0])
    assert cell['storage'] == 'full'
    cell.update(name='hmmt_k2048_w1_raw_qk_fast_full_protected', K=2048)
    cfg.update(created=datetime.datetime.now(ZoneInfo('America/New_York')).isoformat(),
               experiment='hmmt_full_w1_raw_qk_fast_k2048_v2', cells=[cell],
               scope_authorization='2026-10-14 user explicitly requests full HMMT K2048 optimized raw-QK Full-KV W1.',
               rank_backend_selection='Fixed stable_sort, identical to the completed K1024 Full-KV v2 run.',
               compaction_interval=None, baseline_root=str(E/'results/hmmt_full_w1_optimized_k2048_protected_20261009'),
               baseline_budget=2048,
               compaction='None; retain all KV and select protected K for attention every decode step.',
               caveats=[
                   '60 canonical questions x2 sampled rollouts; accuracy per rollout, not pass@2.',
                   'K2048 includes prompt and recent64. Raw BF16 QK selection; attention softmax retained.',
                   'Identical frozen v2 inference implementation and stable_sort backend to the completed K1024 run; only K changes.',
                   'One private representative Q head per layer/step, correct GQA; all KV heads share positions, not values.',
                   'Full KV retains all history; no permanent eviction or random4096 pool.',
                   'GPU-synchronized timings include selection/gather/attention/sampling; exclude loading/warmup/serialization.',
                   'Batch2 seconds are per question batch, not per individual answer.',
                   'Different A100 allocations and freely generated lengths limit isolated speed comparisons.',
                   'Probability-scored K2048 is an accuracy comparator, not an exact-output reference.',
                   '20 formal shards x3 questions, at most2 GPUs; formal work waits for the 3-question smoke gate.'
               ])
    write(ROOT/'config.json', cfg)
    write(ROOT/'smoke_work.json', [dict(cell=0, shard=0, num_shards=1)])
    write(ROOT/'formal_work.json', [dict(cell=0, shard=s, num_shards=20) for s in range(20)])
    probability = E/'results/hmmt_full_w1_optimized_k2048_protected_20261009'
    shutil.copyfile(probability/'grades.jsonl', ROOT/'reference_previous'/f"{cell['name']}_grades.jsonl")

    p = ROOT/'source/scripts/selector_grid.py'
    text = p.read_text()
    text = text.replace('Frozen2-cell HMMT W1 K1024 interval1000 grid', 'Frozen Full-KV raw-QK W1 K2048 run')
    text = text.replace("read(root/'microbenchmark.json')['selected_backends'][cell['storage']]", "'stable_sort'")
    text = text.replace("assert r['max_selected_tokens']<=max(P,1024)", "assert r['max_selected_tokens']<=max(P,cell['K'])")
    start = text.index("    refs={(r['run'],r['idx']):r['completion']")
    end = text.index("    byid={r['idx']:r for r in metrics}", start)
    text = text[:start]+text[end:]
    start = text.index('    for (run,idx),text in comps.items():')
    end = text.index('    return metrics,comps', start)
    text = text[:start]+text[end:]
    text = text.replace("name+'_previous_same_storage'", "name+'_probability_scored'")
    text = text.replace("'implementation_only_same_raw_QK_algorithm'", "'raw_QK_vs_probability_matched_K_storage_protection'")
    # Match the canonical regrading used by the current report. The external
    # 30-second worker bound handles timeouts without a cold 3-second false negative.
    text = text.replace('check_is_correct(pred,gold)', 'check_is_correct(pred,gold,timeout=False)')
    p.write_text(text)

    # Reuse established GPU regression suites, plus the actual K2048 full-history path.
    check = '''"""K2048 validation; backend is fixed, no performance tuning."""
import gc, hashlib, importlib.util, json, os, sys, unittest
from pathlib import Path
import torch
root=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(root/'source/ra_harness'))
from kvcompress.engine.full_w1_optimized_cache import FullW1OptimizedLayer
from kvcompress.engine.w1_qk_fast_cache import FastFullW1Layer
suite=unittest.TestSuite()
for name in ('test_w1_qk_fast','test_full_w1_qk','test_w1_interval1000'):
    spec=importlib.util.spec_from_file_location(name,root/'source/tests'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
    if name=='test_w1_qk_fast':make_layer=module.layer

class K2048Checks(unittest.TestCase):
    def test_full_history_masks_attention_and_protection(self):
        for n in (300,2048,2049,8192,32700):
            old=make_layer(FullW1OptimizedLayer,'cuda',n,budget=2048,hkv=8,dim=128)
            new=make_layer(FastFullW1Layer,'cuda',n,budget=2048,hkv=8,dim=128)
            new.rank_backend='stable_sort'
            before=torch.cuda.get_rng_state().clone()
            ids,counts=new.select_indices()
            self.assertTrue(torch.equal(ids,old.select_indices()[0]))
            self.assertEqual(counts,[min(n,2048)]*2)
            self.assertTrue(torch.all(ids[:,1:]>ids[:,:-1]).item())
            self.assertTrue(torch.equal(ids[:,:137],torch.arange(137,device='cuda')[None].expand(2,-1)))
            self.assertTrue(torch.equal(ids[:,-64:],torch.arange(n-64,n,device='cuda')[None].expand(2,-1)))
            k,v=old.k_cache[:,n-1:n].contiguous(),old.v_cache[:,n-1:n].contiguous()
            q=old.queries.transpose(1,2)
            self.assertTrue(torch.equal(old.attention(q,k,v,128**-.5),new.attention(q,k,v,128**-.5)))
            self.assertTrue(torch.equal(before,torch.cuda.get_rng_state()))
            self.assertEqual(new._lengths,[n+1]*2)
            del old,new
            gc.collect();torch.cuda.empty_cache()
suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(K2048Checks))
with (root/'correctness.log').open('x') as log:
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
proof=dict(status='passed' if result.wasSuccessful() and not result.skipped else 'failed',
 tests=result.testsRun,skipped=result.skipped,gpu=torch.cuda.get_device_name(),
 node=os.uname().nodename,job=os.environ.get('SLURM_JOB_ID'),K=2048,backend='stable_sort',
 log_sha256=hashlib.sha256((root/'correctness.log').read_bytes()).hexdigest())
(root/'correctness.json').write_text(json.dumps(proof,indent=2)+'\\n')
print(json.dumps(proof),flush=True)
assert proof['status']=='passed'
'''
    (ROOT/'source/scripts/benchmark_w1_qk_fast.py').write_text(check)
    for stage in ('correctness', 'smoke', 'gate', 'formal', 'collect'):
        p = ROOT/'source/scripts'/f'{stage}.sbatch'
        text = p.read_text().replace('hmmt-qk-fast-', 'hmmt-qk2048-')
        if stage == 'smoke': text = text.replace('--array=0-1%2', '--array=0-0%1')
        if stage == 'formal': text = text.replace('--array=0-39%2', '--array=0-19%2')
        p.write_text(text)
    write(ROOT/'provenance.json', dict(base=str(BASE), probability_reference=str(probability),
          score='raw BF16 current-query QK; stable chronological ties; no selector softmax',
          implementation='full_w1_raw_qk_fast_v2', backend='stable_sort',
          implementation_sha256=sha(ROOT/'source/ra_harness/kvcompress/engine/w1_qk_fast_cache.py'),
          input_sha256=sha(ROOT/'inputs/hmmt.jsonl')))
    (ROOT/'README.md').write_text('''# HMMT Full-KV W1 raw-QK v2: K2048

Authorized completion of the missing K2048 cell, using the same frozen v2 engine
and stable_sort backend as the completed K1024 Full-KV run. Qwen3-4B BF16,
60 canonical questions x2 sampled outputs, batch2, seed42, representative seed0,
temperature0.6, generation top-p0.95, total prompt+output limit32768.
Prompt+recent64 count inside K2048. All KV retained; attention selects K each step.
GPU correctness -> 3 complete smoke questions -> CPU validation/grading gate ->
20 formal shards (3 questions each, at most2 A10080GB GPUs) -> collection.
The independent collector requires120 distinct answers, full input SHA alignment,
analytic pair accounting, bounded grading, timing, memory and sparsity telemetry.
Historical probability K2048 is a labeled comparison, never an exact-output oracle.
''')
    assert sha(ROOT/'source/ra_harness/kvcompress/engine/w1_qk_fast_cache.py') == sha(BASE/'source/ra_harness/kvcompress/engine/w1_qk_fast_cache.py')
    for p in (ROOT/'source').rglob('*.py'): ast.parse(p.read_text(), filename=str(p))
    for stage in ('correctness','smoke','gate','formal','collect'):
        subprocess.run(['bash','-n',str(ROOT/'source/scripts'/f'{stage}.sbatch')], check=True)
    write(ROOT/'static_validation.json',dict(status='passed',K=2048,cells=1,
          questions=60,answers=120,formal_shards=20,max_concurrent_gpus=2,
          engine_identical_to_completed_K1024=True,backend='stable_sort'))
    files = [ROOT/p for p in ('config.json','smoke_work.json','formal_work.json',
                             'model_assets_sha256.json','inputs/hmmt.jsonl','data/hmmt/test.jsonl','provenance.json')]
    files += [p for folder in ('source','reference_previous') for p in (ROOT/folder).rglob('*') if p.is_file()]
    write(ROOT/'manifest.json', {str(p.relative_to(ROOT)):sha(p) for p in sorted(files)})
    print(json.dumps(dict(root=str(ROOT),static_validation='passed',manifest_files=len(files))), flush=True)


if __name__ == '__main__': main()
