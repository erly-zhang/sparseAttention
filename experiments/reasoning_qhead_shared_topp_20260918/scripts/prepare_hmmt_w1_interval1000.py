"""Freeze two HMMT K1024 raw-QK interval1000 methods and gated jobs."""
import ast
import argparse
import datetime
import hashlib
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

E = Path(__file__).resolve().parents[1]
BASE = E/'results/hmmt_w1_raw_qk_k1024_full_compact500_20261009'
ROOT = None


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, value):
    p.write_text(json.dumps(value, indent=2)+'\n')


def edit(p, old, new):
    text = p.read_text()
    assert old in text, (p, old)
    p.write_text(text.replace(old, new))


def main():
    global ROOT
    parser=argparse.ArgumentParser()
    parser.add_argument('--budget',type=int,choices=(1024,2048),default=1024)
    K=parser.parse_args().budget
    ROOT=E/f'results/hmmt_w1_raw_qk_k{K}_hybrid4096_compact1000_20261011'
    assert not ROOT.exists(), 'Never overwrite an existing experiment'
    ROOT.mkdir()
    for folder in ('source','inputs','data'):
        files=[Path(s) for s in subprocess.check_output(['rg','--files','--hidden','--no-ignore',str(BASE/folder)],text=True).splitlines()]
        files=[p for p in files if '__pycache__' not in p.parts and p.suffix!='.pyc']
        for parent in sorted({(ROOT/folder/p.relative_to(BASE/folder)).parent for p in files}):
            parent.mkdir(parents=True,exist_ok=True)
        def copy(p):
            with p.open('rb') as src,(ROOT/folder/p.relative_to(BASE/folder)).open('xb') as dst:
                shutil.copyfileobj(src,dst)
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(copy,files))
    for name in ('model_assets_sha256.json','prompt_protection_stats.json'):
        shutil.copyfile(BASE/name,ROOT/name)
    for folder in ('logs','reference_previous'):
        (ROOT/folder).mkdir()
    for rel in ('ra_harness/kvcompress/engine/w1_interval1000_cache.py',
                'scripts/run_w1_interval1000.py','tests/test_w1_interval1000.py'):
        shutil.copyfile(E/rel,ROOT/'source'/rel)
    cfg=json.loads((BASE/'config.json').read_text())
    cfg.update(created=datetime.datetime.now().astimezone().isoformat(),
               experiment=f'hmmt_w1_raw_qk_k{K}_hybrid4096_compact1000_v1',
               scope_authorization='User requested HMMT W1 attention K plus random4096 candidates updated every1000 steps and W1 compact500 changed to1000 on2026-10-11, both K1024 and K2048; prioritize K1024.',
               max_concurrent_gpus=2,compaction_interval=1000,
               compaction='After attention on absolute decode steps divisible by1000. Compact keeps K; hybrid keeps current attention K plus independent Uniform-TopK4096 outside core. No compaction when all candidates already fit.',
               caveats=[
                   '60 questions x2 sampled rollouts; accuracy is per rollout, not pass@2.',
                   'K1024 includes prompt and recent64. Raw BF16 QK selection; attention softmax is retained.',
                   'Each layer and step/event uses the historical representative identity and GQA mapping; all KV heads share positions, not values.',
                   'Hybrid selects attention K every step and updates physical candidates only every1000 steps. Its live candidate pool can grow to K+4096+1000 between events.',
                   'Compact1000 attends all surviving KV plus new KV and compresses to K after attention. Its attention can grow to K+1000.',
                   'Extra4096 means retained extra candidates, not4096 discarded tokens. Residuals exclude the attention core before random TopK.',
                   'Uniform float32 priorities use independent secondary random bits, then position for double ties; private RNG does not affect generation sampling.',
                   'Absolute RoPE positions are retained; physically evicted KV cannot return.',
                   'GPU-synchronized decode seconds include all selection, random generation, gather, compaction, attention and sampling. Batch2 seconds are per question batch, not per rollout.',
                   'Same A10080GB model/settings as historical baselines, but different nodes/dates; small timing differences need caution.'])
    cfg['cells']=[]
    cfg['caveats']=[sentence.replace('K1024',f'K{K}') for sentence in cfg['caveats']]
    cfg['random']='Extra4096: independent Uniform float32 priorities; private stable_hash(0x48595253,mask_seed42,canonical_sample_idx,layer,absolute_decode_step,original_rollout). Independent secondary random bits break finite precision ties, then position. Core masked before TopK; new stream differs from historical random masks.'
    for storage in ('compact','hybrid'):
        cell=dict(name=f'hmmt_k{K}_w1_raw_qk_{storage}1000_protected',task='hmmt',count=60,K=K,
                  shards=4,method=f'grid_{storage}1000_w1_topk_protected',storage=storage,
                  scorer='w1',selector='topk',protection='protected',window=1,
                  compaction_interval=1000,score_mode='qk',extra_budget=4096 if storage=='hybrid' else 0,
                  candidate_budget=K+4096 if storage=='hybrid' else K,
                  baseline_score_mode='qk' if K==1024 else 'probability')
        cfg['cells'].append(cell)
        old_name=f"hmmt_k1024_w1_raw_qk_{'compact' if storage=='compact' else 'full'}_protected"
        for artifact in ('summary.json','grades.jsonl'):
            source=BASE/'cells'/old_name/'formal'/artifact
            if K==2048:
                source=(E/'results/hmmt_compact_interval500_k1024_k2048_k4096_20261007/cells/hmmt_k2048_grid_compact500_w1_topk_protected/formal'/artifact) if storage=='compact' else E/'results/hmmt_full_w1_optimized_k2048_protected_20261009'/artifact
            shutil.copyfile(source,
                            ROOT/'reference_previous'/f"{cell['name']}_{artifact}")
    write(ROOT/'config.json',cfg)
    write(ROOT/'smoke_work.json',[dict(cell=i,shard=0,num_shards=1) for i in range(2)])
    write(ROOT/'formal_work.json',[dict(cell=i,shard=s,num_shards=4) for s in range(4) for i in range(2)])

    gen=ROOT/'source/ra_harness/kvcompress/harness/generation_utils.py'
    edit(gen,"        layers=current_cache.layers;first=layers[0]",
         """        layers=current_cache.layers;first=layers[0]
        metrics.update(compaction_events=sum(l.compaction_events for l in layers),
                       candidate_budget=getattr(first,'candidate_budget',first.token_budget),
                       extra_candidate_budget=getattr(first,'extra_budget',0))
        if first.storage=='hybrid':
            metrics.update(cache_semantics='bounded_candidates_temporary_attention_mask',
                           compaction_core_token_sum=sum(l.compaction_core_token_sum for l in layers),
                           compaction_residual_token_sum=sum(l.compaction_residual_token_sum for l in layers),
                           compaction_token_observations=sum(l.compaction_token_observations for l in layers))""")
    ev=ROOT/'source/ra_harness/kvcompress/harness/eval_hf.py'
    edit(ev,"compaction_phase='after_attention' if storage=='compact' else None", "compaction_phase='after_attention' if storage in ('compact','hybrid') else None")
    edit(ev,"compaction_interval=args.eviction_interval if storage=='compact' else None", "compaction_interval=args.eviction_interval if storage in ('compact','hybrid') else None")
    script=ROOT/'source/scripts/selector_grid.py'
    text=script.read_text().replace('run_w1_qk.py','run_w1_interval1000.py')
    text=text.replace("if cell['storage']=='compact' else '1'", "if cell['storage'] in ('compact','hybrid') else '1'")
    start=text.index("        assert r['w1_implementation']==")
    end=text.index("        ts=r['generated_tokens_per_rollout']",start)
    text=text[:start]+"""        assert r['w1_implementation']==('shared_w1_raw_qk_compact1000_v1' if cell['storage']=='compact' else 'w1_raw_qk_bounded_random4096_interval1000_v1')
        assert r['compaction_interval']==cell['compaction_interval']==r['eviction_interval']==1000
        assert r['compaction_phase']=='after_attention'
        assert r['cache_semantics']==('persistent_eviction' if cell['storage']=='compact' else 'bounded_candidates_temporary_attention_mask')
        assert r['candidate_budget']==cell['candidate_budget']
        assert r['extra_candidate_budget']==cell['extra_budget']
        assert r['max_capacity']==max(P,cell['candidate_budget'])+1000
        for row in r['final_cache_seqlens']:
            assert all(n<=cell['candidate_budget']+1000 for n in row)
        if cell['storage']=='hybrid':
            assert r['compaction_core_token_sum']==cell['K']*r['compaction_token_observations']
            assert r['compaction_residual_token_sum']==4096*r['compaction_token_observations']
"""+text[end:]
    start=text.index("        if cell['selector']=='topk':\n            total=0")
    end=text.index("        assert all(c==P+max(ts)-1",start)
    text=text[:start]+"""        total=0;events=0
        for t in ts:
            cached=P
            for step in range(1,t):
                cached+=1
                total+=min(cell['K'],cached) if cell['storage']=='hybrid' else cached
                if step%1000==0 and cached>cell['candidate_budget']:
                    cached=cell['candidate_budget'];events+=1
        assert r['decode_selected_pairs']==factor*total,(r['idx'],r['decode_selected_pairs'],factor*total)
        # Per-layer event counters count one batched event, rather than one per rollout.
        longest=max(ts);cached=P;batch_events=0
        for step in range(1,longest):
            cached+=1
            if step%1000==0 and cached>cell['candidate_budget']:
                cached=cell['candidate_budget'];batch_events+=1
        assert r['compaction_events']==layers*batch_events
        expected_selections=(longest-1) if cell['storage']=='hybrid' else batch_events
        assert r['selection_events']==layers*expected_selections
        if cell['storage']=='hybrid':assert r['compaction_token_observations']==layers*events
        expected_attention_max=min(cell['K'],P+longest-1) if cell['storage']=='hybrid' else max(P,cell['K']+1000)
        assert r['max_selected_tokens']<=max(P,expected_attention_max)
        for row in r['final_cache_seqlens']:assert all(n==cached for n in row)
"""+text[end:]
    text=text.replace("'passed/interval500_analytic' if cell['storage']=='compact' else 'passed/full_kv_fixed_budget_analytic'", "'passed/interval1000_analytic' if cell['storage']=='compact' else 'passed/bounded_hybrid_fixed_attention_analytic'")
    text=text.replace("root/'reference_probability'", "root/'reference_previous'")
    text=text.replace("name+'_probability'", "name+'_previous_compact500' if cell['storage']=='compact' else name+'_previous_full'")
    text=text.replace("'raw_QK_vs_probability_matched_K_storage_protection'", "'compact1000_vs_compact500_raw_QK' if cell['storage']=='compact' else 'hybrid_random4096_interval1000_vs_full_raw_QK'")
    if K==2048:
        text=text.replace("'compact1000_vs_compact500_raw_QK' if cell['storage']=='compact' else 'hybrid_random4096_interval1000_vs_full_raw_QK'", "'interval1000_raw_QK_vs_historical_probability_different_score_mode'")
    text=text.replace('Frozen2-cell HMMT raw-QK W1 K1024 grid','Frozen2-cell HMMT W1 K1024 interval1000 grid')
    script.write_text(text)
    checker=ROOT/'source/scripts/check_budgets.py'
    edit(checker,"('test_full_w1_qk','test_w1_qk_grid')", "('test_w1_interval1000',)")
    edit(checker,"Raw QK: W1 full/compact500, real K1024 and GQA geometry, no selector softmax, signed stable ties, protection, short history, absolute positions, two compactions, independent FP32 attention reference and batch shrink; unchanged probability default",
         "W1 raw-QK compact1000/hybrid1000; core K1024 plus random4096; private Uniform priorities and marginal-frequency sanity; exclusion, budgets, GQA, protection, absolute positions, two compactions, batch shrink, analytic pairs and independent FP32 attention oracle; historical modes unchanged")
    for stage in ('correctness','smoke','gate','formal','collect'):
        p=ROOT/'source/scripts'/f'{stage}.sbatch'
        text=p.read_text().replace('hmmt-qk-',f'hmmt-k{K}-i1000-')
        if stage=='smoke':text=text.replace('--array=0-1%1','--array=0-1%2')
        if stage=='formal':text=text.replace('--time=16:00:00','--time=20:00:00')
        p.write_text(text)
    write(ROOT/'provenance.json',dict(baseline_root=str(BASE),
          input_reference_sha256=sha(ROOT/'inputs/hmmt.jsonl'),
          data_sha256=sha(ROOT/'data/hmmt/test.jsonl'),
          implementation_sources={rel:sha(E/rel) for rel in ('ra_harness/kvcompress/engine/w1_interval1000_cache.py','scripts/run_w1_interval1000.py','tests/test_w1_interval1000.py')}))
    readme='''# HMMT W1 raw-QK: compact1000 and hybrid random4096

Two protected K1024 groups; same canonical60 HMMT inputs, Qwen3-4B, BF16, batch2,
two sampled rollouts/question, seed42, representative seed0, generation temperature0.6,
top-p0.95, total prompt+output limit32768. Prompt and recent64 count inside K.

Compact1000: dense prefill, attend all surviving KV plus new token each step;
after attention every1000 decode forwards retain W1-selected K and evict the rest.
Live cache and attention length are bounded by K+1000 after stabilization.

Hybrid1000: dense prefill, every decode forward score all live candidates with
current representative raw QK, select protected W1 TopK, and attend only this K.
After attention every1000 forwards, if cache exceeds K+4096, retain the just-used
K plus4096 Uniform-TopK random positions from its complement. Retain all if fit.
New tokens accumulate between events; physical capacity is K+4096+1000=6120.
Each layer/rollout owns its candidate positions; all KV heads share those positions.
Residual priorities use private seed42/sample/layer/absolute-step/original-rollout,
independent float32 Uniform primary scores and random secondary tie breakers.
This is a new random stream and is not intended to reproduce older random masks.

Correctness (independent masks and FP32 attention, two physical events, RNG isolation,
sampling sanity and batch shrink) must pass before full generation smoke on inputs0,1,2.
Both six-output smokes must pass technical input and analytic pair checks before
eight formal shards may run. At most two scheduler-assigned A10080GB GPUs.
All methods use independent output folders; historical source/results are preserved.
The collector requires60 unique inputs/120 unique answers per cell and passed/full
input alignment. Decode timing includes random generation, scoring, sorting, gather,
compaction, attention and generation bookkeeping, excluding load/warmup/serialization.
'''
    readme=readme.replace('K1024',f'K{K}').replace('=6120',f'={K+5096}')
    if K==2048:
        cfg['caveats'].append('Historical K2048 comparison baselines use probability scoring; this run uses raw QK. Differences therefore do not isolate the compression interval or candidate-cache change.')
        write(ROOT/'config.json',cfg)
        readme+='\nHistorical K2048 references use probability scoring, so comparisons also change score arithmetic.\n'
    (ROOT/'README.md').write_text(readme)
    for p in (ROOT/'source').rglob('*.py'):
        ast.parse(p.read_text(),filename=str(p))
    write(ROOT/'static_validation.json',dict(status='passed',cells=2,formal_shards=8,
          smoke_ids=[0,1,2],python_ast='passed',input_reference_sha256=sha(ROOT/'inputs/hmmt.jsonl')))
    files=[ROOT/p for p in ('config.json','smoke_work.json','formal_work.json','model_assets_sha256.json','inputs/hmmt.jsonl','data/hmmt/test.jsonl','provenance.json')]
    files += [p for folder in ('source','reference_previous') for p in (ROOT/folder).rglob('*') if p.is_file()]
    write(ROOT/'manifest.json',{str(p.relative_to(ROOT)):sha(p) for p in sorted(files)})
    print(json.dumps(dict(root=str(ROOT),static_validation='passed',cells=2,formal_shards=8)),flush=True)


if __name__=='__main__':main()
