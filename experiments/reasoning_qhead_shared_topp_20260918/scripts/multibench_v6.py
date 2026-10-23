"""Frozen multi-dataset comparison: hybrid W1 v6 vs official random_pp vs Dense.

Fairness: one frozen source tree, one data directory, the same chat template,
tokenizer, generation settings (seed42, batch2, temperature0.6, generation
top-p0.95, 32768 total tokens, 2-token untimed warmup) and timing scope. Every
generated input is checked at runtime against a per-question reference
(input_tokens + SHA-256 of the input IDs) via RA_GRID_REFERENCE. Methods differ
only in the attention/cache algorithm. Prefill is dense for every method.

Actions:  preflight | check | infer WAVE INDEX | collect WAVE
"""
import concurrent.futures, datetime, hashlib, importlib.metadata, importlib.util, json, math
import multiprocessing, os, platform, re, shutil, statistics, subprocess, sys, time, unittest
from pathlib import Path

ROOT = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else None
HYBRID_MODE = 'grid_hybrid1024_w1_topk_protected'
INTERVAL = 1024
EXTRA = 4096


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(p): return json.loads(Path(p).read_text())
def rows(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def write(p, x): Path(p).write_text(json.dumps(x, indent=2)+'\n')


def verify():
    for rel, h in read(ROOT/'manifest.json').items():
        assert sha(ROOT/rel) == h, rel
    return read(ROOT/'config.json')


def env(task):
    e = os.environ.copy()
    e.update(PYTHONPATH=str(ROOT/'source/ra_harness'), RA_DATA_DIR=str(ROOT/'data'),
             RA_GRID_REFERENCE=str(ROOT/'inputs'/f'{task}.jsonl'), QHEAD_DEBUG='0',
             PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1',
             TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
             TOKENIZERS_PARALLELISM='false', RA_W1_QK_FAST_BACKEND='packed_qk_topk')
    for k in ('QHEAD_LOG', 'KEEPLOG', 'KEEPLOG_POS', 'FORCE_KEEP_RANGE', 'RA_W1_TOKEN_OUTPUT'):
        e.pop(k, None)
    return e


# ---------------------------------------------------------------- inputs
def preflight():
    """Rebuild every prompt with the frozen harness; compare to the references."""
    cfg = verify()
    sys.path[:0] = [str(ROOT/'source/ra_harness/kvcompress/harness'), str(ROOT/'source/ra_harness')]
    spec = importlib.util.spec_from_file_location('frozen_eval_hf', ROOT/'source/ra_harness/kvcompress/harness/eval_hf.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg['model'])
    report = {}
    for task in cfg['tasks']:
        name = task['name']
        examples = m.load_data(name, 'test', str(ROOT/'data'))
        refs = {r['idx']: r for r in rows(ROOT/'inputs'/f'{name}.jsonl')}
        assert len(examples) == task['count'] == len(refs) and set(refs) == set(range(task['count'])), name
        lengths = []
        for ex in examples:
            q = m.parse_question(ex, name)
            _, _, fmt = m.get_three_prompt('qwen-instruct', name)
            prompt = fmt.format(question=q)
            if name in ['aime24', 'aime25', 'aime26', 'hmmt', 'math', 'olympiadbench']:
                messages = [{'role': 'user', 'content': prompt+'\nPlease reason step by step, and put your final answer within \\boxed{}.'}]
            else:
                messages = [{'role': 'user', 'content': prompt}]
            ids = m.get_conversation_prompt_by_messages(tokenizer=tok, messages=messages)['input_ids'][0]
            r = refs[int(ex['idx'])]
            digest = hashlib.sha256(ids.numpy().astype('<i8').tobytes()).hexdigest()
            assert ids.shape[0] == r['input_tokens'] and digest == r['input_ids_sha256'], (name, ex['idx'])
            lengths.append(int(ids.shape[0]))
        report[name] = dict(questions=len(lengths), input_alignment='passed/full', max_prompt=max(lengths),
                            median_prompt=statistics.median(lengths),
                            long_prompts_k1024=sum(x+64 > 1024 for x in lengths),
                            long_prompts_k2048=sum(x+64 > 2048 for x in lengths))
    write(ROOT/'preflight.json', dict(status='passed', tasks=report, tokenizer=cfg['model']))
    print(json.dumps(report), flush=True)


# ---------------------------------------------------------------- correctness
def check():
    verify()
    import torch
    sys.path[:0] = [str(ROOT/'source/ra_harness'), str(ROOT/'source/tests')]
    suite = unittest.TestSuite()
    for name in ('test_w1_hybrid_v6', 'test_w1_hybrid_variants', 'test_w1_hybrid_gpu_select',
                 'test_official_random_pp', 'test_multibench_dense_metrics'):
        spec = importlib.util.spec_from_file_location(name, ROOT/'source/tests'/f'{name}.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(mod))
    with (ROOT/'correctness.log').open('x') as f:
        result = unittest.TextTestRunner(stream=f, verbosity=2).run(suite)
    proof = dict(status='passed' if result.wasSuccessful() and not result.skipped else 'failed',
                 tests=result.testsRun, skipped=[str(s) for s in result.skipped],
                 gpu=torch.cuda.get_device_name(), node=platform.node(), job=os.environ.get('SLURM_JOB_ID'),
                 log_sha256=sha(ROOT/'correctness.log'))
    write(ROOT/'correctness.json', proof)
    print(json.dumps(proof), flush=True)
    assert proof['status'] == 'passed'


# ---------------------------------------------------------------- inference
def cell_dir(cell):
    return ROOT/'cells'/cell['name']


def command(cfg, cell, task, work, out):
    base = ['--model_name_or_path', cfg['model'], '--data_name', task['name'], '--split', 'test',
            '--prompt_type', 'qwen-instruct', '--surround_with_messages',
            '--attention_implementation', 'flash_attention_2', '--seed', '42', '--batch_size', '2',
            '--run_id', '0', '--max_tokens', str(cfg['max_tokens']), '--warmup_new_tokens', '2',
            '--use_batch_exist', '--output_dir', str(out),
            '--num_shards', str(work['num_shards']), '--shard_id', str(work['shard'])]
    eval_hf = str(ROOT/'source/ra_harness/kvcompress/harness/eval_hf.py')
    if cell['method'] == 'hybrid':
        return [sys.executable, '-u', str(ROOT/'source/scripts/run_w1_hybrid_v6.py'), *base,
                '--sparsity_method', 'eviction', '--eviction_mode', HYBRID_MODE, '--token_budget', str(cell['K']),
                '--residual_length', '0', '--top_p', '.95', '--query_window', '1', '--mask_seed', '42',
                '--rep_seed', '0', '--eviction_interval', str(INTERVAL)]
    if cell['method'] == 'random_pp':
        return [sys.executable, '-u', eval_hf, *base, '--sparsity_method', 'eviction',
                '--eviction_mode', 'official_random_pp', '--token_budget', str(cell['K']),
                '--residual_length', '64', '--query_window', '1', '--eviction_interval', '1',
                '--top_p', '.95', '--mask_seed', '42', '--rep_seed', '0']
    assert cell['method'] == 'dense'
    return [sys.executable, '-u', eval_hf, *base, '--sparsity_method', 'dense']


def infer(wave, index):
    cfg = verify()
    proof = read(ROOT/'correctness.json')
    assert proof['status'] == 'passed' and sha(ROOT/'correctness.log') == proof['log_sha256']
    assert read(ROOT/'preflight.json')['status'] == 'passed'
    work = read(ROOT/f'work_{wave}.json')[index]
    cell = cfg['cells'][work['cell']]
    task = next(t for t in cfg['tasks'] if t['name'] == cell['task'])
    out = cell_dir(cell)/f"shard_{work['shard']}"
    if out.exists():
        if (out/'validation.json').exists():
            print('already validated', out, flush=True)
            return
        out.rename(out.with_name(out.name+f'.failed.{int(time.time())}'))
    out.mkdir(parents=True)
    for rel, h in read(ROOT/'model_assets_sha256.json').items():
        assert sha(Path(cfg['model'])/rel) == h, rel
    import torch
    gpu = torch.cuda.get_device_properties(0)
    assert torch.cuda.device_count() == 1 and 'A100' in gpu.name and gpu.total_memory > 75*1024**3
    packages = {k: importlib.metadata.version(k) for k in cfg['packages']}
    assert packages == cfg['packages'], packages
    write(out/'hardware.json', dict(node=platform.node(), gpu=gpu.name, memory=gpu.total_memory,
                                    packages=packages, job=os.environ.get('SLURM_JOB_ID'),
                                    array=os.environ.get('SLURM_ARRAY_JOB_ID'), task_id=index))
    args = command(cfg, cell, task, work, out)
    write(out/'command.json', args)
    with (out/'runner.log').open('x') as f:
        proc = subprocess.run(args, env=env(task['name']), stdout=f, stderr=subprocess.STDOUT)
    write(out/'exit.json', dict(exit_code=proc.returncode))
    if proc.returncode:
        raise RuntimeError(f'runner exited {proc.returncode}')
    validate(cfg, cell, task, work)
    write(out/'validation.json', dict(status='passed', cell=cell['name'], shard=work['shard']))


# ---------------------------------------------------------------- validation
def validate(cfg, cell, task, work):
    d = cell_dir(cell)/f"shard_{work['shard']}"
    assert read(d/'exit.json')['exit_code'] == 0, d
    refs = {r['idx']: r for r in rows(ROOT/'inputs'/f"{task['name']}.jsonl")}
    ids = set(range(work['shard'], task['count'], work['num_shards']))
    metrics = rows(d/'cache_metrics.jsonl')
    assert len(metrics) == len(ids) and {r['idx'] for r in metrics} == ids, d
    g = cfg['model_geometry']
    layers, hq = g['num_hidden_layers'], g['num_attention_heads']
    factor = layers*hq
    limit = cfg['max_tokens']
    K = cell['K']
    for r in metrics:
        P = r['input_tokens']
        ref = refs[r['idx']]
        assert P == ref['input_tokens'] and r['input_ids_sha256'] == ref['input_ids_sha256'], (d, r['idx'])
        ts = r['generated_tokens_per_rollout']
        assert len(ts) == len(r['eos_reached']) == 2
        for t, eos in zip(ts, r['eos_reached']):
            assert 1 <= t <= limit-P
            if not eos:
                assert t == limit-P
        assert r['decode_forward_calls'] == max(ts)-1 and r['decode_query_tokens'] == sum(ts)-2
        pf = 2*factor*P*(P+1)//2
        causal = factor*sum((P+t-1)*(P+t)//2 for t in ts)
        assert r['prefill_selected_pairs'] == r['prefill_causal_pairs'] == pf
        assert r['causal_pairs'] == causal
        assert r['prefill_seconds'] > 0 and r['decode_seconds'] >= 0
        longest = max(ts)
        if cell['method'] == 'dense':
            assert r['cache_semantics'] == 'standard_dense_DynamicCache' and r['selection_events'] == 0
            assert r['selected_pairs'] == causal and r['final_cache_length'] == P+longest-1
        elif cell['method'] == 'hybrid':
            assert r['grid_method'] == HYBRID_MODE and r['grid_storage'] == 'hybrid'
            assert 'v6_single_launch' in r['w1_implementation'] and r['w1_score_mode'] == 'qk'
            assert r['compaction_interval'] == r['eviction_interval'] == INTERVAL
            assert r['cache_semantics'] == 'bounded_candidates_temporary_attention_mask'
            assert r['candidate_budget'] == K+EXTRA and r['max_capacity'] == max(P, K+EXTRA)+INTERVAL
            total = 0
            for t in ts:
                cached = P
                for step in range(1, t):
                    cached += 1
                    total += min(K, cached)
                    if step % INTERVAL == 0 and cached > K+EXTRA:
                        cached = K+EXTRA
            assert r['decode_selected_pairs'] == factor*total, (d, r['idx'])
            assert r['decode_selected_pairs'] == hq*r['attention_token_sum']
            cached, events = P, 0
            for step in range(1, longest):
                cached += 1
                if step % INTERVAL == 0 and cached > K+EXTRA:
                    cached, events = K+EXTRA, events+1
            assert r['compaction_events'] == layers*events
            assert r['selection_events'] == layers*(longest-1)
            assert r['max_selected_tokens'] <= max(P, K)
            assert all(all(n == cached for n in row) for row in r['final_cache_seqlens'])
            assert all(c == P+longest-1 for c in r['final_cumulative_length'])
        else:
            assert r['selection_strategy'] == 'official_random_pp' and r['token_budget'] == K
            assert r['residual_length'] == 64 and r['random_scope'] == 'independent_per_kv_head'
            selected = 0
            for t in ts:
                stored = P
                for step in range(1, t):
                    stored += 1
                    selected += stored
                    if stored > K and stored % 64 == 0:
                        stored = K
            assert r['selected_pairs'] == pf+factor*selected, (d, r['idx'])
            stored, events = P, 0
            for step in range(1, longest):
                stored += 1
                if stored > K and stored % 64 == 0:
                    stored, events = K, events+1
            assert r['selection_events'] == layers*events
            assert all(all(n == stored for n in row) for row in r['final_cache_seqlens'])
            assert r['max_capacity'] == max(P, K)+64 and r['max_selected_tokens'] <= max(P, K+64)
    comps = {}
    for run in (0, 1):
        files = list((d/f'run_{run}').glob('completions*.jsonl'))
        assert len(files) == 1, d
        rr = rows(files[0])
        assert len(rr) == len(ids) and {r['idx'] for r in rr} == ids
        for r in rr:
            prefix = refs[r['idx']]['prompt_text']
            assert r['completion'].startswith(prefix), (d, r['idx'])
            comps[run, r['idx']] = r['completion'][len(prefix):]
    return metrics, comps


# ---------------------------------------------------------------- grading
def _math_child(conn, text, gold):
    try:
        from Utils.parser import extract_answer
        from Utils.grader import check_is_correct
        pred = extract_answer(text)
        conn.send(dict(prediction=pred, correct=bool(check_is_correct(pred, gold, timeout=False))))
    except BaseException as e:
        conn.send(dict(error=repr(e)))
    finally:
        conn.close()


def grade_math(text, gold):
    ctx = multiprocessing.get_context('fork')
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_math_child, args=(child, text, gold))
    proc.start()
    child.close()
    try:
        if not parent.poll(30):
            raise TimeoutError('math grader exceeded 30 seconds')
        result = parent.recv()
        if 'error' in result:
            raise RuntimeError(result['error'])
        return result['prediction'], int(result['correct'])
    finally:
        parent.close()
        proc.join(1)
        if proc.is_alive():
            proc.kill()
            proc.join()


def gpqa_answer(text):
    matches = list(re.finditer(r'\\boxed\s*\{', text))
    if not matches:
        return ''
    start = matches[-1].end()
    depth, end = 1, start
    while end < len(text) and depth:
        if text[end] == '{':
            depth += 1
        elif text[end] == '}':
            depth -= 1
        end += 1
    if depth:
        return ''
    value = text[start:end-1].strip()
    for _ in range(3):
        wrapped = re.fullmatch(r'\\(?:text|mathrm|mathbf|textbf)\s*\{\s*([\s\S]*)\s*\}', value)
        if wrapped:
            value = wrapped[1].strip()
    value = value.strip(' $().')
    return value if re.fullmatch('[A-D]', value) else ''


def grade_lcb(item):
    root, idx, run, text = item
    p = Path(root)/'source/ra_harness/kvcompress/eval/grader_lcb.py'
    spec = importlib.util.spec_from_file_location('mb_lcb_grader', p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    data = Path(root)/'data/livecodebench'
    store = m.ProblemStore(str(data/'test.jsonl'), str(data/'livecodebench_tests'), cache_mb=0)
    problem = store.get(idx)
    assert problem is not None and len(problem[1]) == len(problem[2]) > 0, ('ungradeable', idx)
    result = m.judge_completion(text, problem, timeout=20, mem_mb=2048, py=sys.executable, max_tests=0)
    assert result in (0, 1), ('ungradeable result', idx)
    return dict(idx=idx, run=run, prediction='', correct=int(result))


def grade(task, comps, refs):
    name = task['name']
    if name == 'livecodebench':
        items = [(str(ROOT), idx, run, text) for (run, idx), text in sorted(comps.items())]
        with concurrent.futures.ProcessPoolExecutor(max_workers=int(os.environ.get('MB_LCB_WORKERS', '16')),
                                                    mp_context=multiprocessing.get_context('spawn')) as pool:
            return list(pool.map(grade_lcb, items, chunksize=1))
    grades = []
    for (run, idx), text in sorted(comps.items()):
        if name == 'gpqa':
            pred = gpqa_answer(text)
            grades.append(dict(idx=idx, run=run, prediction=pred, correct=int(pred == refs[idx]['gold'])))
        else:
            pred, correct = grade_math(text, refs[idx]['gold'])
            grades.append(dict(idx=idx, run=run, prediction=pred, correct=correct))
    return grades


def summarize_cell(cfg, cell):
    task = next(t for t in cfg['tasks'] if t['name'] == cell['task'])
    work = [w for w in read(ROOT/f"work_{cell['wave']}.json") if cfg['cells'][w['cell']]['name'] == cell['name']]
    metrics, comps, hardware = [], {}, []
    for w in work:
        mm, cc = validate(cfg, cell, task, w)
        metrics += mm
        assert not set(comps) & set(cc)
        comps.update(cc)
        hardware.append(read(cell_dir(cell)/f"shard_{w['shard']}"/'hardware.json')['node'])
    refs = {r['idx']: r for r in rows(ROOT/'inputs'/f"{task['name']}.jsonl")}
    assert {m['idx'] for m in metrics} == set(refs) and len(comps) == 2*len(refs), cell['name']
    if task['name'] == 'livecodebench':
        for rel, h in read(ROOT/'data/lcb_test_assets_sha256.json').items():
            assert sha(ROOT/'data/livecodebench/livecodebench_tests'/rel) == h, rel
    grades = grade(task, comps, refs)
    byid = {m['idx']: m for m in metrics}
    for g in grades:
        g.update(generated_tokens=byid[g['idx']]['generated_tokens_per_rollout'][g['run']],
                 eos_reached=byid[g['idx']]['eos_reached'][g['run']])
    decode = sum(m['decode_seconds'] for m in metrics)
    forwards = sum(m['decode_forward_calls'] for m in metrics)
    selected = sum(m['selected_pairs'] for m in metrics)
    causal = sum(m['causal_pairs'] for m in metrics)
    tokens = [g['generated_tokens'] for g in grades]
    summary = dict(cell=cell, questions=len(refs), answers=len(grades),
                   correct=sum(g['correct'] for g in grades),
                   accuracy=statistics.mean(g['correct'] for g in grades),
                   runs=[statistics.mean(g['correct'] for g in grades if g['run'] == r) for r in (0, 1)],
                   decode_seconds_mean=decode/len(metrics),
                   decode_seconds_median=statistics.median(m['decode_seconds'] for m in metrics),
                   prefill_seconds_mean=statistics.mean(m['prefill_seconds'] for m in metrics),
                   decode_ms_per_step=1000*decode/forwards,
                   throughput_tokens_per_second=sum(m['decode_query_tokens'] for m in metrics)/decode,
                   generated_tokens_mean=statistics.mean(tokens),
                   generated_tokens_median=statistics.median(tokens),
                   length_limited=sum(not g['eos_reached'] for g in grades),
                   selected_pairs=selected, causal_pairs=causal, global_pair_sparsity=1-selected/causal,
                   peak_memory_gb=max(m['peak_memory_allocated'] for m in metrics)/1e9,
                   nodes=sorted(set(hardware)), input_alignment='passed/full',
                   pair_accounting='passed/analytic')
    out = cell_dir(cell)
    (out/'grades.jsonl').write_text(''.join(json.dumps(g)+'\n' for g in grades))
    write(out/'summary.json', summary)
    return summary


def paired(a, b):
    ga = {(g['run'], g['idx']): g['correct'] for g in rows(cell_dir(a)/'grades.jsonl')}
    gb = {(g['run'], g['idx']): g['correct'] for g in rows(cell_dir(b)/'grades.jsonl')}
    assert set(ga) == set(gb)
    return dict(only_first=sum(ga[k] > gb[k] for k in ga), only_second=sum(ga[k] < gb[k] for k in ga))


def collect(wave):
    cfg = verify()
    done = {}
    for cell in cfg['cells']:
        if cell['wave'] == wave:
            done[cell['name']] = summarize_cell(cfg, cell)
            print(json.dumps({k: done[cell['name']][k] for k in ('accuracy', 'decode_ms_per_step', 'throughput_tokens_per_second')} | {'cell': cell['name']}), flush=True)
    lines = ['| Task | Method | K | Accuracy | ms/step | Throughput tok/s | Decode s/question | Output tokens | Length-limited | Sparsity | Peak GB | vs Dense: only method / only Dense |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for task in cfg['tasks']:
        dense = next(c for c in cfg['cells'] if c['method'] == 'dense' and c['task'] == task['name'])
        for cell in cfg['cells']:
            if cell['task'] != task['name'] or not (cell_dir(cell)/'summary.json').exists():
                continue
            s = read(cell_dir(cell)/'summary.json')
            vs = '' if cell['method'] == 'dense' or not (cell_dir(dense)/'grades.jsonl').exists() else '{only_first} / {only_second}'.format(**paired(cell, dense))
            lines.append(f"| {task['name']} | {cell['method']} | {cell['K'] or '—'} | {100*s['accuracy']:.2f}% | {s['decode_ms_per_step']:.1f} | "
                         f"{s['throughput_tokens_per_second']:.2f} | {s['decode_seconds_mean']:,.1f} | {s['generated_tokens_mean']:,.0f} | "
                         f"{s['length_limited']} | {100*s['global_pair_sparsity']:.1f}% | {s['peak_memory_gb']:.1f} | {vs} |")
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    write(ROOT/f'collect_{wave}.json', dict(status='completed', cells=sorted(done)))
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    action = sys.argv[1]
    if action == 'preflight':
        preflight()
    elif action == 'check':
        check()
    elif action == 'infer':
        infer(sys.argv[3], int(sys.argv[4]))
    elif action == 'collect':
        collect(sys.argv[3])
    else:
        raise ValueError(action)
