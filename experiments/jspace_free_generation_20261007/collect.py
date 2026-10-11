"""Collect the 15 own-generation diagnostics; no GPU or inference required."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from free_generate import ARMS, CAP, FROZEN, K, LABELS, lcp, ids_sha, repeated_fraction, periodic_suffix
sys.path.insert(0,str(FROZEN/'source/ra_harness/kvcompress/harness'))
from Utils.grader import check_is_correct
from transformers import AutoTokenizer


def complete_boxes(text):
    """Only balanced boxed expressions; an unfinished capped box is not final."""
    boxes = []
    offset = 0
    marker = '\\boxed{'
    while True:
        start = text.find(marker,offset)
        if start < 0:break
        i = start+len(marker)
        depth = 1
        end = i
        while end < len(text) and depth:
            if text[end]=='{':depth+=1
            elif text[end]=='}':depth-=1
            end += 1
        if depth==0:boxes.append(text[i:end-1])
        offset = end
    return boxes


def independent_pairs(t):
    p = t['input_tokens']
    T = t['generated_tokens']
    causal = p*(p+1)//2+T*p+T*(T+1)//2
    selected = p*(p+1)//2
    # Original positions survive compaction; attention precedes every eviction.
    surviving = p
    for s in range(1,T+1):
        surviving += 1
        selected += min(K,p+s) if t['arm'].startswith('full') else surviving
        if t['arm'].startswith('compact') and s%64==0 and surviving>K:
            surviving = K
    factor = 36*32
    assert causal*factor == t['causal_pairs_including_final_forward']
    assert selected*factor == t['selected_pairs_including_final_forward']
    assert abs(1-selected/causal-t['global_pair_sparsity_including_final_forward']) < 1e-12


def main():
    tok = AutoTokenizer.from_pretrained('/scratch/yl258/jz648/Qwen3-4B',local_files_only=True)
    model_config = json.loads(Path('/scratch/yl258/jz648/Qwen3-4B/config.json').read_text())
    full_vocab_size = model_config['vocab_size']
    assert model_config['num_hidden_layers']==36 and model_config['num_attention_heads']==32
    results, pairs, source_records = [], [], []
    notes = ['HMMT own-generation follow-up: Dense, Random and W1',
             'Three original inputs, K=2048, prompt+recent64 protection, compact interval64.',
             'BF16, batch1, seed42, temperature0.6, top-p0.95, max4096 generated tokens.',
             'Every method samples its own next token. No Dense continuation is forced.',
             'This is a15-rollout diagnostic, not a formal accuracy or timing benchmark.',
             'After text divergence, J-lens differences mix attention effects and differing input histories.',
             'A final unsampled forward is included in pair totals and residual checkpoints.', '']
    for idx in [0,1,2]:
        out = HERE/'captures'/f'sample_{idx}'
        assert json.loads((out/'complete.json').read_text())['status']=='completed'
        validation = json.loads((out/'validation.json').read_text())
        assert validation['status']=='passed'
        paths = {a:out/f'{a}.json' for a in ARMS}
        ts = {a:json.loads(p.read_text()) for a,p in paths.items()}
        assert len({t['input_ids_sha256'] for t in ts.values()})==1
        assert len({t['gpu'] for t in ts.values()})==1
        assert len({t['node'] for t in ts.values()})==1
        readouts = [json.loads(x) for x in (out/'readouts.jsonl').read_text().splitlines()]
        keys = [(r['arm'],r['step'],r['layer']) for r in readouts]
        assert len(keys)==len(set(keys))
        notes.extend([f'Input {idx}: gold {ts["dense"]["gold"]}; prompt {ts["dense"]["input_tokens"]} tokens',
                      f'GPU: {ts["dense"]["gpu"]}; node: {ts["dense"]["node"]}',
                      'Prompt:',tok.decode(ts['dense']['prompt_ids']), ''])
        for arm,t in ts.items():
            assert t['teacher_forced'] is False and t['global_rng_unchanged']
            assert len(t['continuation_ids'])==t['generated_tokens']<=CAP
            assert t['continuation_ids_sha256']==ids_sha(t['continuation_ids'])
            assert t['prefix_sha256']==ids_sha(t['prompt_ids']+t['continuation_ids'])
            assert t['input_ids_sha256']==ids_sha(t['prompt_ids'])
            assert tok.decode(t['continuation_ids'],clean_up_tokenization_spaces=False)==t['continuation_text']
            independent_pairs(t)
            assert repeated_fraction(t['continuation_ids'])==t['duplicate_16gram_fraction']
            assert periodic_suffix(t['continuation_ids'])==t['exact_periodic_suffix']
            text = t['continuation_text']
            allboxes = complete_boxes(text)
            final_text = text.split('</think>',1)[1] if '</think>' in text else ''
            final_boxes = complete_boxes(final_text)
            candidate = final_boxes[-1] if final_boxes else None
            correct = bool(check_is_correct(candidate,t['gold'])) if candidate is not None else None
            common = lcp(ts['dense']['continuation_ids'],t['continuation_ids'])
            first = common+1 if common < min(ts['dense']['generated_tokens'],t['generated_tokens']) else None
            first_full = K-t['input_tokens']+1
            visible = first_full if arm.startswith('full') else ((first_full+63)//64)*64+1
            rs = [r for r in readouts if r['arm']==arm]
            active = [r for r in rs if r['layer']<35 and r['reasoning'] and r['step']>=visible and r['js_nats_vs_dense_at_same_step'] is not None]
            for r in rs:
                assert r['same_token_prefix_as_dense']==(r['step']<=common)
                for item in r['top8']:
                    assert 1<=item['rank_lower']<=item['rank_upper']<=full_vocab_size
                    assert 0<=item['probability']<=1
            row = dict(idx=idx,arm=arm,label=LABELS[arm],prompt_tokens=t['input_tokens'],
                       generated_tokens=t['generated_tokens'],stop_reason=t['stop_reason'],
                       think_closed=t['think_closed'],think_close_token=t['think_close_token'],
                       final_boxed_answer=candidate,correct_final_boxed_answer=correct,
                       last_box_anywhere=allboxes[-1] if allboxes else None,
                       gold=t['gold'],common_prefix_tokens_vs_dense=common,
                       first_different_generated_token=first,
                       duplicate_16gram_fraction=t['duplicate_16gram_fraction'],
                       tail1024_duplicate_16gram_fraction=t['tail1024_duplicate_16gram_fraction'],
                       exact_periodic_suffix_tokens=t['exact_periodic_suffix']['tokens'],
                       selected_pairs_including_final_forward=t['selected_pairs_including_final_forward'],
                       causal_pairs_including_final_forward=t['causal_pairs_including_final_forward'],
                       global_pair_sparsity_including_final_forward=t['global_pair_sparsity_including_final_forward'],
                       diagnostic_seconds_with_capture=t['wall_generation_seconds_with_capture'],
                       peak_memory_gib=t['peak_memory_gib'],gpu=t['gpu'],
                       active_reasoning_readout_cells=len(active),
                       own_prefix_recall25_vs_dense_same_step=statistics.mean(r['recall25_vs_dense_at_same_step'] for r in active) if active else None,
                       own_prefix_js_nats_vs_dense_same_step=statistics.mean(r['js_nats_vs_dense_at_same_step'] for r in active) if active else None,
                       readout_interpretation='different-own-prefix diagnostics; not isolated sparse fidelity')
            results.append(row)
            notes.extend([f'{LABELS[arm]}: generated={row["generated_tokens"]}; stop={row["stop_reason"]}; think_closed={row["think_closed"]}; final boxed={candidate}; final boxed correct={correct}',
                          f'Common generated prefix vs Dense={common} tokens; first different output token={first}.',
                          f'Duplicate16-gram fraction={row["duplicate_16gram_fraction"]:.4f}; tail1024={row["tail1024_duplicate_16gram_fraction"]:.4f}; exact periodic suffix={row["exact_periodic_suffix_tokens"]} tokens.',
                          f'Own text: captures/sample_{idx}/{arm}.txt', ''])
            for suffix in ['.json','.txt','.pt']:
                path=out/f'{arm}{suffix}'
                source_records.append(dict(path=str(path.relative_to(HERE)),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        for ia,a in enumerate(ARMS):
            for b in ARMS[ia+1:]:
                common = lcp(ts[a]['continuation_ids'],ts[b]['continuation_ids'])
                first = common+1 if common<min(ts[a]['generated_tokens'],ts[b]['generated_tokens']) else None
                snippets = {}
                if first is not None:
                    for arm in [a,b]:
                        ids=ts[arm]['continuation_ids']
                        snippets[arm]=dict(before=tok.decode(ids[max(0,common-40):common]),
                                           different_token=tok.decode([ids[common]]),
                                           after=tok.decode(ids[common+1:common+81]))
                pairs.append(dict(idx=idx,arm_a=a,arm_b=b,common_prefix_tokens=common,
                                  first_different_generated_token=first,snippets=snippets))
    assert len(results)==15 and len(pairs)==30
    with (HERE/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]))
        writer.writeheader();writer.writerows(results)
    by_arm = []
    for arm in ARMS:
        rows=[r for r in results if r['arm']==arm]
        by_arm.append(dict(arm=arm,label=LABELS[arm],rollouts=len(rows),
                           eos=sum(r['stop_reason']=='eos' for r in rows),
                           capped=sum(r['stop_reason']=='max_new_tokens' for r in rows),
                           think_closed=sum(r['think_closed'] for r in rows),
                           final_boxes_present=sum(r['final_boxed_answer'] is not None for r in rows),
                           correct_final_boxes=sum(r['correct_final_boxed_answer'] is True for r in rows),
                           average_tokens=statistics.mean(r['generated_tokens'] for r in rows)))
    summary=dict(status='completed',expected_rollouts=15,completed_rollouts=len(results),
                 sample_ids=[0,1,2],teacher_forced=False,input_alignment='passed/full',
                 pair_accounting='passed/independent',methods=by_arm,results=results,pairs=pairs,
                 caveats=['Only3 questions and one seeded rollout per method.',
                          'A capped trace without a final box is censored, not a completed wrong answer.',
                          'J-lens comparison after divergence mixes sparse and text-history differences.',
                          'Diagnostic timers include residual capture/transfers; not formal runtime.',
                          'Token n-gram repetition measures exact reuse; mathematical repetition can be legitimate.',
                          'An unsampled final forward is explicitly included in pair totals.'])
    (HERE/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (HERE/'comparison_notes_en.txt').write_text('\n'.join(notes)+'\n')
    (HERE/'source_manifest.json').write_text(json.dumps(source_records,indent=2)+'\n')
    (HERE/'completion.json').write_text(json.dumps(dict(status='completed',rollouts='15/15',
        input_alignment='passed/full',validation='passed',teacher_forced=False,
        analysis_scope='own-generation diagnostic; no formal accuracy claims'),indent=2)+'\n')
    print(json.dumps(dict(status='completed',rollouts='15/15',methods=by_arm)),flush=True)


if __name__=='__main__':main()
