"""Export the requested five-method HMMT comparison, including raw-QK K2048."""
import csv
import hashlib
import json
from pathlib import Path
import sys

E = Path(__file__).resolve().parents[1]
WORKSPACE = E.parents[2]
sys.path.insert(0, str(E/'scripts'))
sys.path.insert(0, str(WORKSPACE/'.tools/report-runtime'))
from build_current_report import pdf_report
from consolidate_results import add
from current_report_layout import primary, k


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    out = WORKSPACE/'report_build/hmmt_comparison_20261015'
    out.mkdir(parents=True, exist_ok=True)
    old = read(WORKSPACE/'report_build/current/report_data.json')['results']
    rows = [dict(r) for r in old if primary(r)]
    root = 'hmmt_full_w1_raw_qk_fast_k2048_20261014'
    source = root+'/cells/hmmt_k2048_w1_raw_qk_fast_full_protected/formal/summary.json'
    new = add(source, dataset='hmmt', method='W1', storage='full',
              budget='total K=2048', protection='protected', score='raw-QK', window=1,
              notes='Completed raw-QK v2 follow-up; stable_sort backend, full KV, 20 formal shards.')
    rows.append(new)
    historical = next(dict(r) for r in old if r['dataset']=='hmmt' and
                      r['method']=='W1' and r['storage']=='full' and
                      r['score']=='probability' and k(r)==4096 and r['protection']=='protected')
    rows.append(historical)
    assert len(rows)==10 and len({(r['method'],r['score'],r['budget'],r['storage']) for r in rows})==10
    assert len({r['input_fingerprint'] for r in rows})==1
    assert all(r['questions']==60 and r['answers']==120 and r['input_alignment']=='passed/full' for r in rows)
    for r in rows:
        assert sha(E/'results'/r['source'])==r['source_sha256'], r['source']
        assert r['seed']==42 and r['batch_size']==2 and r['max_tokens']==32768
        assert r['generation_temperature']==.6 and r['generation_top_p']==.95
    dense = next(r for r in rows if r['method']=='Dense')

    def label(r):
        if r['method']=='Dense': return 'Dense'
        if r['method']=='Official random_pp': return 'Official random_pp'
        if r['storage']=='hybrid': return 'W1 + random4096 pool'
        if r['storage']=='full':
            return 'Full-KV W1 (probability; legacy)' if r['score']=='probability' else 'Full-KV W1 (optimized v2)'
        return 'W1: compress every1000 steps'

    def table(records, title):
        return dict(title=title, headers=['Method','Correct','Accuracy','Decode s / question','ms / decode step','Output tokens / answer'],
                    widths=[148,49,54,71,65,76], rows=[
                        [label(r), f"{r['correct']}/120", f"{100*r['accuracy']:.2f}%",
                         f"{r['decode_seconds']:,.1f}", f"{r['decode_ms_per_forward']:.1f}",
                         f"{r['output_tokens_mean']:,.1f}"] for r in records])

    def group(budget):
        candidates=[r for r in rows if k(r)==budget]
        ordered=[dense]
        ordered += [next(r for r in candidates if r['method']=='Official random_pp')]
        ordered += [next(r for r in candidates if r['storage']=='hybrid')]
        ordered += [next(r for r in candidates if r['storage']=='full')]
        ordered += [next(r for r in candidates if r['storage']=='compact' and r['method']=='W1')]
        assert len(ordered)==5
        return ordered

    content=[dict(title='HMMT results by attention budget', paragraphs=[
        '<b>Qwen3-4B · 60 questions × 2 sampled outputs · 15 October 2026</b>',
        'The new <b>raw-QK Full-KV W1 K=2048</b> run is complete: <b>55/120 (45.83%)</b>, '
        'with a mean decode time of <b>1,236.4 seconds per question batch</b>. All120 answers '
        'are present without duplicates; input alignment and analytic pair accounting passed.',
        '<b>Common settings:</b> BF16, batch2, seed42, representative seed0, sampling temperature0.6, '
        'generation Top-p0.95, and a total prompt+output limit of32768. Sparse methods protect '
        'prompt + recent64 within K. All W1 rows in the two tables below use <b>raw-QK scoring</b>.',
        '<b>Units:</b> decode seconds measure one question generating two sampled answers concurrently. '
        'Step time is total decode wall time divided by batch decode forwards. Output length is the '
        'mean per individual answer. Dense is the same baseline repeated in both tables.'
    ],tables=[table(group(1024),'K=1024'),table(group(2048),'K=2048')]),
    dict(title='Historical K=4096 result and method notes', paragraphs=[
        'The completed K=4096 Full-KV W1 result below uses the <b>earlier probability-scored implementation</b>. '
        'It is historical context, not a raw-QK K=4096 result. Local completed K=4096 results are unavailable '
        'for official random_pp, the W1 + random4096 pool, and W1 compression every1000 steps.',
    ],tables=[table([dense,historical],'K=4096: historical probability-scored control'),
        dict(title='How the five methods differ',headers=['Method','Decode attention and KV storage'],widths=[148,315],rows=[
            ['Dense','Attend to all historical KV; retain all KV.'],
            ['Official random_pp','Independent per-KV-head random selection and physical cache compression using its native64-step update phase.'],
            ['W1 + random4096 pool','Select protected K by current-query raw-QK every step. After attention every1000 steps, retain that core plus4096 Uniform-TopK candidates from its complement.'],
            ['Full-KV W1','Keep every historical KV; select protected K by raw-QK before each decode attention. K1024 and K2048 use optimized v2 with stable_sort.'],
            ['W1: compress every1000 steps','Attend to all surviving KV between updates. After attention every1000 steps, physically compress the cache to protected K; attention length can exceed K between updates.']
        ])])]
    # Notes follow the historical table on the second page.
    content[1]['tables'].append(dict(title='Interpretation and verification',headers=['Item','Details'],widths=[100,363],rows=[
        ['Timing','All methods ran on A100 GPUs; allocations and generated lengths differ. Decode includes scoring, selection, KV operations, attention and sampling, but excludes loading, warmup and diagnostic serialization. Small timing gaps do not establish a speed advantage.'],
        ['Accuracy','Scores are correct answers /120, not pass@2. At K2048, Full-KV W1 and the random4096 pool both score55/120; their1.1% mean decode-time gap is not a controlled speed comparison.'],
        ['New-run validation','24 correctness tests passed, followed by three full-generation smoke questions. All20 formal shards completed. Input alignment, coverage, raw-file hashes and analytic selected/causal pair totals passed.'],
        ['Source evidence','Exact source-summary paths, SHA-256 hashes, grades and metric-file references are retained in report_data.json and results.csv beside this PDF. Existing reports and raw experiment files are preserved.']
    ]))
    # Preserve established typography while using spaces in prose identifiers.
    replacements={'All120':'All 120','All20':'All 20','batch2':'batch 2','seed42':'seed 42','seed0':'seed 0',
                  'temperature0.6':'temperature 0.6','Top-p0.95':'Top-p 0.95','of32768':'of 32768',
                  'every1000':'every 1000','plus4096':'plus 4096','native64':'native 64',
                  'K1024':'K=1024','K2048':'K=2048','score55':'score 55','their1.1%':'their 1.1%'}
    for section in content:
        section['paragraphs']=[replace_text(p,replacements) for p in section['paragraphs']]
        for t in section['tables']:
            t['rows']=[[replace_text(str(v),replacements) for v in row] for row in t['rows']]
    fonts=WORKSPACE/'report_build/hmmt_compact_results_20261006/fonts'
    pdf_report(out/'hmmt_results_en.pdf', content, fonts, as_of='15 October 2026')
    snapshot=dict(as_of='2026-10-15',language='English',font='Times New Roman',results=rows,
                  new_run_verification=read(E/'results'/root/'completion_verification.json'),
                  scope='Five methods by K; K4096 probability-scored historical control explicitly separate.')
    (out/'report_data.json').write_text(json.dumps(snapshot,indent=2)+'\n')
    columns=['method','score','budget','storage','compaction_interval','questions','answers','correct','accuracy',
             'decode_seconds','decode_ms_per_forward','output_tokens_mean','output_tokens_median',
             'peak_memory_gib','selected_pairs','causal_pairs','global_pair_sparsity','source','source_sha256']
    with (out/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    (out/'README.md').write_text('# HMMT comparison, 15 October 2026\n\nEnglish PDF in Times New Roman. '
                               'Includes completed raw-QK Full-KV W1 K2048. Raw evidence is preserved.\n'
                               'Rebuild with `python scripts/build_hmmt_comparison_20261015.py` from the experiment directory.\n')
    print(json.dumps(dict(pdf=str(out/'hmmt_results_en.pdf'),unique_results=len(rows),new_K2048_correct=new['correct'])))


def replace_text(text,replacements):
    for a,b in replacements.items(): text=text.replace(a,b)
    return text


if __name__=='__main__': main()
