"""CPU-only audit and corpus choice for the speed pilot."""
from pathlib import Path
import statistics
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'jlens_prefill_analysis_20261005'))
from common import read,save,sha


def main():
    c=read(HERE/'config.json')
    inputs=read(HERE/'inputs.json')
    reports=[]
    for worker in range(5):
        p=HERE/'outputs'/('worker-'+str(worker))/'complete.json'
        if not p.exists():
            raise RuntimeError('Speed worker incomplete: '+str(worker))
        r=read(p)
        if (r['worker']!=worker or r['status']!='complete'
                or r['config_sha256']!=sha(HERE/'config.json')
                or r['inputs_sha256']!=sha(HERE/'inputs.json')
                or r['checkpoint_validation']['status']!='PASS'):
            raise ValueError('Invalid worker report')
        reports.append(r)
    short=reports[0]['records']
    wiki=[r['per_prompt_seconds'] for r in short if r['kind']=='wikitext']
    matched=[r['per_prompt_seconds'] for r in short if r['kind']=='infinitebench_128']
    wiki_median=statistics.median(wiki)
    long=[]
    for i,r in enumerate(reports[1:]):
        expected=inputs['infinitebench'][i]
        if r['key']!=expected['key'] or r['input_tokens']!=expected['original_tokens'] or not r['full_context']:
            raise ValueError('Full context was changed')
        successes=[a for a in r['attempts'] if a['status']=='measured_partial']
        best=min(successes,key=lambda a:a['projected_per_prompt_seconds']) if successes else None
        long.append(dict(key=r['key'],task=r['task'],input_tokens=r['input_tokens'],
            full_context=True,status='measured_partial' if best else 'out_of_memory',
            attempts=r['attempts'],best_projection_seconds=best['projected_per_prompt_seconds'] if best else None,
            ratio_to_wikitext=best['projected_per_prompt_seconds']/wiki_median if best else None,
            full_fit_completed=False))
    # Infeasible samples have no invented elapsed-time ratio.
    selected='infinitebench_full_context' if all(r['status']=='measured_partial' and
        r['ratio_to_wikitext']<=c['decision_ratio'] for r in long) else 'wikitext'
    result=dict(status='PASS',selected_fit_corpus=selected,threshold_ratio=c['decision_ratio'],
        decision_rule=c['decision_rule'],wikitext_full_fit_seconds=wiki,
        wikitext_median_seconds=wiki_median,wikitext_range_seconds=[min(wiki),max(wiki)],
        infinitebench_128_full_fit_seconds=matched,
        infinitebench_128_median_seconds=statistics.median(matched),
        matched_length_ratio=statistics.median(matched)/wiki_median,
        full_context_infinitebench=long,formal_pipeline_run=False,
        timing_scope='Model already loaded and warmed; GPU synchronization included. Short results include native fit accumulation and diagnostics but exclude checkpoint serialization. Long projections include gradient-row reduction and CPU copies but omit full-fit accumulation/diagnostics/serialization.',
        caveats=['Long results measure only four of 4096 dimensions and extrapolate remaining backward batches.',
                 'All workers use one H100 80GB; multi-GPU model parallelism and CPU offload are not benchmarked.',
                 'Two WikiText samples and one long sample per task are a speed pilot, not a population estimate.',
                 'Short InfiniteBench results truncate only the supplemental matched-length test; corpus selection uses original complete long prompts.'],
        report_sha256={str(HERE/'outputs'/('worker-'+str(i))/'complete.json'):
            sha(HERE/'outputs'/('worker-'+str(i))/'complete.json') for i in range(5)})
    save(HERE/'summary.json',result)
    print('Corpus decision:',selected)
    print('WikiText median seconds:',wiki_median)
    print('Matched 128-token InfiniteBench ratio:',result['matched_length_ratio'])
    for r in long:
        print(r['task'],r['input_tokens'],r['status'],r['best_projection_seconds'],r['ratio_to_wikitext'])


if __name__=='__main__':
    main()
