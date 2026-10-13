"""Compare the added hybrid with frozen dense/full-C baseline metrics."""
import csv,json,math,os
from pathlib import Path
from hybrid_adapter import BASE,METHOD
HERE=Path(__file__).resolve().parent;RUN=Path(os.environ['PSC_RUN_DIR'])
old=[json.loads(x) for x in (BASE/'run_kl_resume/all_layers.jsonl').read_text().splitlines()]
new=[json.loads(x) for x in (RUN/METHOD/'layers.jsonl').read_text().splitlines()]
assert len(new)==64 and len({(x['layer'],x['point']) for x in new})==64
expected_hash=json.loads((BASE/'inputs/manifest.json').read_text())['records'][3]['input_ids_sha256']
assert all(x['source_index']==3 and x['input_ids_sha256']==expected_hash and x['valid_tokens']==123959 for x in old+new)
assert all(math.isfinite(x['kl_dense_to_method']) for x in old+new)
full={(x['layer'],x['point']):x for x in old if x['method']=='routing_c'}
maxdiff=0
for x in new:
    if x['layer']<=20:
        y=full[x['layer'],x['point']]
        for metric in ('kl_dense_to_method','relative_l2','rmse','mean_token_cosine'):
            maxdiff=max(maxdiff,abs(x[metric]-y[metric]))
            assert math.isclose(x[metric],y[metric],rel_tol=1e-5,abs_tol=1e-7),(x['layer'],metric,x[metric],y[metric])
(RUN/'prefix_parity.json').write_text(json.dumps({'status':'passed','prefix_layers':list(range(21)),
    'metric_max_absolute_difference':maxdiff,'reference':'frozen full-C run47087068','note':'same prefix should match; only layers21..31 change'},indent=2))
main=[x for x in old if x['method'] in ('dense','routing_c')]+new
fields=['method','layer','point','source_index','input_ids_sha256','valid_tokens','kl_dense_to_method','relative_l2','rmse','mean_token_cosine','retained_causal_pairs']
for name,rows in [('comparison',main),('all_six_methods',old+new)]:
    (RUN/(name+'.jsonl')).write_text(''.join(json.dumps(x)+'\n' for x in rows))
    with (RUN/(name+'.csv')).open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
summary=[]
for point in ('attention_post_o_proj','block_output'):
    for method in ('dense','routing_c',METHOD):
        rows=sorted([x for x in main if x['point']==point and x['method']==method],key=lambda x:x['layer'])
        assert [x['layer'] for x in rows]==list(range(32))
        summary.append({'point':point,'method':method,'mean_kl_32_layers':sum(x['kl_dense_to_method'] for x in rows)/32,
            'mean_kl_layers0_20':sum(x['kl_dense_to_method'] for x in rows[:21])/21,
            'mean_kl_layers21_31':sum(x['kl_dense_to_method'] for x in rows[21:])/11,
            'final_layer_kl':rows[-1]['kl_dense_to_method'],'final_layer_cosine':rows[-1]['mean_token_cosine']})
(RUN/'summary.json').write_text(json.dumps(summary,indent=2))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
labels={'dense':'Dense','routing_c':'C: all32 layers',METHOD:'C: layers0-20; Dense:21-31'}
for point in ('attention_post_o_proj','block_output'):
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,metric in zip(axes.flat,['kl_dense_to_method','relative_l2','rmse','mean_token_cosine']):
        for method in ('dense','routing_c',METHOD):
            rr=sorted([x for x in main if x['point']==point and x['method']==method],key=lambda x:x['layer'])
            ax.plot([x['layer'] for x in rr],[x[metric] for x in rr],label=labels[method])
        ax.axvline(20.5,color='grey',linestyle='--',linewidth=1)
        ax.set(xlabel='Layer (zero-based)',ylabel=metric);ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('KV retrieval index3 | '+point+'\nExploratory split chosen after observing index3; own end-to-end trajectories')
    fig.tight_layout();fig.savefig(RUN/(point+'.png'),dpi=160);plt.close(fig)
print(json.dumps(summary,indent=2))
