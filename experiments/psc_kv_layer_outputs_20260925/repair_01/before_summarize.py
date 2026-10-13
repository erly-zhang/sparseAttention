import csv,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
rows=[json.loads(x) for x in (HERE/'run/all_layers.jsonl').read_text().splitlines()]
assert len(rows)==320
fields=['method','layer','point','source_index','input_ids_sha256','valid_tokens','kl_dense_to_method','relative_l2','rmse','mean_token_cosine','retained_causal_pairs']
with (HERE/'run/all_layers.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
for point in sorted({r['point'] for r in rows}):
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for ax,metric in zip(axes.flat,['kl_dense_to_method','relative_l2','rmse','mean_token_cosine']):
        for method in dict.fromkeys(r['method'] for r in rows):
            rr=sorted((r for r in rows if r['point']==point and r['method']==method),key=lambda r:r['layer'])
            ax.plot([r['layer'] for r in rr],[r[metric] for r in rr],label=method)
        ax.set(xlabel='Layer (zero-based)',ylabel=metric);ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7);fig.suptitle('PSC KV retrieval index3: '+point+'; own end-to-end trajectories')
    fig.tight_layout();fig.savefig(HERE/'run'/f'{point}.png',dpi=160);plt.close(fig)
