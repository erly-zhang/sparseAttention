"""Standalone scientific figures from the paired diagnostic rows."""
import json
import statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from capture import HERE, ARMS, SPARSE_ARMS, dump

font=HERE.parents[2]/'report_build/hmmt_compact_results_20261006/fonts/Times.TTF'
if font.exists():
    for face in font.parent.glob('Times*.TTF'):
        font_manager.fontManager.addfont(str(face))
    plt.rcParams['font.family']='Times New Roman'
plt.rcParams.update({'font.size':11,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
COLORS={'full_random':'#cf7140','full_w1':'#a43b55','compact_random':'#43a29a','compact_w1':'#346f9b'}
LABELS={'full_random':'Random, full KV','full_w1':'W1, full KV','compact_random':'Random, compact KV','compact_w1':'W1, compact KV'}
STYLES={'full_random':'--','full_w1':':','compact_random':'-','compact_w1':'-.'}


def read(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def save(fig,name):
    for ext in ('png','pdf','svg'):
        fig.savefig(HERE/'analysis'/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def means_by_sample(rows,metric):
    return statistics.mean(statistics.mean(r[metric] for r in rows if r['idx']==i) for i in {r['idx'] for r in rows})


def main():
    rows=read(HERE/'analysis/readouts.jsonl')
    active=[r for r in rows if r['reasoning'] and r['regime']=='both_sparse']
    fig,ax=plt.subplots(1,2,figsize=(10,3.7))
    for arm in SPARSE_ARMS:
        group=[r for r in active if r['arm']==arm]
        layers=sorted({r['layer'] for r in group})
        for a,m in zip(ax,('meaningful_recall25','js_divergence_nats')):
            a.plot(layers,[means_by_sample([r for r in group if r['layer']==l],m) for l in layers],label=LABELS[arm],color=COLORS[arm],linestyle=STYLES[arm])
    calibration=[r for r in active if r['arm']=='dense_full']
    layers=sorted({r['layer'] for r in calibration})
    for a,m in zip(ax,('meaningful_recall25','js_divergence_nats')):
        a.plot(layers,[means_by_sample([r for r in calibration if r['layer']==l],m) for l in layers],color='#777777',linestyle='--',linewidth=1,label='Dense kernel calibration')
    ax[0].set(ylabel='Meaningful-token recall@25 vs matched Dense',xlabel='Transformer block (zero indexed)',ylim=(0,1.05))
    ax[1].set(ylabel='J-lens JS divergence (nats)',xlabel='Transformer block (zero indexed)')
    ax[0].legend(fontsize=9,frameon=False)
    fig.suptitle('HMMT fixed-prefix reasoning pilot | K=2048, prompt + recent64 protected')
    fig.text(.5,-.04,'Matched dense controls; equal weight per sample. Block35 is actual output; no workspace band is assumed.',ha='center',fontsize=9)
    save(fig,'readout_by_layer')

    fig,axs=plt.subplots(len({r['idx'] for r in rows}),1,figsize=(9,3*len({r['idx'] for r in rows})),squeeze=False)
    for ax,idx in zip(axs[:,0],sorted({r['idx'] for r in rows})):
        for arm in SPARSE_ARMS:
            group=[r for r in rows if r['idx']==idx and r['arm']==arm and r['reasoning'] and r['layer']<35]
            steps=sorted({r['step'] for r in group})
            ax.plot(steps,[statistics.mean(r['meaningful_recall25'] for r in group if r['step']==s) for s in steps],label=LABELS[arm],color=COLORS[arm],linestyle=STYLES[arm])
        trajectory=json.loads((HERE/'captures'/f'sample_{idx}'/'dense_trajectory.json').read_text())
        first_full=max(1,2048-trajectory['input_tokens']+1)
        first_compact=((first_full+63)//64)*64+1
        last=max((r['step'] for r in rows if r['idx']==idx and r['reasoning']),default=0)
        if first_full<=last:
            ax.axvline(first_full,color='#999999',linestyle=':',linewidth=.8)
        if first_compact<=last:
            ax.axvline(first_compact,color='#999999',linestyle='--',linewidth=.8)
        ax.set(title=f'Sample {idx}: matched dense-generated prefix',xlabel='Teacher-forced decode forwards',ylabel='Mean recall@25 (L0-L34)',ylim=(0,1.05))
        ax.legend(fontsize=8,frameon=False,ncol=2)
    fig.tight_layout()
    save(fig,'readout_by_step')

    gp=read(HERE/'analysis/jspace_decomposition.jsonl')
    gp_active=[r for r in gp if r['regime']=='both_sparse' and r['reasoning']]
    fig,ax=plt.subplots(1,2,figsize=(10,3.7))
    for arm in ARMS:
        if arm=='dense_full':
            continue
        group=[r for r in gp_active if r['arm']==arm]
        layers=sorted({r['layer'] for r in group})
        for a,m in zip(ax,('excess_gain','component_cosine_to_dense')):
            a.plot(layers,[means_by_sample([r for r in group if r['layer']==l],m) for l in layers],marker='o',color=COLORS.get(arm,'#333333'),linestyle=STYLES.get(arm,'-'),label=LABELS.get(arm,'Dense'))
    ax[0].axhline(0,color='#aaaaaa',lw=.8)
    ax[0].set(xlabel='Predeclared transformer block',ylabel='Raw reconstruction gain minus random control')
    ax[1].set(xlabel='Predeclared transformer block',ylabel='J-component cosine vs matched Dense')
    ax[0].legend(frameon=False,fontsize=8)
    fig.suptitle('Approximate nonnegative J-space coding, k=25')
    fig.text(.5,-.04,'Full-vocabulary pursuit; descriptive approximation. Raw gain is not corpus variance explained.',ha='center',fontsize=9)
    save(fig,'jspace_components')

    grouped=[]
    for arm in ARMS:
        group=[r for r in gp_active if r['arm']==arm]
        grouped.append(dict(arm=arm,samples=len({r['idx'] for r in group}),
                            **{m:means_by_sample(group,m) if group else None for m in
                               ('raw_reconstruction_gain','random_control_gain','excess_gain','component_cosine_to_dense','active_atoms')}))
    dump(HERE/'analysis/jspace_summary.json',dict(status='completed',groups=grouped))

    # Bounded first-crossing proxy, derived from saved K=1..25 gain curves.
    # A missing crossing is right censored; never call it full occupancy.
    occupancy=[]
    for r in gp:
        gains=[0.,*r['gain_curve']]
        controls=[0.,*r['random_gain_curve']]
        crossing=next((k for k in range(1,26) if gains[k]-gains[k-1]<=controls[k]-controls[k-1]),None)
        occupancy.append(dict(idx=r['idx'],arm=r['arm'],layer=r['layer'],step=r['step'],
                              reasoning=r['reasoning'],regime=r['regime'],
                              first_crossing_occupancy_proxy=crossing-1 if crossing is not None else 25,
                              right_censored_at25=crossing is None))
    (HERE/'analysis/occupancy_proxy.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in occupancy))
    fig,ax=plt.subplots(figsize=(8.8,4))
    for arm in ARMS:
        group=[r for r in occupancy if r['arm']==arm and r['reasoning'] and r['regime']=='both_sparse']
        layers=sorted({r['layer'] for r in group})
        values=[]
        for l in layers:
            rs=[r for r in group if r['layer']==l]
            values.append(statistics.mean(statistics.median(r['first_crossing_occupancy_proxy'] for r in rs if r['idx']==i) for i in {r['idx'] for r in rs}))
        ax.plot(layers,values,marker='o',color=COLORS.get(arm,'#333333' if arm=='dense' else '#777777'),
                linestyle=STYLES.get(arm,'-' if arm=='dense' else '--'),
                label=LABELS.get(arm,'Dense (append)' if arm=='dense' else 'Dense (gather)'))
    ax.set(xlabel='Predeclared transformer block',ylabel='Bounded occupancy proxy (0-25)',ylim=(-1,26),
           title='Mean within-sample median first crossing of random marginal gain')
    ax.legend(frameon=False,fontsize=9,ncol=2)
    fig.text(.5,-.025,'Saved k=1..25 atom curves only; absence of a crossing is right censored at25. This is not full occupancy.',ha='center',fontsize=9)
    save(fig,'occupancy_proxy')


if __name__=='__main__':
    main()
