"""English scientific figures and a self-contained paired token slice viewer."""
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager
from common import HERE,SOURCE,METHODS,TASKS,LABELS,TASK_LABELS,read,save

FONT=HERE.parents[1]/'reports/mask_flexprefill_results_20261002/fonts/TimesNewRoman-Regular.ttf'
font_manager.fontManager.addfont(str(FONT))
plt.rcParams.update({'font.family':'Times New Roman','font.size':11,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
COLORS=['#333333','#2463a5','#2463a5','#bd7622','#bd7622','#7c497e','#61723c']
STYLES=['-','--','-','--','-','-','-']

def curves(data,metric,group='tail',title='',ylabel='',stem='chart'):
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True)
    for ti,task in enumerate(TASKS):
        ax=axes.flat[ti]
        for mi,method in enumerate(METHODS[1:],1):
            rows=sorted([r for r in data if r['task']==task and r['method']==method and r['group']==group and r.get('metric',metric)==metric],key=lambda r:r['layer'])
            if not rows:continue
            ax.plot([r['layer'] for r in rows],[r['mean'] for r in rows],color=COLORS[mi],ls=STYLES[mi],label=LABELS[mi],lw=1.7)
            if all('bootstrap_95_low' in r for r in rows):
                ax.fill_between([r['layer'] for r in rows],[r['bootstrap_95_low'] for r in rows],[r['bootstrap_95_high'] for r in rows],color=COLORS[mi],alpha=.08,lw=0)
        ax.set_title(TASK_LABELS[ti]);ax.set_xlabel('Layer (block output)');ax.set_ylabel(ylabel);ax.grid(alpha=.17);ax.set_xlim(0,31)
        if metric in ['js','residual_relative_l2','transported_relative_l2']:ax.set_ylim(bottom=0)
        if metric in ['top10_recall','top1_preserved','dense_active_support_recall','delta_energy_in_dense_active_span']:ax.set_ylim(0,1)
    fig.suptitle(title);handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False)
    fig.text(.5,.065,'50 paired prompts per task; sampled '+group+' positions; shading: 95% prompt-bootstrap intervals.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.10,1,.95));return fig

def main():
    out=HERE/'figures';out.mkdir(exist_ok=True)
    s=read(SOURCE/'outputs/summary.json');audit=read(SOURCE/'outputs/independent-audit.json')
    if audit['status']!='PASS' or audit['records']!=1400:raise ValueError('Unverified source')
    data=[dict(task=r['task'],method=r['method'],group=r['group'],layer=r['layer'],**r['summary']) for r in s['layer_means']]
    figs=[]
    figs.append(('tail_js',curves(data,'js',title='Dense–Sparse J-lens readout divergence',ylabel='JS divergence (nats; Dense = 0)')))
    figs.append(('uniform_js',curves(data,'js',group='uniform',title='Readout divergence at positions sampled across the full context',ylabel='JS divergence (nats; Dense = 0)')))
    fig,ax=plt.subplots(figsize=(10,6));labels=[];values=[];lo=[];hi=[]
    for task in TASKS:
        for strategy in ['coverage','random']:
            r=s['slash_minus_vertical_tail_js'][task][strategy];labels.append(TASK_LABELS[TASKS.index(task)]+' / '+('Calibrated' if strategy=='coverage' else 'Random'))
            values.append(r['mean']);lo.append(r['mean']-r['bootstrap_95_low']);hi.append(r['bootstrap_95_high']-r['mean'])
    y=np.arange(8);ax.errorbar(values,y,xerr=[lo,hi],fmt='o',color='#2463a5',capsize=4);ax.axvline(0,color='#333333',lw=1)
    ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlabel('Paired change in tail JS: Vertical+Slash − Vertical (nats)')
    ax.set_title('Effect of adding slash on readout divergence');ax.grid(axis='x',alpha=.17)
    fig.text(.5,.02,'Negative values indicate closer Dense readouts. 95% bootstrap intervals, 50 paired prompts; budgets differ.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.05,1,1));figs.append(('slash_effect',fig))
    val=read(SOURCE/'outputs/lens-validation.json');names=list(val['summary']);xx=np.arange(len(names))
    fig,ax=plt.subplots(figsize=(11,5));ax.bar(xx-.18,[val['summary'][n]['jlens']['10'] for n in names],.36,label='J-lens',color='#2463a5')
    ax.bar(xx+.18,[val['summary'][n]['logit_lens']['10'] for n in names],.36,label='Logit lens',color='#bd7622')
    ax.set_xticks(xx,[n.replace('lens-eval-','').replace('order-ops','Arithmetic') for n in names]);ax.set_ylim(0,1);ax.set_ylabel('Concept recovered in Top-10 at any layer');ax.set_title('Readout validation on the original 551 prompt items');ax.legend(frameon=False)
    fig.tight_layout();figs.append(('lens_validation',fig))
    rfile=HERE/'outputs/readout-summary.json'
    if rfile.exists():
        r=read(rfile)
        for metric,label in [('top10_recall','Dense Top-10 retained in Sparse Top-10'),('top1_preserved','Fraction of Dense Top-1 readouts preserved'),('residual_relative_l2','Relative L2 of residual change'),('transported_relative_l2','Relative L2 after J-lens transport')]:
            figs.append((metric,curves(r['layer_means'],metric,title='Dense–Sparse prefill comparison',ylabel=label)))
    jfile=HERE/'outputs/jspace-summary.json'
    if jfile.exists():
        j=read(jfile)
        for k in [16,25]:
            for metric,ylabel in [('dense_active_support_recall','Dense active support recall'),('excess_energy_reconstructed','Reconstructed residual energy above rotated control'),('delta_energy_in_dense_active_span','Fraction of change energy in Dense active span')]:
                rows=[r for r in j['layer_means'] if r['k']==k and r['metric']==metric]
                figs.append(('jspace_'+metric+'_'+str(k),curves(rows,metric,title='J-space comparison (k='+str(k)+')',ylabel=ylabel)))
    with PdfPages(out/'prefill_analysis.pdf') as pdf:
        for stem,fig in figs:
            fig.savefig(out/(stem+'.png'),dpi=150,bbox_inches='tight');fig.savefig(out/(stem+'.pdf'),bbox_inches='tight');pdf.savefig(fig,bbox_inches='tight');plt.close(fig)
    save(out/'figure-provenance.json',dict(source_audit=audit,figures=[x[0] for x in figs],scope='Saved prefill positions; historical task scores; no new generation',font='Times New Roman',jspace_available=jfile.exists()))
    make_viewer(s)
    print('FIGURES READY',len(figs),flush=True)

def make_viewer(summary):
    manifest=read(SOURCE/'input-manifest.json');bykey={i['key']:i for i in manifest['evaluation']};cases=[]
    for task in TASKS:
        scores={r['key']:r for r in summary['sample_summaries'] if r['task']==task and r['method']=='dense_full'}
        rows=[r for r in summary['sample_summaries'] if r['task']==task and r['method']=='coverage_token_vertical_slash']
        declined=[r for r in rows if scores[r['key']]['score']>r['score']]
        pool=declined or rows;pool=sorted(pool,key=lambda r:(r['tail_mean_js_nats'],r['key']));selected=pool[len(pool)//2];item=bykey[selected['key']]
        case=dict(key=item['key'],task=TASK_LABELS[TASKS.index(task)],selection='Median tail-JS among calibrated V+S score-decline samples' if declined else 'Median tail-JS across all samples',methods={})
        for method in METHODS:
            p=SOURCE/'outputs'/('replay-'+method+'-'+str(item['analysis_shard']))/(item['key']+'.json');v=read(p)
            case['methods'][method]=dict(positions=v['positions'],groups=v['position_groups'],context=v['context_tokens'],vocab=v['vocab_fragment'],score=v['historical_score'],history_match=v['history_replay_match']['first_token_equal'],layers={l:{f:z[f] for f in ['dense_top_ids','sparse_top_ids','dense_top1_rank_in_sparse','js_nats']} for l,z in v['layer_metrics'].items()})
        cases.append(case)
    if (HERE/'outputs/jspace-summary.json').exists():
        import torch
        from transformers import AutoTokenizer
        model=read(HERE/'manifest.json')['model'];tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
        for case in cases:
            task=TASKS[TASK_LABELS.index(case['task'])];case['jspace']={}
            for layer in range(32):
                data=torch.load(HERE/'outputs'/('worker-'+str(layer*4+TASKS.index(task)))/(case['key']+'.pt'),weights_only=True,map_location='cpu')
                n=len(data['positions']);case['jspace'][str(layer)]={}
                for k,fit in data['k'].items():
                    support=fit['support'].reshape(7,n,k);coefs=fit['coefficients'].reshape(7,n,k)
                    energy=fit['energy_reconstructed'].reshape(7,n);case['jspace'][str(layer)][str(k)]={}
                    for mi,method in enumerate(METHODS):
                        entries=[]
                        for pos in range(n):
                            ids=support[mi,pos].tolist();cs=coefs[mi,pos].tolist()
                            atoms=sorted([(int(t),float(c)) for t,c in zip(ids,cs) if t>=0 and c>0],key=lambda x:-x[1])
                            entries.append(dict(atoms=[[tokenizer.decode([t]),c] for t,c in atoms],energy=float(energy[mi,pos])))
                        case['jspace'][str(layer)][str(k)][method]=entries
    payload=json.dumps(dict(cases=cases,methods=METHODS,labels=LABELS),ensure_ascii=False).replace('</','<\/')
    template=(HERE/'viewer_template.html').read_text()
    (HERE/'figures/slice_viewer.html').write_text(template.replace('PAYLOAD',payload))
    save(HERE/'figures/case-selection.json',[{k:v for k,v in c.items() if k!='methods'} for c in cases])

if __name__=='__main__':main()
