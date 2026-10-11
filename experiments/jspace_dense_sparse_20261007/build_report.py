"""Audit the paired scientific pilot and export a standalone PDF companion."""
from pathlib import Path
import hashlib
import json
import math
import statistics
import textwrap

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages

from capture import HERE, FROZEN, ARMS, SPARSE_ARMS, dense_control, ids_sha, dump

LABELS={'dense':'Dense (append)', 'dense_full':'Dense (gather)',
        'full_random':'Random / full KV','full_w1':'W1 / full KV',
        'compact_random':'Random / compact KV','compact_w1':'W1 / compact KV'}


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):
            h.update(block)
    return h.hexdigest()


def load_rows(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def mean_by_sample(rows, metric):
    ids=sorted({r['idx'] for r in rows})
    return statistics.mean(statistics.mean(r[metric] for r in rows if r['idx']==i) for i in ids) if ids else None


def page(pdf,title,paragraphs=(),table=None,columns=None):
    fig=plt.figure(figsize=(8.27,11.69))
    fig.text(.085,.935,title,fontsize=19,weight='bold',va='top')
    y=.878
    for p in paragraphs:
        lines=textwrap.wrap(p,width=98)
        fig.text(.085,y,'\n'.join(lines),fontsize=10.5,va='top',linespacing=1.5)
        y-=.021*len(lines)+.024
    if table:
        height=.04*(len(table)+1)
        ax=fig.add_axes([.07,y-height,.86,height])
        ax.axis('off')
        t=ax.table(cellText=table,colLabels=columns,loc='center',cellLoc='center')
        t.auto_set_font_size(False)
        t.set_fontsize(9)
        t.scale(1,1.8)
        for (row,col),cell in t.get_celld().items():
            cell.set_edgecolor('#e2e5e8')
            if row==0:
                cell.set_facecolor('#edf2f5');cell.set_text_props(weight='bold')
    fig.text(.085,.045,'HMMT | Qwen3-4B | K=2048 | Fixed-prefix diagnostic, 2026-10-07',fontsize=9,color='#60666b')
    pdf.savefig(fig)
    plt.close(fig)


def figure_page(pdf,path,title,note):
    fig=plt.figure(figsize=(8.27,11.69))
    fig.text(.07,.94,title,fontsize=17,weight='bold',va='top')
    ax=fig.add_axes([.04,.19,.92,.66])
    ax.imshow(plt.imread(path));ax.axis('off')
    fig.text(.07,.13,'\n'.join(textwrap.wrap(note,width=100)),fontsize=10,va='top',linespacing=1.5)
    pdf.savefig(fig)
    plt.close(fig)


def main():
    config=json.loads((HERE/'run_config.json').read_text())
    summary=json.loads((HERE/'analysis/summary.json').read_text())
    assert summary['status']=='completed'
    assert summary['sample_ids']==config['sample_ids']
    assert summary['input_alignment']=='passed/full'
    assert digest(FROZEN/'inputs/hmmt.jsonl')==config['frozen_input_sha256']
    reference_rows=load_rows(FROZEN/'inputs/hmmt.jsonl')
    reference={r['idx']:r for r in reference_rows}
    assert len(reference)==len(reference_rows)==60
    assert json.loads((HERE/'pursuit_independent_check.json').read_text())['status']=='passed'
    lens=next((HERE/'assets').rglob('Qwen3-4B_jacobian_lens.pt'))
    assert digest(lens)==config['lens_sha256']
    for name,expected in config['source_sha256'].items():
        assert digest(HERE/name)==expected,name
    readouts=load_rows(HERE/'analysis/readouts.jsonl')
    gp=load_rows(HERE/'analysis/jspace_decomposition.jsonl')
    gamma_summary=json.loads((HERE/'analysis/gamma_sensitivity_summary.json').read_text())
    assert gamma_summary['status']=='completed'
    gamma_rows=load_rows(HERE/'analysis/gamma_sensitivity.jsonl')
    keys=lambda rows:{(r['idx'],r['arm'],r['layer'],r['step']) for r in rows}
    assert len(gamma_rows)==len(gp) and keys(gamma_rows)==keys(gp)
    primary_by_key={(r['idx'],r['arm'],r['layer'],r['step']):r for r in gp}
    for r in gamma_rows:
        p=primary_by_key[r['idx'],r['arm'],r['layer'],r['step']]
        assert r['random_control_gain']==p['random_control_gain']
        assert min(r['coefficients'])>=0 and len(set(r['token_ids']))==25
        assert all(math.isfinite(v) and -.00001<=v<=1.00001 for v in r['gain_curve'])
        assert all(b>=a-3e-6 for a,b in zip(r['gain_curve'],r['gain_curve'][1:]))
    assert len({(r['idx'],r['arm'],r['layer'],r['step']) for r in readouts})==len(readouts)
    assert len({(r['idx'],r['arm'],r['layer'],r['step']) for r in gp})==len(gp)
    for r in readouts:
        for metric in ('raw_residual_cosine','relative_residual_l2','recall25','meaningful_recall25','js_divergence_nats'):
            assert math.isfinite(r[metric])
        assert 0<=r['recall25']<=1 and 0<=r['meaningful_recall25']<=1
        assert -2e-6<=r['js_divergence_nats']<=math.log(2)+2e-6
        if r['arm'] in SPARSE_ARMS and r['regime']=='pre_sparse':
            assert r['recall25']==1 and r['relative_residual_l2']==0
    for r in gp:
        assert len(set(r['token_ids']))==len(r['token_ids'])==25
        assert min(r['coefficients'])>=0
        for curve in (r['gain_curve'],r['random_gain_curve']):
            assert all(math.isfinite(g) and -.00001<=g<=1.00001 for g in curve)
            assert all(b>=a-3e-6 for a,b in zip(curve,curve[1:]))
        assert abs(r['excess_gain']-(r['raw_reconstruction_gain']-r['random_control_gain']))<1e-8
        assert all(math.isfinite(r[m]) for m in ('component_cosine_to_dense','component_l2_to_dense','excess_gain'))
        assert -1.00001<=r['component_cosine_to_dense']<=1.00001
        if r['arm'] in SPARSE_ARMS and r['regime']=='pre_sparse':
            assert r['component_l2_to_dense']==0
    samples=[]
    provenance=[]
    for idx in config['sample_ids']:
        directory=HERE/'captures'/f'sample_{idx}'
        tr=json.loads((directory/'dense_trajectory.json').read_text())
        assert ids_sha(tr['prompt_ids'])==tr['input_ids_sha256']
        assert tr['input_ids_sha256']==reference[idx]['input_ids_sha256']
        assert tr['input_tokens']==len(tr['prompt_ids'])==reference[idx]['input_tokens']
        captures={a:torch.load(directory/f'{a}.pt',map_location='cpu',weights_only=True) for a in ARMS}
        d=captures['dense']
        assert captures['full_random']['stats']['selected_pairs']==captures['full_w1']['stats']['selected_pairs']
        assert captures['compact_random']['stats']['selected_pairs']==captures['compact_w1']['stats']['selected_pairs']
        for a,x in captures.items():
            assert x['prefix_sha256']==ids_sha(tr['prompt_ids']+tr['continuation_ids'])
            assert torch.equal(x['steps'],d['steps'])
            before=x['steps']<=2048-tr['input_tokens']
            if a in SPARSE_ARMS:
                assert torch.equal(x['residuals'][before],captures[dense_control(a)]['residuals'][before])
            s=x['stats']
            assert s['pair_accounting']=='passed/fixed_budget_analytic'
            assert abs(s['global_pair_sparsity']-(1-s['selected_pairs']/s['causal_pairs']))<1e-12
            provenance.append(dict(path=str((directory/f'{a}.pt').relative_to(HERE)),sha256=digest(directory/f'{a}.pt')))
        rs=[r for r in readouts if r['idx']==idx and r['arm']=='compact_random' and r['layer']==0]
        active=[r['step'] for r in rs if r['reasoning'] and r['regime']=='both_sparse']
        samples.append(dict(idx=idx,prompt_tokens=tr['input_tokens'],continuation_tokens=len(tr['continuation_ids']),
                            think_closed='</think>' in tr['continuation_text'],
                            captured_reasoning_points=sum(r['reasoning'] for r in rs),
                            active_reasoning_points=len(active),active_step_range=[min(active),max(active)] if active else None))
    active_ids=[x['idx'] for x in samples if x['active_reasoning_points']]
    assert active_ids, 'No active reasoning checkpoints: do not publish an empty sparse comparison.'
    groups=[]
    for a in ['dense_full',*SPARSE_ARMS]:
        rs=[r for r in readouts if r['arm']==a and r['reasoning'] and r['regime']=='both_sparse' and r['layer']<35]
        gs=[r for r in gp if r['arm']==a and r['reasoning'] and r['regime']=='both_sparse']
        groups.append(dict(arm=a,samples=len({r['idx'] for r in rs}),
                           meaningful_recall25=mean_by_sample(rs,'meaningful_recall25'),
                           raw_recall25=mean_by_sample(rs,'recall25'),
                           js_divergence_nats=mean_by_sample(rs,'js_divergence_nats'),
                           component_cosine=mean_by_sample(gs,'component_cosine_to_dense'),
                           excess_gain=mean_by_sample(gs,'excess_gain')))
    audit=dict(status='passed',samples=samples,active_reasoning_sample_ids=active_ids,
               readout_rows=len(readouts),jspace_rows=len(gp),
               checks=['unique rows','finite metrics','input and exact-prefix alignment','matched pre-budget equality',
                       'analytic actual-pair accounting','positive unique sparse coding support','monotone coding gain',
                       'lens and source SHA256','independent nonorthogonal solver sanity',
                       'Random/W1 actual-pair equality within storage setting'],groups=groups,capture_provenance=provenance)
    dump(HERE/'analysis/validation.json',audit)
    reconstruction=json.loads((HERE/'analysis/jspace_summary.json').read_text())
    primary_dense=next(r for r in reconstruction['groups'] if r['arm']=='dense')
    gamma_dense=next(r for r in gamma_summary['groups'] if r['arm']=='dense')
    group_by_arm={r['arm']:r for r in groups}
    interpretation=dict(status='completed',
        uncompressed_w1_minus_random_recall_pp=100*(group_by_arm['full_w1']['raw_recall25']-group_by_arm['full_random']['raw_recall25']),
        compressed_w1_minus_random_recall_pp=100*(group_by_arm['compact_w1']['raw_recall25']-group_by_arm['compact_random']['raw_recall25']),
        primary_dense_reconstruction=primary_dense,gamma_dense_reconstruction=gamma_dense,
        capacity_interpretation='not established: neither primary nor scale-aware frame beats the selected full-vocabulary Gaussian reconstruction control',
        caution='Readout fidelity is descriptive and distinct from capacity, semantic validity, accuracy and causal workspace function.',
        dense_kernel_calibration_max_relative_l2=max(r['relative_residual_l2'] for r in readouts if r['arm']=='dense_full'),
        sparse_arms_own_generation=False,max_new_tokens=4096)
    dump(HERE/'analysis/interpretation.json',interpretation)
    # Predeclared illustration, chosen before observing the pilot outcomes.
    example_rows=[r for r in readouts if r['idx']==0 and r['layer']==17 and r['step']==2560]
    dump(HERE/'analysis/concept_example.json',dict(idx=0,layer=17,step=2560,
          interpretation='Descriptive token readouts at one fixed coordinate; not causal semantic labels.',
          rows=example_rows))
    font=HERE.parents[2]/'report_build/hmmt_compact_results_20261006/fonts/Times.TTF'
    if font.exists():
        for face in font.parent.glob('Times*.TTF'):
            font_manager.fontManager.addfont(str(face))
        plt.rcParams['font.family']='Times New Roman'
    plt.rcParams['pdf.fonttype']=42
    table=[]
    for r in groups:
        table.append([LABELS[r['arm']],f"{r['meaningful_recall25']:.3f}",f"{r['raw_recall25']:.3f}",
                      f"{r['js_divergence_nats']:.4f}",f"{r['component_cosine']:.3f}" if r['component_cosine'] is not None else '-'])
    with PdfPages(HERE/'analysis/jspace_pilot_report.pdf') as pdf:
        page(pdf,'Sparse reasoning vs Dense: J-space pilot',[
            f"Qwen3-4B, HMMT sample IDs {config['sample_ids']}; total K=2048 includes the complete prompt and recent64. Each sparse arm replays exactly the same dense-generated token prefix. {len(active_ids)} of {len(samples)} samples contain checkpoints where both sparse settings are active inside reasoning.",
            "Two all-history Dense controls match the compact append kernel and the full-KV gather kernel. The table compares each sparse path with its matched control. The Dense-gather row is the separate cross-kernel calibration relative to Dense-append.",
            "Recall measures overlap of the 25 highest J-lens tokens. Wordlike recall uses the repository's display filter (letters and numbers allowed; punctuation and special tokens excluded). Raw recall is unfiltered. JS is full-vocabulary Jensen-Shannon divergence in nats. Component cosine compares nonnegative 25-atom J-space reconstructions, using the same fixed lens.",
            "Readout averages cover source blocks 0-34 and active reasoning checkpoints, weighting each sample equally. Component averages cover predeclared blocks 0, 8, 17, 26, 34. These are descriptive pilot measurements, not accuracy or causal workspace evidence.",
            "Reconstruction does not beat the selected random dictionary in either the primary or RMSNorm-scale-aware frame. J cosine is descriptive; the occupancy proxy is not interpreted as established workspace capacity."
        ],table,['Execution','Word recall','Raw recall','JS (nats)','J cosine'])
        page(pdf,'Readout differences are clearer than capacity',[
            f"With full KV retained, W1 increases mean unfiltered recall@25 by {interpretation['uncompressed_w1_minus_random_recall_pp']:.2f} percentage points relative to Random. With compact KV, the difference is {interpretation['compressed_w1_minus_random_recall_pp']:.2f} points. These directions hold in all three sampled cases. They describe agreement with Dense, not answer accuracy or a statistically established population advantage.",
            "Changes are more apparent in intermediate and later source blocks than in the final output readout. Compression moves Random and W1 differently in this pilot: Random compact is closer than Random full, while W1 full is closer than W1 compact. Different survivor history and selection frequency prevent attributing this solely to physical storage.",
            "The reconstruction control gives a negative result even for Dense. The table uses uncentered raw energy gain, averaged over the five predeclared depths and active reasoning positions with equal weight per sample. The Gaussian dictionary has the same full-vocabulary candidate count; its atoms compete through the same pursuit procedure.",
            "Adding the learned final RMSNorm scales changes the reconstructed geometry but does not remove the negative excess. This is a measurement sensitivity check on existing residuals, with no new inference or masks. It preserves the primary W_U J results.",
            "Consequently, this pilot does not establish the paper's workspace occupancy, a decrease in capacity after sparsity, or a causal explanation of the formal accuracy gap. Short-WikiText lens transfer, the chosen random control, uncentered energy and approximate pursuit may matter; this test does not identify their individual contributions."
        ],[['Literal W_U J',f"{primary_dense['raw_reconstruction_gain']*100:.2f}%",f"{primary_dense['random_control_gain']*100:.2f}%",f"{primary_dense['excess_gain']*100:.2f}"],
           ['RMSNorm-scale aware',f"{gamma_dense['raw_reconstruction_gain']*100:.2f}%",f"{gamma_dense['random_control_gain']*100:.2f}%",f"{gamma_dense['excess_gain']*100:.2f}"]],
           ['Dictionary frame','Dense gain','Random gain','Excess (pp)'])
        page(pdf,'Coverage and validation',[
            "All canonical input token hashes, exact paired prefixes, actual selected/causal pair counts, and pre-budget hidden states passed validation. Sparse and matched Dense states are bitwise identical before pruning; no cross-kernel equality is assumed.",
            "The rollout uses private seed 42/input RNG, temperature=0.6, top-p=0.95, BF16 and batch1, at most 4096 new tokens, with normal EOS handling. This matches the formal runner sampling law, not its batch2 RNG consumption. Sparse paths are teacher-forced counterfactuals; they do not generate their own answers.",
            "This pilot measures the onset of sparsity within the first 4096 generated tokens. Open think spans at the cap are truncated diagnostic prefixes. It does not measure the full long-reasoning regime of the formal 32768-token benchmark.",
            "Sampling is every 128 decode forwards plus checkpoints around the first full selection and first compact event. Compact deletion happens after attention every 64 forwards; its effect becomes visible on the next forward. Lines mark these boundaries."
            ,"Full KV can reselect any historical position on each forward. Compact KV chooses only from surviving positions and permanently removes the others. Thus full and compact differ in candidate history and selection frequency, not just physical storage. Random/W1 use identical actual pair budgets within each storage setting."
        ],[[s['idx'],s['prompt_tokens'],s['continuation_tokens'],s['captured_reasoning_points'],s['active_reasoning_points'],'yes' if s['think_closed'] else 'no'] for s in samples],
             ['Sample','Prompt','New tokens','Think points','Active points','Think closed'])
        figure_page(pdf,HERE/'analysis/readout_by_layer.png','J-lens change across depth',
                    'Only reasoning checkpoints where both sparse settings are active. The gray curve measures Dense kernel drift separately. A top-token overlap change is not by itself proof of loss of a semantic workspace.')
        figure_page(pdf,HERE/'analysis/readout_by_step.png','Change over a fixed reasoning trajectory',
                    'Full-KV selection begins at the dotted boundary; compact eviction first affects attention at the dashed boundary. Sparse paths read the same text. Uneven checkpoints near a boundary are deliberate diagnostics, not extra independent samples.')
        figure_page(pdf,HERE/'analysis/jspace_components.png','Approximate J-space sparse decomposition',
                    'Raw reconstruction gain minus a matched-size Gaussian-dictionary control; component cosine uses a kernel-matched all-history control. The negative excess also appears in Dense. These geometry similarities are descriptive and do not establish a privileged workspace or its capacity.')
        figure_page(pdf,HERE/'analysis/occupancy_proxy.png','Bounded occupancy diagnostic',
                    'The proxy counts the initial consecutive directions whose marginal gain exceeds the random control. Values without a crossing through k=25 atoms are right censored. The plotted value is the mean of within-sample medians, not a confidence interval. Negative total excess in Dense limits interpretation: do not call this established workspace capacity.')
        page(pdf,'Method, provenance and limits',[
            "Fixed measurement coordinates: public neuronpedia/jacobian-lens, Qwen3-4B, Salesforce-WikiText lens, fitted on 479 prompts. Immutable commit: "+config['lens_revision']+". Layer readout is softmax(W_U RMSNorm(J_l h_l)). The same 35 transport matrices are applied to every execution.",
            "For decomposition, atom directions are unit-normalized rows of W_U J_l. An explicit nonnegative gradient-pursuit approximation searches the entire vocabulary and uses at most 25 atoms. It optimizes squared reconstruction error with nonnegative coefficients; it is not a linear full-span projection. The exact paper pursuit implementation was not available.",
            "Gain is 1 - ||h-reconstruction||^2 / ||h||^2. It is an uncentered energy measure, not the paper's corpus-centered variance estimate. A full-vocabulary, same-size Gaussian unit dictionary supplies the excess-gain control. Sparse coding may have multiple correlated token explanations, so differences in atom names require care.",
            "The occupancy proxy is derived from existing k=1..25 atom gain curves without additional GPU decomposition. This atom count is separate from the KV budget K=2048. It records the initial run of above-control marginal gains. This bounded, approximate first-crossing diagnostic is not the paper's full occupancy measurement and has no statistical significance threshold.",
            "The lens was fit on short WikiText contexts, not long HMMT reasoning. This transfer is a measurement limitation. Three cases and one seed cannot establish a population effect or explain the formal benchmark accuracy difference. Future causal tests would need controlled intervention in the measured components.",
            "All sparse executions import the frozen HMMT selector implementation. No historical source or result was edited. Two independently scheduled L40 GPUs were used; capture timings include diagnostics and serialization, and are not compared with formal A100 decode times.",
            "References: transformer-circuits.pub/2026/workspace/index.html; local jacobian-lens repository. Machine-readable scope, source/lens hashes, exact token trajectories, residuals, per-position readouts, decompositions and validation are retained beside this PDF."
        ])
    print(json.dumps(dict(status='passed',active_ids=active_ids,groups=groups),indent=2),flush=True)


if __name__=='__main__':
    main()
