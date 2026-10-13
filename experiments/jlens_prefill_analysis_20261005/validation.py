"""GPU compatibility smoke test and original prompt-set readout evaluation."""
import argparse
from pathlib import Path
from common import HERE,config,read,save,sha,require_approval,runtime,load_model,Llama44LensModel,legacy
from metrics import ranks_of,distribution_metrics

NUMBER_WORDS={3:'three',4:'four',5:'five',6:'six',7:'seven',8:'eight',9:'nine',
              10:'ten',11:'eleven',12:'twelve',13:'thirteen',15:'fifteen',16:'sixteen',20:'twenty',24:'twenty-four'}
SYNONYMS={'addition':['addition','add','plus','+'],
          'subtraction':['subtraction','subtract','minus','-'],
          'multiplication':['multiplication','multiply','times','*','×'],
          'division':['division','divide','divided','/','÷'],
          'mod':['mod','modulo','remainder','%'],
          'squared':['squared','square','^','²']}


def dense_parity(stock,historical):
    """BF16 logits from FA2 and explicit FP32 attention are numerically distinct.

    Keep a per-logit bound and add output-distribution and decision checks.
    This checks compatibility of two Dense implementations, not sparse quality.
    """
    import torch
    metrics=distribution_metrics(stock,historical,min(10,stock.shape[-1]))
    report=dict(max_abs_logit_difference=float((historical-stock).abs().max()),
                js_nats=float(metrics['js_nats'].max()),
                kl_stock_to_historical_nats=float(metrics['kl_dense_to_sparse_nats'].max()),
                first_token_equal=bool(torch.equal(stock.argmax(-1),historical.argmax(-1))),
                top10_overlap=float(metrics['top_k_overlap'].min()),
                atol=.2,rtol=.02,maximum_js_nats=.001)
    torch.testing.assert_close(historical,stock,atol=.2,rtol=.02)
    if report['js_nats']>.001:
        raise ValueError('Dense output-distribution divergence exceeds smoke tolerance')
    if not report['first_token_equal']:
        raise ValueError('Dense first token differs in smoke comparison')
    return report


def concept_tokens(tokenizer,concept,order_ops=False):
    forms=[concept]
    if order_ops:
        forms=SYNONYMS.get(concept,forms)
        if concept.isdigit(): forms=forms+[NUMBER_WORDS.get(int(concept),concept)]
    ids=set()
    for form in forms:
        for variant in {form,form.title(),form.lower()}:
            for text in (variant,' '+variant):
                encoded=tokenizer.encode(text,add_special_tokens=False)
                if len(encoded)==1: ids.add(encoded[0])
    return sorted(ids)


def read_position(tokenizer,prompt,poetry=False,max_length=512):
    encoded=tokenizer(prompt,truncation=True,max_length=max_length)
    if not poetry: return len(encoded['input_ids'])-1
    if '\n' not in prompt: raise ValueError('Poetry readout newline missing/truncated')
    # The pinned Llama tokenizer returns zero-width offsets for some tokens,
    # including newlines. Locate the actual newline-bearing token instead;
    # punctuation and a newline may share one token. Keep the full tokenization
    # to reject truncation of the last newline rather than use an earlier one.
    full=tokenizer(prompt,truncation=False)['input_ids']
    matches=[i for i,token in enumerate(full) if '\n' in tokenizer.decode(
        [token],skip_special_tokens=True,clean_up_tokenization_spaces=False)]
    if not matches or matches[-1]>=len(encoded['input_ids']) or full[:len(encoded['input_ids'])]!=encoded['input_ids']:
        raise ValueError('Poetry readout newline missing/truncated')
    return matches[-1]


def smoke():
    require_approval()
    import torch
    from transformers import DynamicCache
    from jlens import ActivationRecorder
    runtime()
    old=legacy()
    c,_=old.preflight()
    model,tokenizer=load_model()
    wrapper=Llama44LensModel(model,tokenizer)
    text=read(HERE/'fit-prompts.json')[0]['text']
    ids=wrapper.encode(text,max_length=128)
    with torch.no_grad():
        stock=wrapper.forward(ids)
        stock_logits=model.lm_head(stock.last_hidden_state[:,-1]).float()
    # Fit's gradients come from the unpatched, stock Dense decoder.
    with ActivationRecorder(wrapper.layers,at=[0,15,31],start_graph_at=0) as recorder:
        wrapper.forward(ids.expand(2,-1))
        sources=[recorder.activations[0],recorder.activations[15]]
        grads=torch.autograd.grad(recorder.activations[31][:,16:-1].sum(),sources)
        if any(not torch.isfinite(g).all() for g in grads): raise ValueError('Invalid Dense backward')
    del grads,sources,recorder,stock
    old.runtime_quality().install_mlp(model)
    from adapter import install
    state=install(model,'dense_full',layers=c['layers'])
    try:
        with torch.no_grad():
            result=model.model(input_ids=ids,position_ids=torch.arange(ids.shape[1],device='cuda')[None],
                 attention_mask=torch.ones_like(ids),use_cache=True,past_key_values=DynamicCache(),return_dict=True)
            historical=model.lm_head(result.last_hidden_state[:,-1]).float()
        # Save diagnostics before acceptance checks, including on a failed run.
        observed=distribution_metrics(stock_logits,historical,10)
        save(HERE/'outputs/dense-parity-diagnostic.json',dict(
            max_abs_logit_difference=float((historical-stock_logits).abs().max()),
            js_nats=float(observed['js_nats'].max()),
            first_token_equal=bool(torch.equal(stock_logits.argmax(-1),historical.argmax(-1)))))
        parity=dense_parity(stock_logits,historical)
        delta=parity['max_abs_logit_difference']
    finally: state['close']()
    from selection import build_index,reference_attention
    from experiments.token_compacted_sparse import triton_token_compacted_prefill_attention
    from patterns import PatternIndex
    from kernel import pattern_prefill_attention
    q=torch.randn(1,513,32,128,device='cuda',dtype=torch.bfloat16)
    k=torch.randn(1,513,8,128,device='cuda',dtype=torch.bfloat16)
    v=torch.randn_like(k)
    options=dict(c['selection'],target_token_budget=256,autoblock_final_token_budget=256,
                 vertical_max_tokens=3,slash_max_offsets=5)
    for mode in sorted({s['selector_method'] for s in c['scenarios'].values()}):
        index,_=build_index(q,k,v,17,0,mode,options)
        dispatch=pattern_prefill_attention if isinstance(index,PatternIndex) else triton_token_compacted_prefill_attention
        actual=dispatch(q,k,v,index,token_chunk_size=options['token_chunk_size'])
        expected=reference_attention(q,k,v,index)
        torch.testing.assert_close(actual,expected,atol=.025,rtol=.02)
    from baselines import baseline_attention
    bq=q[:,:256]; bk=k[:,:256]; bv=torch.randn_like(bk)[:,0:1].expand_as(bk).contiguous()
    expected=bv.repeat_interleave(4,2)
    for mode in c['baselines']:
        for layer in (0,15,31):
            actual,_=baseline_attention(bq,bk,bv,mode,layer,c['baselines'])
            torch.testing.assert_close(actual,expected,atol=.1,rtol=.02)
    save(HERE/'outputs/gpu-smoke.json',dict(status='passed',stock_vs_historical_max_abs_logit_difference=delta,
       dense_implementation_parity=parity,
       backward='unpatched Dense, batch 2, finite gradients to block outputs 0 and 15',
       sparse_parity='original independent mask/GQA oracle tests reused',weights_updated=False))


def evaluate():
    require_approval()
    import torch
    from jlens import JacobianLens
    runtime()
    c=config()
    lens=JacobianLens.load(str(HERE/'outputs/lens.pt'))
    model,tokenizer=load_model()
    wrapper=Llama44LensModel(model,tokenizer)
    rows=[]
    for path in sorted(Path(c['jlens_repository']).glob('data/evaluations/*.json')):
        slug=path.stem
        for item in read(path)['items']:
            position=read_position(tokenizer,item['prompt'],poetry=slug.endswith('poetry'),
                                   max_length=c['validation']['max_seq_len'])
            j_logits,final_logits,_=lens.apply(wrapper,item['prompt'],positions=[position],max_seq_len=512)
            plain,_,_=lens.apply(wrapper,item['prompt'],positions=[position],use_jacobian=False,max_seq_len=512)
            j_logits[wrapper.n_layers-1]=final_logits
            plain[wrapper.n_layers-1]=final_logits
            targets=[concept_tokens(tokenizer,v,slug.endswith('order-ops')) for v in item['intermediates']]
            observed={}
            for name,logits in [('jlens',j_logits),('logit_lens',plain)]:
                best=[]
                for ids in targets:
                    if not ids:
                        best.append(None)
                        continue
                    best.append(min(int(ranks_of(values,torch.tensor([ids])).min()) for values in logits.values()))
                observed[name]=best
            rows.append(dict(dataset=slug,name=item['name'],position=position,intermediates=item['intermediates'],
                             single_token_ids=targets,min_layer_ranks=observed))
            if len(rows)%50==0: save(HERE/'outputs/lens-validation-progress.json',dict(completed=len(rows)))
    if len(rows)!=c['validation']['items']: raise ValueError('Evaluation incomplete')
    summary={}
    for dataset in sorted({r['dataset'] for r in rows}):
        selected=[r for r in rows if r['dataset']==dataset]
        summary[dataset]={}
        for name in ['jlens','logit_lens']:
            summary[dataset][name]={}
            for k in c['validation']['pass_k']:
                # Unnameable concepts count as failures and are reported separately.
                fractions=[sum(v is not None and v<=k for v in r['min_layer_ranks'][name])/
                           len(r['intermediates']) for r in selected]
                summary[dataset][name][str(k)]=sum(fractions)/len(fractions)
    save(HERE/'outputs/lens-validation.json',dict(status='complete',rows=rows,summary=summary,
       unnameable_concepts=sum(not ids for r in rows for ids in r['single_token_ids']),
       scope='original 551 prompt items; local Llama token forms and synonym inventory, not paper model reproduction',
       fitted_source_layers=c['fit']['source_layers'],readout_layers=list(range(wrapper.n_layers)),
       final_layer_identity=True,lens_sha256=sha(HERE/'outputs/lens.pt')))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['smoke','evaluate'])
    args=p.parse_args()
    if args.action=='smoke': smoke()
    else: evaluate()
