"""Reader-facing report selection: latest HMMT, separate history and validation."""


def k(r):
    return int(r['budget'].split('=')[1]) if r['budget'].startswith('total K=') else None


def primary(r):
    if r['dataset'] != 'hmmt':
        return False
    if r['method'] == 'Dense':
        return True
    if r['protection'] != 'protected' or k(r) not in (1024,2048):
        return False
    if r['method'] == 'Official random_pp':
        return True
    return r['method'] in ('W1','W1 + random4096') and r['score']=='raw-QK' and (r['storage'] in ('full','hybrid') or
            (r['storage']=='compact' and r['compaction_interval']==1000))


def name(r):
    if r['method'].startswith('Dense'): return r['method']
    if r['method'] == 'Official random_pp': return 'Official random_pp'
    if r['method'] == 'Shared Random': return 'Shared Random'
    if r['score'] == 'probability': return r['method']+' (attention-probability scores)'
    if r['storage'] == 'full': return 'Full-KV W1'
    if r['storage'] == 'hybrid': return 'W1 + random4096'
    return 'Compressed W1'


def fmt(v,d=1):
    return '—' if v is None else format(v, ',.'+str(d)+'f')


def policy(r):
    if r['method']=='Dense': return 'Full history'
    if r['storage']=='full': return 'Full KV'
    return r['storage'].capitalize()+' / '+str(r['compaction_interval'])+' steps'


def section(title,paragraphs,tables=()):
    return dict(title=title,paragraphs=list(paragraphs),tables=list(tables))


def comparison(rows,title=None,show_k=False):
    headers=['Method','KV / update interval','Correct','Accuracy','Decode s','Output tokens']
    widths=[119,99,54,55,62,74]
    values=[[name(r),policy(r),str(r['correct'])+'/'+str(r['answers']),
             fmt(r['accuracy']*100,2)+'%',fmt(r.get('decode_seconds')),
             fmt(r.get('output_tokens_mean'),0)] for r in rows]
    if show_k:
        headers.insert(1,'K');widths=[112,35,90,50,51,56,69]
        for r,row in zip(rows,values):row.insert(1,str(k(r) or 'All'))
    t=dict(headers=headers,widths=widths,rows=values)
    if title:t['title']=title
    return t


def current_sections(data):
    rows=[r for r in data['results'] if primary(r)]
    dense=next(r for r in rows if r['method']=='Dense')
    best=max(rows,key=lambda r:r['accuracy'])
    current=section('Current HMMT comparison',[
        '<b>Qwen3-4B · 60 questions × 2 sampled outputs · 14 October 2026</b>',
        'This main report contains only the latest protected raw-QK W1 configurations and '
        'the matched Dense / official random_pp baselines. Earlier probability-scored W1, '
        'Shared Random controls, the 500-step predecessor, K=4096 controls, unprotected '
        'settings and other datasets are retained in a <b>separate historical report</b>.',
        '<b>All W1 rows below use raw-QK scoring.</b> Prompt + recent64 protection counts '
        'inside K. Dense is repeated for comparison; it has no K limit.',
        '<b>Read the columns:</b> accuracy is correct outputs /120, not pass@2. Decode seconds '
        'measure a question batch generating two sampled outputs concurrently. Output tokens '
        'are the mean per individual output. Runs use BF16, batch2, seed42, temperature0.6, '
        'generation Top-p0.95 and a total prompt+output limit of32768.',
        'The highest observed result in this comparison is <b>'+name(best)+', K='+str(k(best))+
        ': '+str(best['correct'])+'/120 ('+fmt(best['accuracy']*100,2)+'%)</b>. '
        'Dense scores '+str(dense['correct'])+'/120 ('+fmt(dense['accuracy']*100,2)+'%). '
        'These single-evaluation scores do not establish general superiority.',
        'A completed raw-QK full-KV W1 K=2048 formal result is not available; '
        'the older probability-scored result is kept in the historical report.',
    ],[comparison([dense]+[r for r in rows if k(r)==1024],'K=1024'),
        comparison([dense]+[r for r in rows if k(r)==2048],'K=2048')])
    diagnostics=section('Generation length, per-step cost and memory',[
        'Each row is named directly; no lookup IDs are needed. <b>Step ms</b> is total decode '
        'wall time divided by total batch decode forwards. <b>Median output</b> is per '
        'individual output. <b>Cap</b> counts outputs stopping at the total-length limit.',
        'Full-KV W1 K=1024 averages22,296 output tokens and52.5ms/forward; Dense averages '
        '19,316 output tokens and47.3ms/forward. Longer generation and higher per-step cost '
        'both contribute to its higher total decode time. These measurements do not isolate '
        'sorting time from the other selector, gather and attention costs.',
        '<b>Memory</b> is PyTorch peak allocated GiB, including the model. <b>Pair sparsity</b> '
        'is 1 − sum(selected pairs)/sum(causal pairs), including dense prefill and actual '
        'output lengths. Equal K does not imply equal global sparsity when lengths or KV policies differ.',
    ],[dict(headers=['Method','K','Step ms','Median output','Cap /120','Peak GiB','Pair sparsity'],
            widths=[143,36,51,75,57,49,52],rows=[
                [name(r),str(k(r) or 'All'),fmt(r['decode_ms_per_forward']),
                 fmt(r['output_tokens_median'],0),str(r['cap_hits']),fmt(r['peak_memory_gib'],2),
                 fmt(r['global_pair_sparsity']*100,2)+'%'] for r in rows])])
    methods=section('W1 scoring and KV policies',[
        '<b>W1 means one query:</b> use the current query from a representative Q head. '
        'For this model, a Q head reads its corresponding GQA KV head (four Q heads per KV head). '
        'Selected positions are shared within that layer; each head keeps its own K/V values.',
        '<b>Current raw-QK scoring:</b> score_j = dot(q_representative, k_j). '
        'Select TopK from the unprotected live candidates, then merge the protected positions. '
        'Earlier chronological positions break equal BF16 scores.',
        '<b>Earlier attention-probability scoring:</b> p_j = softmax(score_j /sqrt(head_dimension)). '
        'The old implementation ranks these probabilities. With one query, softmax preserves '
        'ranking in exact arithmetic; BF16 scaling, rounding and probability underflow can '
        'change ties in the implemented versions. The two score modes therefore remain '
        'separate in the historical record.',
        '<b>The selector softmax and attention softmax are different.</b> The current W1 selector '
        'does not use softmax. Actual attention still scales and jointly normalizes over the '
        'selected keys. Generation Top-p0.95 is also separate from KV TopK selection.',
        '<b>Hybrid update order:</b> W1 first chooses the current attention core C. After '
        'attention at each1000-step update, retain C plus4096 Uniform-TopK positions from its '
        'complement. For K=2048, an actual update retains6144 candidates; new KV can grow the '
        'pool to7144. Each next step selects2048 anew from that live pool. Skip eviction while '
        'all candidates fit.',
        '<b>Implementation status:</b> the main full-KV/hybrid K1024 rows are the completed '
        'optimized v2 results. K2048 compact1000/hybrid rows are the completed reference runs. '
        'The fused v3 has only GPU correctness and single-layer timings; those appear in '
        '<b>w1_v3_validation_en.pdf</b>, not in the formal accuracy tables.',
        '<b>Comparability:</b> all selected rows have full input-token/hash alignment and '
        'complete120-output coverage. A100 allocations differ across runs. The paired '
        'question-cluster95% interval for the best observed accuracy difference versus Dense '
        'is [0.0,12.5] percentage points, unadjusted for selecting among configurations. '
        'A real-model no-pruning equivalence test between native Dense and the sparse attention '
        'kernel has not yet been completed.',
    ],[dict(headers=['Method','Decode attention','Stored KV'],widths=[110,163,190],rows=[
        ['Full-KV W1','Select K every step','All historical KV'],
        ['Compressed W1','All surviving KV + new tokens','Compress to K after attention every1000 steps'],
        ['W1 + random4096','Select K every step','Every1000 steps retain the W1 core +4096 random candidates'],
        ['Official random_pp','All surviving per-head KV','Independent per-KV-head masks; native64-step update phase']])])
    return normalize([current,diagnostics,methods])


def history_sections(data):
    rows=[r for r in data['results'] if not primary(r)]
    hs=[r for r in rows if r['dataset']=='hmmt']
    dense=next(r for r in data['results'] if r['dataset']=='hmmt' and r['method']=='Dense')
    protected=[r for r in hs if r['protection']=='protected' and k(r)]
    old=section('Historical protected HMMT controls',[
        '<b>This is the historical reference report.</b> These configurations are excluded '
        'from the simplified main report but retain their verified numerical results. They '
        'include probability-scored W1, Shared Random and earlier KV-update controls. '
        'Dense is shown only as the common baseline. All sparse rows here protect prompt+recent64 inside K.',
        'Probability-scored W1 ranks attention probabilities from the current representative '
        'query. Raw-QK ranks its BF16 query-key products directly. They are independently labeled '
        'because finite-precision ties can differ. Results are correct/120; decode seconds '
        'are per question batch of two sampled outputs.',
    ],[comparison([dense]+[r for r in protected if k(r)==1024],'Historical K=1024'),
        comparison([dense]+[r for r in protected if k(r)==2048],'Historical K=2048')])
    sources={r['source']:r for r in data['results']+data['history']}
    proofs=[]
    for p in data['supersession_proofs']:
        a,b=sources[p['old']],sources[p['new']]
        proofs.append([name(b),str(k(b)),fmt(a['decode_seconds']),fmt(b['decode_seconds']),
                       str(b['correct'])+'/120'])
    k4=section('Historical K=4096 and replaced implementations',[
        'The K=4096 rows are local Shared Random / W1 controls, not official random_pp. '
        'There is no completed local official random_pp K=4096 result in this verified catalog.',
        'The replacement table retains earlier timings only for implementations with '
        '120/120 exactly identical complete output texts, input hashes and pair counts. '
        'Their current counterparts appear once in the appropriate main or historical table. '
        'Timing allocations differ across these formal runs.',
    ],[comparison([dense]+[r for r in protected if k(r)==4096],'Historical K=4096'),
        dict(title='Equivalent older implementations',headers=['Configuration','K','Old decode s','Current s','Correct'],
             widths=[180,42,81,80,80],rows=proofs)])
    unprotected=section('Historical HMMT without forced protection',[
        'These settings do not force retention of prompt or recent64. They are independent '
        'controls, retained outside the main report. All use the same60 HMMT inputs with two '
        'sampled outputs each. Their bad or good scores do not make them replacements for protected configurations.',
    ],[comparison([r for r in hs if r['protection']=='none'],show_k=True)])
    other=section('Older budget convention and other datasets',[
        'The HMMT extra-prompt reasoning1024 budget below is separate from total K=1024. '
        'All other dataset results are also independent splits; their scores are not pooled with HMMT.',
        'MATH500 has500 questions ×2 outputs. Legacy Dense has no separated synchronized '
        'decode or saved input-ID hashes. The full-KV random reasoning1024 result combines '
        '429 historical +71 repaired questions; its mixed timing is not a fresh full500 rerun.',
        'The other splits use Shared Random compact64 with an extra-prompt reasoning1024 '
        'budget. LCB is release_v6 medium383 with a test-based pass@1 estimate; no matched '
        'Dense baseline is available for those four splits in this catalog.',
    ],[comparison([r for r in hs if not k(r)],'Older HMMT: prompt + reasoning1024'),
        dict(headers=['Dataset','Method / scoring','Budget','KV / interval','Correct','Accuracy','Decode s'],
             widths=[55,116,92,61,48,44,47],rows=[
                [{'math':'MATH500','gpqa':'GPQA-D','aime25':'AIME25','aime26':'AIME26',
                  'livecodebench':'LCB'}[r['dataset']],name(r),r['budget'],policy(r),
                 str(r['correct'])+'/'+str(r['answers']),fmt(r['accuracy']*100,2)+'%',
                 fmt(r.get('decode_seconds'))] for r in rows if r['dataset']!='hmmt'])])
    return normalize([old,k4,unprotected,other])


def validation_sections(data):
    micro=data['v3_microbenchmark']
    return normalize([section('W1 fused v3: validation only',[
        '<b>No new full-HMMT accuracy or end-to-end speed result exists for v3.</b> '
        'This page is a separate correctness/microbenchmark record, not a formal result table.',
        'v3 merges final-index assembly, KV gather and selected-length updates in one launch. '
        'It reuses the v2 raw-BF16 QK matmul, rank/tie rules, representative identity and random '
        'streams. GPU checks passed for exact masks/KV/attention, ties, short histories, batch '
        'shrink, two physical pool updates and sampled token sequences on a controlled tiny Qwen3.',
        'Same A10020GB MIG slice; batch2,32 Q heads,8 KV heads, dimension128. Six alternating '
        'rounds of80 repetitions use synchronized wall time and exclude warmup. Timing covers '
        'the single-layer selection/append/gather/attention path; it excludes MLP, projections, '
        'output sampling and physical compaction events. The N=K short-history row skips QK ranking.',
    ],[dict(headers=['KV policy','K','Candidates N','v2 layer ms','v3 layer ms','Reduction'],
             widths=[70,40,92,87,87,87],rows=[
                [r['storage'],str(r['K']),str(r['N']),
                 fmt(r['measurements']['v2']['complete_layer_median_ms'],4),
                 fmt(r['measurements']['v3']['complete_layer_median_ms'],4),
                 fmt(r['complete_layer_reduction_percent'],2)+'%'] for r in micro['results']])])])


def normalize(content):
    replacements={'batch2':'batch 2','seed42':'seed 42','temperature0.6':'temperature 0.6',
        'Top-p0.95':'Top-p 0.95','of32768':'of 32768','averages22,':'averages 22,',
        'and52.5':'and 52.5','and47.3':'and 47.3','each1000':'each 1000',
        'plus4096':'plus 4096','retains6144':'retains 6144','to7144':'to 7144',
        'selects2048':'selects 2048','complete120':'complete 120','question-cluster95%':'question-cluster 95%',
        'native64':'native 64','every1000':'every 1000','same60':'same 60',
        'has500':'has 500','×2':'× 2','+71':'+ 71','full500':'full 500',
        'medium383':'medium 383','A10020GB':'A100 20GB','batch 2,32':'batch 2, 32',
        'heads,8':'heads, 8','dimension128':'dimension 128','of80':'of 80'}
    for s in content:
        for a,b in replacements.items():s['paragraphs']=[p.replace(a,b) for p in s['paragraphs']]
        for t in s['tables']:
            for a,b in replacements.items():
                t['rows']=[[v.replace(a,b) if isinstance(v,str) else v for v in row] for row in t['rows']]
    return content
