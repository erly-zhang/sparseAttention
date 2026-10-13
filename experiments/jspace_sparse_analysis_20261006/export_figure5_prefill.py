"""Figure-5-style prefill explorer, including all correct-value token probes.

Full target ranks come from verified saved-residual post-processing. Arbitrary
non-target probes retain the original partial Dense Top-10 rank coverage.
"""
from pathlib import Path
from html import escape
import hashlib
import json
import re
import math
import numpy as np
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
OUT = HERE / 'review'
SOURCE = HERE.parent / 'jlens_prefill_analysis_20261005'
METHODS = ['dense_full', 'coverage_token_vertical_slash', 'flexprefill']
LABELS = ['Dense', 'Calibrated V+S', 'FlexPrefill']


def plot_value_readouts(data):
    """Export a shareable snapshot of the full-value panel at the prompt end."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import LogNorm
    font = HERE.parents[1] / 'reports/mask_flexprefill_results_20261002/fonts/TimesNewRoman-Regular.ttf'
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': 'Times New Roman', 'svg.fonttype': 'path'})
    fig, axes = plt.subplots(1, 3, figsize=(17, 9), sharey=True)
    pi = data['positions'].index(data['input_tokens'] - 1)
    ti = [data['readout_target_ids'].index(t) for t in data['target_ids']]
    for method, label, ax in zip(METHODS, LABELS, axes):
        values = np.array([[data['target_readouts'][method][l]['ranks'][pi][j] for j in ti] for l in range(32)])
        image = ax.imshow(values, cmap='Blues_r', norm=LogNorm(vmin=1, vmax=data['vocab_size']), aspect='auto')
        ax.set_title(label, fontsize=17, pad=12)
        ax.set_xticks(range(25), [f'{i+1}: {p}' for i, p in enumerate(data['target_pieces'])], rotation=90, fontsize=10)
        ax.set_yticks(range(0, 32, 2), range(0, 32, 2))
        ax.set_xlabel('Ordered correct-value token pieces', fontsize=13)
        ax.axvline(data['uniqueness']['half_tokens'] - .5, color='#bd7622', lw=1.5, linestyle='--')
    axes[0].set_ylabel('Layer (block output)', fontsize=14)
    fig.suptitle('KV Retrieval · kv_retrieval-359 · final prefill position 124,404', fontsize=20, y=.98)
    fig.subplots_adjust(left=.045, right=.915, bottom=.22, top=.90, wspace=.10)
    color_axis = fig.add_axes([.933, .27, .012, .58])
    bar = fig.colorbar(image, cax=color_axis, ticks=[1, 10, 100, 1000, 10000, 128256])
    bar.set_label('Full-vocabulary rank (lower is better)', fontsize=13)
    bar.ax.set_yticklabels(['1', '10', '100', '1,000', '10,000', '128,256'])
    fig.text(.5, .095, 'Dashed line: first 13/25 tokens (rounded-up 50% prefix). Repeated token IDs repeat the same probe.', ha='center', fontsize=13)
    fig.text(.5, .06, 'These are individual next-token readouts at one prompt position; they do not measure a joint value probability or K→V binding.', ha='center', fontsize=13)
    for ext in ['png', 'svg']:
        fig.savefig(OUT / f'kv_full_value_token_ranks.{ext}', dpi=160, bbox_inches='tight')
    plt.close(fig)


def main():
    manifest = json.loads((SOURCE / 'input-manifest.json').read_text())
    item, = [i for i in manifest['evaluation'] if i['key'] == 'kv_retrieval-359']
    config = json.loads((SOURCE / 'config.json').read_text())
    tok = AutoTokenizer.from_pretrained(config['model'], local_files_only=True)
    target_ids = tok(item['answers'][0], add_special_tokens=False)['input_ids']
    data = dict(key=item['key'], answer=item['answers'][0], input_tokens=item['input_tokens'],
                methods=METHODS, labels=LABELS, target_ids=target_ids,
                target_pieces=[tok.decode([t]) for t in target_ids], vocab={}, method_data={},
                vocab_size=128256, source_sha256={},
                rank_scope='Complete full-vocabulary ranks for all 18 unique correct-value token IDs, at all 32 layers and 202 saved positions. Other pinned tokens retain partial Dense Top-10 reference coverage.')
    for method in METHODS:
        path = SOURCE / 'outputs' / f"replay-{method}-{item['analysis_shard']}" / (item['key'] + '.json')
        r = json.loads(path.read_text())
        data['source_sha256'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        if method == 'dense_full':
            data['positions'] = r['positions']
            data['groups'] = r['position_groups']
            data['context'] = r['context_tokens']
            data['dense_ids'] = [r['layer_metrics'][str(l)]['dense_top_ids'] for l in range(32)]
        assert r['positions'] == data['positions']
        data['vocab'].update(r['vocab_fragment'])
        history = json.loads(Path(item['history'][method]['path']).read_text())
        data['method_data'][method] = dict(
            top_ids=[r['layer_metrics'][str(l)]['sparse_top_ids'] for l in range(32)],
            reference_ranks=[r['layer_metrics'][str(l)]['dense_top_ranks_in_sparse'] for l in range(32)],
            score=r['historical_score'], prediction=history['prediction'],
            stop_reason=history['stop_reason'], output_tokens=len(history['generated_ids']))
    for t in target_ids:
        data['vocab'][str(t)] = tok.decode([t])
    # Count both vocabulary-token occurrences and value-prefix matches at their
    # respective grains. A token match is not a match of an entire UUID.
    inputs = {}
    for suffix in ['.prompt', '.ids']:
        path, = [Path(p) for p in manifest['files'] if p.endswith(item['key'] + suffix)]
        expected = manifest['files'][str(path)]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == (expected['sha256'] if isinstance(expected, dict) else expected)
        data['source_sha256'][str(path)] = digest
        inputs[suffix] = path
    ids = np.fromfile(inputs['.ids'], dtype='<i8')
    assert len(ids) == item['input_tokens']
    prompt = inputs['.prompt'].read_text()
    uuid = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
    pairs = re.findall('"(' + uuid + ')"\\s*:\\s*"(' + uuid + ')"', prompt)
    values = [v for _, v in pairs]
    half = math.ceil(len(target_ids) / 2)
    half_prefix = tok.decode(target_ids[:half])
    data['uniqueness'] = dict(
        value_count=len(values), distinct_values=len(set(values)),
        token_input_occurrences={str(t): int(np.sum(ids == t)) for t in set(target_ids)},
        first_piece_value_prefix_matches=sum(v.startswith(tok.decode([target_ids[0]])) for v in values),
        half_tokens=half, half_prefix=half_prefix,
        half_prefix_value_matches=sum(v.startswith(half_prefix) for v in values),
        full_value_matches=values.count(item['answers'][0]))
    assert len(values) == 2500 and data['uniqueness']['first_piece_value_prefix_matches'] == 7
    assert data['uniqueness']['token_input_occurrences'][str(target_ids[0])] == 289
    assert data['uniqueness']['half_prefix_value_matches'] == data['uniqueness']['full_value_matches'] == 1

    readout_dir = OUT / 'target_value_readouts'
    complete = json.loads((readout_dir / 'complete.json').read_text())
    readout_provenance = json.loads((readout_dir / 'provenance.json').read_text())
    assert complete['status'] == 'PASS' and complete['layers'] == 32
    assert readout_provenance['positions'] == data['positions']
    assert readout_provenance['target_ids'] == sorted(set(target_ids))
    data['readout_target_ids'] = readout_provenance['target_ids']
    data['target_readouts'] = {m: [] for m in METHODS}
    compatibility = {m: [] for m in METHODS}
    for l in range(32):
        name = f'layer-{l}.json'
        content = (readout_dir / name).read_bytes()
        assert hashlib.sha256(content).hexdigest() == complete['files'][name]
        row = json.loads(content)
        assert row['layer'] == l and row['target_ids'] == data['readout_target_ids']
        for method in METHODS:
            record = row['methods'][method]
            assert record['complete'] and len(record['ranks']) == 202
            assert all(len(r) == 18 and all(1 <= t <= 128256 for t in r) for r in record['ranks'])
            data['target_readouts'][method].append({k: record[k] for k in ['ranks', 'rank_best', 'rank_worst']})
            compatibility[method].append(record['legacy_rank_tie_compatibility'])
    data['target_readout_provenance'] = readout_provenance
    data['legacy_rank_compatibility'] = compatibility
    assert len(data['positions']) == 202
    for method in METHODS:
        d = data['method_data'][method]
        assert all(len(layer) == 202 for layer in d['reference_ranks'])
        assert all(1 <= rank <= 128256 for layer in d['reference_ranks'] for row in layer for rank in row)
    position = data['positions'].index(item['input_tokens'] - 1)
    target = target_ids[0]
    full_rank_trace = {}
    for method in METHODS:
        ranks = []
        for layer in range(32):
            ranks.append(data['target_readouts'][method][layer]['ranks'][position][data['readout_target_ids'].index(target)])
        full_rank_trace[method] = ranks
    assert full_rank_trace['coverage_token_vertical_slash'][30] == 305
    assert full_rank_trace['coverage_token_vertical_slash'][31] == 1948
    data['first_piece_final_position_full_rank_trace'] = full_rank_trace
    (OUT / 'figure5_prefill_data.json').write_text(json.dumps(data, ensure_ascii=False))
    plot_value_readouts(data)

    template = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Figure 5-style Prefill Explorer · KV Retrieval</title><link rel="stylesheet" href="assets/report.css">
<style>
.layout{display:block;max-width:1550px}.controls{display:flex;flex-wrap:wrap;gap:14px;align-items:end;margin:20px 0}.controls label{font-size:16px;display:flex;flex-direction:column;gap:4px}.value-scroll{overflow:auto;border:1px solid #dce1e6;max-height:570px}.value-table{overflow:auto;max-height:450px;margin-top:16px}.value-table table{width:100%;font-size:15px}.value-table td,.value-table th{padding:7px;text-align:left;border-bottom:1px solid #e1e6eb}.value-table thead{position:sticky;top:0;background:white}.value-table button{font:inherit;background:white;border:1px solid #bac6d1;cursor:pointer;padding:3px 7px}
select,input,.controls button{font:inherit;padding:7px;border:1px solid #bac6d1;border-radius:3px;background:white}.controls button{cursor:pointer}.controls button:disabled{cursor:default;opacity:.5}#token-pin{max-width:280px}.explorer{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(310px,1fr);gap:24px;margin-top:22px}
.grid-scroll{overflow:auto;max-height:590px;border:1px solid #dce1e6}canvas{display:block}.panel{margin:0 0 25px}.panel h2{font-size:23px}.chart svg{display:block;width:100%;height:auto}.chart{margin-top:10px}.details-list{font-size:16px;display:grid;grid-template-columns:1fr 1fr;gap:10px}.details-list ol{padding-left:26px;margin:8px 0}.details-list h3{font-size:17px}.warning{background:#f4f7fa;border-left:3px solid #7ca7cb;padding:14px;font-size:16px}.cell-control{display:flex;gap:10px;flex-wrap:wrap;margin-top:14px;font-size:16px}.prediction{overflow-wrap:anywhere;padding:12px;background:#f7f8fa}.legend{font-size:15px;color:#58616b}.key-pieces{overflow-wrap:anywhere}.readout-description{margin-top:12px;font-size:16px}#selected-cell{font-weight:bold}.chart-placeholder{padding:18px;border:1px solid #dce1e6;color:#58616b}
@media(max-width:1050px){.explorer{grid-template-columns:1fr}.grid-scroll{max-height:480px}main{padding:24px}}
</style></head><body><div class="layout"><main>
<header><p><a href="index.html">← Back to reviewed results</a> · <a href="kv_retrieval_case.html">Case measurements</a></p>
<p class="eyebrow">Interactive prototype · prefill observations only</p><h1>Figure 5-style Prefill Explorer</h1>
<p>KV Retrieval · <strong>kv_retrieval-359</strong> · 124,405 input tokens · all 32 layers · 202 saved positions.</p>
<p>Correct value: <strong>ca3b3e39-bf38-4633-974a-3a5d1bdc34cd</strong>. The target is a 25-token UUID. This viewer tracks its individual token pieces; their presence does not establish the complete value or its key binding.</p>
<p class="warning"><strong>Rank coverage:</strong> full-vocabulary ranks are available for every token piece of the correct value at all 32 layers and all 202 saved positions. These were recomputed from saved residuals with the fixed J-lens; no new LLM forward, generation, or lens fitting was performed. Arbitrary non-target tokens clicked in the historical Top-10 lists still have partial rank coverage; gray means unknown.</p>
<p id="uniqueness"></p>
</header>
<div class="controls"><label>Method<select id="method"></select></label><label>Saved positions<select id="group"><option value="tail">Prompt tail (128 consecutive positions)</option><option value="kv_evidence">Correct-value source span and neighbors</option><option value="uniform">Uniform context samples</option><option value="prefix">Prompt prefix</option><option value="all">All 202 saved positions</option></select></label><label>Pinned token<select id="token-pin"></select></label><label>Grid labels<select id="grid-label"><option value="top1">Method’s Top-1 readout</option><option value="rank">Pinned-token exact rank</option></select></label></div>
<section class="panel"><h2>Correct-value token sequence</h2>
<div class="controls"><label>Value coverage<select id="value-scope"><option value="full">Full V (25 tokens)</option><option value="half">First 50% (13 of 25 tokens, rounded up)</option></select></label><label>Selected input position<select id="absolute-position"></select></label><button type="button" id="previous-position">← Previous position</button><button type="button" id="next-position">Next position →</button><button type="button" id="prompt-end">Go to prompt end (124404)</button></div>
<p class="note">Choose an input position directly above, or click a column in the layer-position grid below. “Saved positions” filters the available positions. This panel probes all correct-value tokens using the hidden state at one selected input position; it does not average the group. Switching groups keeps your position if available, otherwise selects the group’s last position. Only the 202 saved positions are available.</p>
<p id="value-caption"></p><p class="legend">Columns retain the value’s token order; rows show all 32 layers at the selected absolute prompt position. Cell colors show individual-token ranks, not a joint probability of the value. Click a cell or token to pin it. Repeated vocabulary IDs intentionally repeat the same probe.</p>
<div class="value-scroll"><canvas id="value-grid" aria-label="Correct-value token ranks by layer"></canvas></div>
<p class="note">The table compares all three methods at the selected layer and position. A rank range indicates tied logits; smaller ranks are better. Input counts include every occurrence of that vocabulary token, in keys, values, and other prompt text.</p>
<div class="value-table"><table><thead><tr><th>V token #</th><th>Piece / ID</th><th>Input occurrences</th><th>Dense rank</th><th>Calibrated V+S rank</th><th>FlexPrefill rank</th></tr></thead><tbody id="value-ranks"></tbody></table></div>
<p><a href="assets/figures/kv_full_value_token_ranks.png">Full-V comparison snapshot (all methods, final prompt position)</a> · <a href="data/figure5-prefill-provenance.json" download>Measurement provenance</a></p></section>
<p class="legend">The following grid has absolute prompt positions as columns and block-output layers as rows. Blue intensity encodes the pinned token’s rank (log scale); gray means unknown for non-target pins. Columns in sampled groups need not be adjacent positions. Click a cell to select it.</p>
<div class="explorer"><div><div class="grid-scroll" id="grid-scroll"><canvas id="grid" aria-label="Layer by saved-position readout grid"></canvas></div>
<div class="cell-control"><label>Selected layer <select id="layer"></select></label><label>Position within group <input id="position" type="range" min="0" value="0"></label></div><p class="readout-description" id="selected-cell" aria-live="polite"></p>
<div class="panel"><h2>Pinned-token rank across positions</h2><div class="chart" id="position-chart"></div></div>
<details><summary>Ordered token pieces of the correct value</summary><p class="key-pieces" id="target-pieces"></p><p class="note">Repeated token IDs refer to the same vocabulary direction; separate occurrences cannot be distinguished by pinning that direction.</p></details></div>
<div><div class="panel"><h2>Readout at the selected cell</h2><p id="rank-detail"></p><div class="details-list"><div><h3>Dense Top-10 (historical)</h3><ol id="dense-top"></ol></div><div><h3>Selected method Top-10 (historical)</h3><ol id="method-top"></ol></div></div><p class="note">Top-10 list order and argsort rank can differ for tied logits. Historical Top-10 lists are unchanged. Correct-value ranks were recomputed on V100 with BF16 unembedding and FP32 lens transport; small arithmetic differences from the original GPU run are recorded in the provenance. Tie compatibility of cached reference ranks: Dense 99.943%, Calibrated V+S 99.950%, FlexPrefill 99.957%.</p></div>
<div class="panel"><h2>Pinned-token rank across layers</h2><p class="note">Same absolute position and pinned token; all three methods shown. Every correct-value token has complete rank coverage. Missing data for other pins breaks the line.</p><div class="chart" id="layer-chart"></div></div>
<div class="panel"><h2>Historical output</h2><p id="score"></p><p id="prediction" class="prediction"></p><p class="note">Output text comes from the earlier evaluation. Its generation-time activations are not part of this grid.</p></div></div></div>
<footer>Figure 5-style reading interface, using the same J-lens readout definition. This viewer covers one case, three methods, sampled prefill positions, and all correct-value token probes. The ordered token panel does not measure sequence probability, distinguish repeated token occurrences, or establish K→V binding. The paper’s prompt-plus-generated-output coverage requires additional generation-time activations. No new LLM forward was performed.</footer>
</main></div>
<script type="application/json" id="slice-data">__DATA__</script>
<script>
(() => {
 'use strict';
 const D=JSON.parse(document.getElementById('slice-data').textContent), $=id=>document.getElementById(id);
 const COLORS=['#235b86','#bd7622','#7c497e'];
 const state={method:D.methods[0],group:'tail',pin:D.target_ids[0],layer:31,position:D.input_tokens-1,scope:'full'};
 const index=new Map(D.positions.map((p,i)=>[p,i]));
 const targetIndex=new Map(D.readout_target_ids.map((t,i)=>[t,i]));
 const text=id=>D.vocab[String(id)] || '[token '+id+']';
 function option(select,value,label){const o=document.createElement('option');o.value=value;o.textContent=label;select.appendChild(o);}
 D.methods.forEach((m,i)=>option($('method'),m,D.labels[i]));
 Array.from(new Set(D.target_ids)).forEach(t=>option($('token-pin'),t,JSON.stringify(text(t))+' · token '+t));
 for(let l=0;l<32;l++)option($('layer'),l,l);
 $('layer').value='31';
 $('target-pieces').textContent=D.target_ids.map((t,i)=>(i+1)+': '+JSON.stringify(D.target_pieces[i])+' ['+t+']').join(' · ');
 const U=D.uniqueness;
 $('uniqueness').textContent='“'+D.target_pieces[0]+'” is not unique: its vocabulary token occurs '+U.token_input_occurrences[String(D.target_ids[0])]+' times in the full input, and '+U.first_piece_value_prefix_matches+' of '+U.value_count+' values start with it. The first '+U.half_tokens+'/'+D.target_ids.length+' token prefix '+JSON.stringify(U.half_prefix)+' matches '+U.half_prefix_value_matches+' value; the full correct V matches '+U.full_value_matches+'. Prefix uniqueness was checked against the input values; it is not a measure of hidden-state retrieval.';
 function positions(){return state.group==='all'?D.positions:D.groups[state.group];}
 function rank(method,layer,pos,token=state.pin){const i=index.get(pos),ti=targetIndex.get(token);if(ti!==undefined)return D.target_readouts[method][layer].ranks[i][ti];const ids=D.dense_ids[layer][i],j=ids.indexOf(token);return j<0?null:D.method_data[method].reference_ranks[layer][i][j];}
 function pin(token){state.pin=token;$('token-pin').value=token;render();}
 function valueCount(){return state.scope==='half'?U.half_tokens:D.target_ids.length;}
 function valuePaint(){
  const count=valueCount(),c=$('value-grid'),ctx=c.getContext('2d'),cw=62,ch=25,left=46,top=52;
  c.width=left+count*cw;c.height=top+32*ch;ctx.fillStyle='white';ctx.fillRect(0,0,c.width,c.height);
  ctx.font='13px "Report Times", "Times New Roman", serif';ctx.textAlign='center';ctx.textBaseline='middle';
  for(let j=0;j<count;j++){ctx.fillStyle='#20252b';ctx.fillText('#'+(j+1),left+j*cw+cw/2,14);ctx.fillText(JSON.stringify(D.target_pieces[j]),left+j*cw+cw/2,35);}
  for(let l=0;l<32;l++){ctx.fillStyle='#20252b';ctx.fillText(l,20,top+l*ch+ch/2);for(let j=0;j<count;j++){const r=rank(state.method,l,state.position,D.target_ids[j]),x=left+j*cw,y=top+l*ch;ctx.fillStyle=color(r);ctx.fillRect(x,y,cw-1,ch-1);ctx.fillStyle=r<=10?'white':'#20252b';ctx.fillText(r,x+cw/2,y+ch/2);if(l===state.layer&&D.target_ids[j]===state.pin){ctx.strokeStyle='#cc5c27';ctx.lineWidth=2;ctx.strokeRect(x+1,y+1,cw-3,ch-3);}}}
  $('value-caption').textContent=D.labels[D.methods.indexOf(state.method)]+' · absolute prompt position '+state.position+' · '+count+'/'+D.target_ids.length+' value tokens · selected layer '+state.layer;
  const root=$('value-ranks');root.replaceChildren();for(let j=0;j<count;j++){const t=D.target_ids[j],tr=document.createElement('tr'),n=document.createElement('td'),piece=document.createElement('td'),button=document.createElement('button'),occ=document.createElement('td');n.textContent=j+1;button.textContent=JSON.stringify(D.target_pieces[j])+' ['+t+']';button.onclick=()=>pin(t);piece.appendChild(button);occ.textContent=U.token_input_occurrences[String(t)];tr.append(n,piece,occ);D.methods.forEach(m=>{const td=document.createElement('td'),r=rank(m,state.layer,state.position,t),record=D.target_readouts[m][state.layer],i=index.get(state.position),ti=targetIndex.get(t),best=record.rank_best[i][ti],worst=record.rank_worst[i][ti];td.textContent=String(r)+(best===worst?'':' (tie '+best+'–'+worst+')');tr.appendChild(td);});root.appendChild(tr);}
 }
 function color(r){if(r===null)return '#edf0f3';const a=.12+.84*(1-Math.log(r)/Math.log(D.vocab_size));return 'rgba(35,91,134,'+a+')';}
 function paint(){
  const ps=positions(),c=$('grid'),ctx=c.getContext('2d'),cw=66,ch=25,left=50,top=56;
  c.width=left+ps.length*cw;c.height=top+32*ch;ctx.fillStyle='white';ctx.fillRect(0,0,c.width,c.height);
  ctx.font='13px "Report Times", "Times New Roman", serif';ctx.textAlign='center';ctx.textBaseline='middle';
  ps.forEach((p,j)=>{ctx.fillStyle='#20252b';ctx.fillText(String(p),left+j*cw+cw/2,15);ctx.fillText((D.context[String(p)]||'').replace(/\n/g,'↵').slice(0,7),left+j*cw+cw/2,37);});
  for(let l=0;l<32;l++){
   ctx.fillStyle='#20252b';ctx.fillText(String(l),20,top+l*ch+ch/2);
   ps.forEach((p,j)=>{const r=rank(state.method,l,p),i=index.get(p),x=left+j*cw,y=top+l*ch;ctx.fillStyle=color(r);ctx.fillRect(x,y,cw-1,ch-1);ctx.fillStyle=r!==null&&r<=10?'white':'#20252b';
    const label=$('grid-label').value==='rank'?(r===null?'—':String(r)):text(D.method_data[state.method].top_ids[l][i][0]).replace(/\n/g,'↵').slice(0,7);
    ctx.save();ctx.beginPath();ctx.rect(x,y,cw-1,ch-1);ctx.clip();ctx.fillText(label,x+cw/2,y+ch/2);ctx.restore();
    if(l===state.layer&&p===state.position){ctx.strokeStyle='#cc5c27';ctx.lineWidth=2;ctx.strokeRect(x+1,y+1,cw-3,ch-3);}
   });
  }
 }
 function chart(container,series,axis){
  const w=560,h=220,x0=62,x1=w-14,y0=20,y1=h-44,ps=positions();
  const minimum=axis==='layer'?0:ps[0],maximum=axis==='layer'?31:ps[ps.length-1];
  const xx=v=>x0+(v-minimum)/Math.max(1,maximum-minimum)*(x1-x0),yy=r=>y0+Math.log10(r)/Math.log10(D.vocab_size)*(y1-y0);
  let svg='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 '+w+' '+h+'" role="img" aria-label="Pinned-token full-vocabulary rank chart">';
  for(const tick of [1,10,100,1000,10000,128256]){const y=yy(tick);svg+='<line x1="'+x0+'" x2="'+x1+'" y1="'+y+'" y2="'+y+'" stroke="#e1e6eb"/><text x="'+(x0-7)+'" y="'+(y+4)+'" text-anchor="end" font-size="12">'+tick+'</text>';}
  for(const tick of (axis==='layer'?[0,8,16,24,31]:[minimum,Math.round((minimum+maximum)/2),maximum]))svg+='<text x="'+xx(tick)+'" y="'+(y1+19)+'" text-anchor="middle" font-size="12">'+tick+'</text>';
  svg+='<text x="'+((x0+x1)/2)+'" y="'+(h-5)+'" text-anchor="middle" font-size="13">'+(axis==='layer'?'Layer':'Absolute position')+'</text>';
  let recorded=0;
  series.forEach(s=>{let last=null;for(const p of s.points){if(p.rank===null){last=null;continue;}recorded++;const x=xx(p.x),y=yy(p.rank);if(last)svg+='<line x1="'+last.x+'" y1="'+last.y+'" x2="'+x+'" y2="'+y+'" stroke="'+s.color+'" stroke-width="1.6"/>';svg+='<circle cx="'+x+'" cy="'+y+'" r="3" fill="'+s.color+'"><title>'+s.label+', '+(axis==='layer'?'layer ':'position ')+p.x+': rank '+p.rank+'</title></circle>';last={x,y};}});
  svg+='</svg>';
  $(container).innerHTML=recorded?svg:'<p class="chart-placeholder">No exact ranks recorded for this token in this selection.</p>';
  if(recorded){const legend=document.createElement('p');legend.className='legend';series.forEach((s,i)=>{if(i)legend.appendChild(document.createTextNode(' · '));const item=document.createElement('span');item.style.color=s.color;item.textContent=s.label;legend.appendChild(item);});legend.appendChild(document.createTextNode(' · Lower rank is better; log scale.'));$(container).appendChild(legend);}
 }
 function list(id,ids){const root=$(id);root.replaceChildren();ids.forEach(t=>{const li=document.createElement('li');li.textContent=JSON.stringify(text(t));li.title='Click to pin token '+t;li.tabIndex=0;li.style.cursor='pointer';const choose=()=>{if(!Array.from($('token-pin').options).some(o=>Number(o.value)===t))option($('token-pin'),t,JSON.stringify(text(t))+' · token '+t);state.pin=t;$('token-pin').value=t;render();};li.onclick=choose;li.onkeydown=e=>{if(e.key==='Enter')choose();};root.appendChild(li);});}
 function details(){
  const ps=positions(),j=ps.indexOf(state.position),i=index.get(state.position),m=D.method_data[state.method];
  $('position').max=Math.max(0,ps.length-1);$('position').value=j;$('layer').value=state.layer;
  $('absolute-position').replaceChildren();ps.forEach(p=>option($('absolute-position'),p,p+' · input '+JSON.stringify(D.context[String(p)]||'')));$('absolute-position').value=state.position;
  $('previous-position').disabled=j===0;$('next-position').disabled=j===ps.length-1;
  const r=rank(state.method,state.layer,state.position);
  $('selected-cell').textContent='Layer '+state.layer+' · absolute position '+state.position+' · context '+JSON.stringify(D.context[String(state.position)]||'');
  $('rank-detail').textContent='Pinned '+JSON.stringify(text(state.pin))+' · full-vocabulary rank: '+(r===null?'not recorded':r);
  list('dense-top',D.dense_ids[state.layer][i]);list('method-top',m.top_ids[state.layer][i]);
  chart('layer-chart',D.methods.map((method,mi)=>({label:D.labels[mi],color:COLORS[mi],points:Array.from({length:32},(_,l)=>({x:l,rank:rank(method,l,state.position)}))})),'layer');
  chart('position-chart',[{label:D.labels[D.methods.indexOf(state.method)],color:COLORS[D.methods.indexOf(state.method)],points:ps.map(p=>({x:p,rank:rank(state.method,state.layer,p)}))}],'position');
  $('score').textContent='Historical score: '+m.score+' · output tokens: '+m.output_tokens+' · stop: '+m.stop_reason;
  $('prediction').textContent=m.prediction;
 }
 function render(){paint();details();valuePaint();}
 $('value-scope').onchange=()=>{state.scope=$('value-scope').value;render();};
 $('absolute-position').onchange=()=>{state.position=Number($('absolute-position').value);render();};
 $('previous-position').onclick=()=>{const ps=positions(),j=ps.indexOf(state.position);if(j>0){state.position=ps[j-1];render();}};
 $('next-position').onclick=()=>{const ps=positions(),j=ps.indexOf(state.position);if(j<ps.length-1){state.position=ps[j+1];render();}};
 $('prompt-end').onclick=()=>{state.position=D.input_tokens-1;if(!positions().includes(state.position)){state.group='tail';$('group').value='tail';}render();};
 $('value-grid').onclick=e=>{const c=$('value-grid'),rect=c.getBoundingClientRect(),x=(e.clientX-rect.left)*c.width/rect.width,y=(e.clientY-rect.top)*c.height/rect.height,j=Math.floor((x-46)/62),l=Math.floor((y-52)/25);if(j<0||j>=valueCount()||l<0||l>31)return;state.layer=l;pin(D.target_ids[j]);};
 $('grid').onclick=e=>{const rect=$('grid').getBoundingClientRect(),x=(e.clientX-rect.left)*$('grid').width/rect.width,y=(e.clientY-rect.top)*$('grid').height/rect.height,j=Math.floor((x-50)/66),l=Math.floor((y-56)/25),ps=positions();if(j<0||j>=ps.length||l<0||l>31)return;state.layer=l;state.position=ps[j];render();};
 $('method').onchange=()=>{state.method=$('method').value;render();};
 $('group').onchange=()=>{state.group=$('group').value;const ps=positions();if(!ps.includes(state.position))state.position=ps[ps.length-1];render();};
 $('token-pin').onchange=()=>{state.pin=Number($('token-pin').value);render();};
 $('grid-label').onchange=render;
 $('layer').onchange=()=>{state.layer=Number($('layer').value);render();};
 $('position').oninput=()=>{state.position=positions()[Number($('position').value)];render();};
 document.fonts.ready.then(()=>{render();$('grid-scroll').scrollTop=$('grid').height;$('grid-scroll').scrollLeft=$('grid').width;});
})();
</script></body></html>'''
    html = template.replace('__DATA__', json.dumps(data, ensure_ascii=False).replace('<', '\\u003c'))
    (OUT / 'figure5_prefill.html').write_text(html)
    result = dict(status='PASS',model_execution=False,cases=1,methods=3,layers=32,
                  saved_positions=202,rank_scope=data['rank_scope'],
                  target_first_piece=tok.decode([target]),
                  target_first_piece_final_position_full_rank_trace=full_rank_trace,
                  source_sha256=data['source_sha256'],
                  target_readout_provenance=readout_provenance,
                  uniqueness=data['uniqueness'],
                  target_readout_completeness=complete,
                  legacy_rank_compatibility=compatibility,
                  html_sha256=hashlib.sha256(html.encode()).hexdigest())
    (OUT / 'figure5-prefill-provenance.json').write_text(json.dumps(result, indent=2))
    print('FIGURE 5-STYLE PREFILL PROTOTYPE READY', len(html.encode()), 'bytes')


if __name__ == '__main__':
    main()
