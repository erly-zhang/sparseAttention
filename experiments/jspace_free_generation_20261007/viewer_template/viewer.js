'use strict';
// A new checkpoint-aware viewer, inspired by the paper's Figure 5.
// Ranks are exact full-vocabulary tie intervals, not ranks among displayed words.
window.FIGURE5_CASES={};window.FIGURE5_RANKS={};
const $=id=>document.getElementById(id), colors=['#197985','#b36a31','#845a9c','#69833f'];
const names={dense:'Dense',full_random:'Random · Full KV',full_w1:'W1 · Full KV',compact_random:'Random · Compact KV',compact_w1:'W1 · Compact KV'};
const state={textKey:null,data:null,arm:4,point:0,layer:17,pins:[],heat:null,visible:[],ticket:0,hoverTicket:0,rankCache:new Map(),scripts:new Map()};
window.FIGURE5_STATE=state;
const escape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const clean=s=>s.replace(/\n/g,'↵').replace(/\t/g,'⇥');
const rankText=(lo,hi)=>lo===hi?lo.toLocaleString():`${lo.toLocaleString()}–${hi.toLocaleString()}`;
const status=(message,error=false)=>{$('status').textContent=message;$('status').classList.toggle('error',error);};
const hideTooltip=()=>{state.hoverTicket++;$('tooltip').hidden=true;};
const cellRankText=(lo,hi)=>lo===hi?String(lo):`${lo}–${hi}`;

function script(src){
  if(state.scripts.has(src))return state.scripts.get(src);
  const p=new Promise((resolve,reject)=>{const s=document.createElement('script');s.src=src;s.onload=resolve;s.onerror=()=>reject(new Error(`Failed to load local file: ${src}. Extract the complete folder, then open index.html.`));document.head.appendChild(s);});
  state.scripts.set(src,p);return p;
}
async function ranks(tid){
  const d=state.data,key=`${d.idx}/${tid}`;
  if(state.rankCache.has(key))return state.rankCache.get(key);
  const promise=(async()=>{
    await script(`ranks_${d.idx}/${d.rank_chunks[tid]}.js`);
    if(!window.DecompressionStream)throw new Error('Local decompression is unsupported. Use a recent Chrome, Edge, or Firefox browser.');
    const encoded=window.FIGURE5_RANKS[key];
    const bytes=Uint8Array.from(atob(encoded),c=>c.charCodeAt(0));
    const buf=new Uint8Array(await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
    const out=new Uint32Array(5*d.P*36*2);let ptr=0;
    const read=()=>{let value=0,shift=0,b;do{if(ptr>=buf.length)throw new Error('Corrupt rank data');b=buf[ptr++];value|=(b&127)<<shift;shift+=7;}while(b&128);return value;};
    const cells=d.P*36;
    for(let c=0;c<cells;c++){const lo=read(),width=read();out[c*2]=lo;out[c*2+1]=lo+width;}
    for(let a=1;a<5;a++)for(let c=0;c<cells;c++){
      const zlo=read(),zw=read(),dlo=(zlo>>>1)^-(zlo&1),dw=(zw>>>1)^-(zw&1);
      const lo=out[c*2]+dlo,hi=lo+(out[c*2+1]-out[c*2])+dw;
      if(lo<1||hi<lo||hi>d.vocab_size)throw new Error('Invalid rank bounds');
      out[(a*cells+c)*2]=lo;out[(a*cells+c)*2+1]=hi;
    }
    if(ptr!==buf.length)throw new Error('Invalid rank data length');return out;
  })();state.rankCache.set(key,promise);return promise;
}
const labelsFor=(a=state.arm)=>state.data.labels_by_arm[a];
const offset=(a,p,l,k=0)=>((a*state.data.P+p)*36+l)*state.data.N+k;
const interval=(array,a,p,l)=>{const i=((a*state.data.P+p)*36+l)*2;return [array[i],array[i+1]];};
const sparseAt=(arm,label)=>arm.startsWith('compact')?label.compact_eviction_visible:arm.startsWith('full')?label.full_selection_active:false;
function tokenButton(tid,top=false){
  const i=state.pins.indexOf(tid),d=state.data;
  return `<button class="token${top?' top':''}${i>=0?' pinned':''}" style="--pin:${colors[i]||'#aaa'}" data-pin="${tid}" title="token #${tid} · ${escape(d.vocab[tid])} · Click to pin / unpin">${escape(clean(d.vocab[tid]))}</button>`;
}
function wordRow(a,p,l,n=4){return Array.from({length:n},(_,k)=>tokenButton(state.data.top_ids[offset(a,p,l,k)],k===0)).join('');}
function setPins(){
  const d=state.data;
  $('pins').innerHTML=state.pins.length?state.pins.map((t,i)=>`<button class="pin" data-pin="${t}"><b style="background:${colors[i]}"></b>${escape(clean(d.vocab[t]))}<span class="muted"> #${t} ×</span></button>`).join(''):'<span class="muted">Click any readout token on the right to track its rank.</span>';
  if(!state.pins.includes(state.heat))state.heat=state.pins[0]??null;
  $('heatToken').innerHTML=state.pins.length?state.pins.map(t=>`<option value="${t}">${escape(clean(d.vocab[t]))} · #${t}</option>`).join(''):'<option value="">Pin a token</option>';
  if(state.heat!==null)$('heatToken').value=state.heat;
  $('heatToken').disabled=!state.pins.length;
}
function search(){
  if(!state.data)return;
  const d=state.data,q=$('search').value.trim().toLowerCase();
  const ids=d.tracked.filter(t=>d.vocab[t].toLowerCase().includes(q)||String(t)===q).sort((a,b)=>(d.vocab[a].trim().toLowerCase()===q?-1:0)-(d.vocab[b].trim().toLowerCase()===q?-1:0)||a-b).slice(0,150);
  $('searchResults').innerHTML=ids.length?ids.map(t=>`<option value="${t}">${escape(clean(d.vocab[t]))} · #${t}</option>`).join(''):'<option value="">No matching token</option>';
  $('addPin').disabled=!ids.length;
}
function select(p,l,scroll=false){state.point=p;state.layer=l;$('position').value=p;$('layer').value=l;render();if(scroll)scrollToPoint();}
function scrollToPoint(){const c=state.visible.indexOf(state.point);if(c<0)return;for(const id of ['topScroll','rankScroll']){const el=$(id);el.scrollLeft=Math.max(0,c*84-el.clientWidth/2+42);}}
function visibility(){
  state.visible=labelsFor().map((r,i)=>i).filter(i=>(!$('reasoningOnly').checked||labelsFor()[i].reasoning)&&(!$('commonOnly').checked||labelsFor()[i].same_token_prefix_as_dense));
  if(!state.visible.includes(state.point))state.point=state.visible.find(i=>sparseAt(state.data.arms[state.arm],labelsFor()[i]))??state.visible[0];
  $('position').innerHTML=state.visible.map(i=>`<option value="${i}">step ${labelsFor()[i].step} · abs ${labelsFor()[i].absolute_position}</option>`).join('');
  $('position').value=state.point;
}
function axes(){const html=Array.from({length:36},(_,i)=>`<div>${35-i}</div>`).join('');$('topAxis').innerHTML=html;$('rankAxis').innerHTML=html;}
function canvasSetup(id){
  const canvas=$(id),w=state.visible.length*84,h=37+36*18,ratio=window.devicePixelRatio||1;
  canvas.width=Math.round(w*ratio);canvas.height=Math.round(h*ratio);canvas.style.width=w+'px';canvas.style.height=h+'px';
  const ctx=canvas.getContext('2d');ctx.scale(ratio,ratio);ctx.font='10px Georgia, Times New Roman, serif';ctx.textBaseline='middle';return ctx;
}
function heatColor(rank){const t=Math.min(1,Math.max(0,Math.log(rank)/Math.log(state.data.vocab_size)));const mix=(a,b)=>Math.round(a+(b-a)*t);return `rgb(${mix(19,246)},${mix(95,245)},${mix(103,238)})`;}
function heatTextColor(rank){
  const t=Math.min(1,Math.max(0,Math.log(rank)/Math.log(state.data.vocab_size)));
  const linear=value=>{const s=value/255;return s<=.04045?s/12.92:((s+.055)/1.055)**2.4;};
  const rgb=[[19,246],[95,245],[103,238]].map(([a,b])=>linear(Math.round(a+(b-a)*t)));
  const luminance=.2126*rgb[0]+.7152*rgb[1]+.0722*rgb[2];
  return luminance<.179?'#ffffff':'#111111';
}
function drawGrid(id,array=null){
  const d=state.data,a=state.arm,ctx=canvasSetup(id),isHeat=id==='rankGrid';
  if(isHeat){
    const ready=Boolean(array),empty=state.heat===null;
    $('rankGrid').dataset.rankState=ready?'ready':empty?'empty':'loading';
    $('rankEmpty').hidden=ready;
    $('rankEmpty').querySelector('strong').textContent=empty?'No token selected':'Loading pinned-token ranks…';
    $('rankEmpty').querySelector('span').textContent=empty?'Click any readout token to pin it, then choose it in the Heatmap token menu.':'Rank numbers will appear after the local rank data are loaded.';
    $('pinCurrentTop').hidden=!empty;
    $('rankState').textContent=empty?'No token selected. The grid does not contain rank measurements yet.':`${clean(d.vocab[state.heat])} #${state.heat} · ${names[d.arms[a]]} · ${ready?'Exact ranks in each cell; smaller = higher rank; range = tie.':'Loading ranks…'}`;
    $('rankGrid').setAttribute('aria-label',empty?'No pinned token; rank grid is empty':`Full-vocabulary ranks of ${clean(d.vocab[state.heat])}, token ${state.heat}, ${names[d.arms[a]]}`);
  }
  ctx.fillStyle='#fffefb';ctx.fillRect(0,0,state.visible.length*84,685);
  for(let col=0;col<state.visible.length;col++){
    const p=state.visible[col],label=labelsFor()[p],active=sparseAt(d.arms[a],label),x=col*84;
    ctx.fillStyle=active?'#d8e9e4':'#edece7';ctx.fillRect(x,0,83,36);
    ctx.fillStyle='#3b5155';ctx.font='10px system-ui';ctx.fillText(String(label.step),x+6,11);ctx.fillStyle='#7a8589';ctx.font='8px system-ui';ctx.fillText('abs '+label.absolute_position,x+6,25);if(!label.same_token_prefix_as_dense){ctx.fillStyle='#a75d20';ctx.font='11px system-ui';ctx.fillText('Δ',x+70,11);}
    for(let l=0;l<36;l++){
      const y=37+(35-l)*18,i=offset(a,p,l),tid=d.top_ids[i],different=tid!==d.top_ids[offset(0,p,l)];
      if(isHeat){
        if(array){const [lo,hi]=interval(array,a,p,l);ctx.fillStyle=heatColor(Math.sqrt(lo*hi));}else ctx.fillStyle='#f0f0e9';
      }else ctx.fillStyle=different&&a!==0?'#f8e2ce':active||a===0?'#fffefb':'#f1f1ed';
      ctx.fillRect(x,y,83,17);
      if(isHeat&&array){
        const [lo,hi]=interval(array,a,p,l),rank=Math.sqrt(lo*hi),text=cellRankText(lo,hi);
        ctx.fillStyle=heatTextColor(rank);ctx.font='9px ui-monospace, Consolas, monospace';
        ctx.save();ctx.beginPath();ctx.rect(x+2,y,79,17);ctx.clip();ctx.textAlign='center';ctx.fillText(text,x+41.5,y+8.5);ctx.restore();
      }
      if(!isHeat){
        ctx.fillStyle='#2e3738';ctx.font=(l===state.layer?'bold ':'')+'10px Georgia';let word=clean(d.vocab[tid]);if(word.length>13)word=word.slice(0,12)+'…';
        ctx.save();ctx.beginPath();ctx.rect(x,y,83,17);ctx.clip();ctx.fillText(word,x+3,y+8);ctx.restore();
        const pi=state.pins.indexOf(tid);if(pi>=0){ctx.fillStyle=colors[pi];ctx.fillRect(x,y+15,83,2);}
      }
      if(p===state.point&&l===state.layer){ctx.strokeStyle='#a75d20';ctx.lineWidth=2;ctx.strokeRect(x+1,y+1,81,15);}
      else if(p===state.point||l===state.layer){ctx.strokeStyle='#bda77d88';ctx.lineWidth=.7;ctx.strokeRect(x+.5,y+.5,82,16);}
    }
  }
}
function drawReadouts(){
  const d=state.data,a=state.arm,p=state.point,l=state.layer;
  $('layerSubtitle').textContent=`decode step ${labelsFor()[p].step} · ${names[d.arms[a]]}`;
  $('positionSubtitle').textContent=`Block ${l}${l===35?' · Final model output':''} · ${state.visible.length} captured positions`;
  $('layerReadouts').innerHTML=Array.from({length:36},(_,r)=>35-r).map(layer=>`<div class="readrow${layer===l?' active':''}" data-layer-row="${layer}"><button class="coord" data-layer="${layer}">L${layer}</button><div class="tokenrow">${wordRow(a,p,layer)}</div></div>`).join('');
  $('positionReadouts').innerHTML=state.visible.map(point=>`<div class="readrow${point===p?' active':''}" data-point-row="${point}"><button class="coord" data-point="${point}">${labelsFor()[point].step}</button><div class="tokenrow">${wordRow(a,point,l)}</div></div>`).join('');
  for(const id of ['layerReadouts','positionReadouts']){const el=$(id),active=el.querySelector('.active');if(active)el.scrollTop=Math.max(0,active.offsetTop-el.offsetTop-el.clientHeight/2+active.offsetHeight/2);}
  const metric=d.metrics[a][p*36+l],same=labelsFor()[p].same_token_prefix_as_dense;
  $('cellMetrics').innerHTML=`<span>step <b>${labelsFor()[p].step}</b> · Block <b>${l}</b></span><span>Recall@25 vs Dense at same step <b>${(100*metric[0]).toFixed(1)}%</b></span><span>JS <b>${metric[1].toFixed(5)}</b> nats</span><span class="comparison-scope">${same?'Same token prefix':'Different token prefixes · mixed context effects'}</span>`;
  let rows='';
  for(let k=0;k<d.N;k++){const i=offset(a,p,l,k),b=offset(0,p,l,k);rows+=`<tr><td>${tokenButton(d.top_ids[i])}</td><td>${rankText(d.top_low[i],d.top_high[i])}</td><td>${(100*d.top_probs[i]).toPrecision(3)}%</td><td>${tokenButton(d.top_ids[b])}</td><td>${rankText(d.top_low[b],d.top_high[b])}</td></tr>`;}
  $('cellReadout').innerHTML=`<table><thead><tr><th>${escape(names[d.arms[a]])}</th><th>Full-vocabulary rank</th><th>Probability</th><th>Dense</th><th>Rank</th></tr></thead><tbody>${rows}</tbody></table>`;
}
function renderGeneration(){
  const d=state.data,a=state.arm,b=d.arms.indexOf($('compareMethod').value),current=d.trajectory_stats[a];
  $('runSummary').innerHTML='<table><thead><tr><th>Method</th><th>Generated tokens</th><th>Stop</th><th>First different token vs Dense</th><th>Think closes at token</th><th>Final boxed answer</th></tr></thead><tbody>'+d.trajectory_stats.map((r,i)=>`<tr class="${i===a?'current':''}"><td>${escape(names[r.arm])}</td><td>${r.generated_tokens.toLocaleString()}</td><td>Token cap</td><td>${r.first_different_generated_token??'—'}</td><td>${r.think_close_token??'Not closed'}</td><td>${r.final_boxed_answer===null?'Not completed':escape(r.final_boxed_answer)}</td></tr>`).join('')+'</tbody></table>';
  const key=`${d.idx}/${a}/${b}`;
  if(state.textKey===key)return;
  state.textKey=key;
  $('textLeftTitle').textContent=names[d.arms[a]];
  $('textRightTitle').textContent=names[d.arms[b]];
  $('textLeft').textContent=d.continuations[a];$('textRight').textContent=d.continuations[b];
  $('continuation').textContent=d.continuations[a];
  $('continuationTitle').textContent=`${names[d.arms[a]]} · Full own-generated continuation`;
  $('rawTextLink').href=`raw/sample_${d.idx}/${d.arms[a]}.txt`;
  $('rawTextLink').download=`hmmt_${d.idx}_${d.arms[a]}.txt`;
  const pair=d.pairs.find(r=>(r.arm_a===d.arms[a]&&r.arm_b===d.arms[b])||(r.arm_a===d.arms[b]&&r.arm_b===d.arms[a]));
  if(!pair){
    $('pairInfo').textContent='The same method is selected on both sides; both panels show the identical trace.';
    $('diffLeft').textContent='Select a different comparison method to inspect the first differing token.';
    $('diffRight').textContent='';return;
  }
  $('pairInfo').textContent=`These traces share their first ${pair.common_prefix_tokens.toLocaleString()} generated tokens. First different output token: ${pair.first_different_generated_token??'none within the shared length'} (one-based). The marked snippets below show the first differing token; the full text panels contain the complete saved continuations.`;
  for(const [id,arm] of [['diffLeft',d.arms[a]],['diffRight',d.arms[b]]]){
    const snippet=pair.snippets[arm];
    $(id).innerHTML=snippet?escape(snippet.before)+`<mark>${escape(clean(snippet.different_token))}</mark>`+escape(snippet.after):'No differing token in the observed overlap.';
  }
}
function chart(id,byLayer,arrays){
  const d=state.data,a=state.arm,w=360,h=200,left=46,right=12,top=15,bottom=28;
  const xmin=0,xmax=byLayer?35:4096,x=v=>left+(v-xmin)/(xmax-xmin)*(w-left-right),y=v=>top+Math.log(Math.max(1,v))/Math.log(d.vocab_size)*(h-top-bottom);
  let svg=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${byLayer?'Across-layer ':'Across-position '}token ranks"><rect width="360" height="200" fill="#fffefb"/>`;
  if(!byLayer&&d.arms[a]!=='dense'){
    const first=labelsFor().find(label=>sparseAt(d.arms[a],label));if(first){const start=x(first.step);svg+=`<rect x="${start}" y="${top}" width="${w-right-start}" height="${h-top-bottom}" fill="#edf2ee"/>`;}
  }
  for(const rank of [1,10,100,1000,10000,d.vocab_size]){const yy=y(rank);svg+=`<line x1="${left}" y1="${yy}" x2="${w-right}" y2="${yy}" stroke="#e2e4df"/><text x="${left-6}" y="${yy+3}" text-anchor="end" fill="#68737b">${rank===d.vocab_size?'Vocab':rank>=1000?rank/1000+'k':rank}</text>`;}
  for(const tick of byLayer?[0,8,17,26,35]:[0,1024,2048,3072,4096])svg+=`<text x="${x(tick)}" y="${h-8}" text-anchor="middle" fill="#68737b">${tick}</text>`;
  const selected=x(byLayer?state.layer:labelsFor()[state.point].step);svg+=`<line x1="${selected}" y1="${top}" x2="${selected}" y2="${h-bottom}" stroke="#bc7838" stroke-dasharray="3 3"/>`;
  state.pins.forEach((tid,index)=>{
    const array=arrays[index],points=byLayer?Array.from({length:36},(_,l)=>[state.point,l]):state.visible.map(p=>[p,state.layer]);
    const series=arm=>points.map(([p,l])=>{const [lo,hi]=interval(array,arm,p,l);return [x(byLayer?l:labelsFor()[p].step),y(Math.sqrt(lo*hi)),p,l,lo,hi];});
    const poly=points=>points.map(p=>p.slice(0,2).join(',')).join(' ');
    if($('baseline').checked&&a!==0)svg+=`<polyline fill="none" stroke="${colors[index]}" stroke-opacity=".45" stroke-width="1" stroke-dasharray="4 3" points="${poly(series(0))}"/>`;
    const active=series(a);svg+=`<polyline fill="none" stroke="${colors[index]}" stroke-width="1.5" points="${poly(active)}"/>`;
    for(const [xx,yy,p,l,lo,hi] of active)svg+=`<circle cx="${xx}" cy="${yy}" r="2.4" fill="${colors[index]}" data-point="${p}" data-layer="${l}" style="cursor:pointer"><title>${escape(d.vocab[tid])} #${tid} · step ${labelsFor()[p].step} · L${l} · rank ${rankText(lo,hi)}</title></circle>`;
  });
  if(!state.pins.length)svg+='<text x="180" y="85" text-anchor="middle" fill="#68737b">Pin tokens to display trajectories</text>';
  $(id).innerHTML=svg+'</svg>';
}
async function render(){
  if(!state.data)return;
  hideTooltip();
  const ticket=++state.ticket,d=state.data,a=state.arm,p=state.point,label=labelsFor()[p];
  renderGeneration();
  setPins();drawGrid('topGrid');drawGrid('rankGrid');drawReadouts();
  $('position').value=p;$('layer').value=state.layer;
  const snippet=d.snippets_by_arm[a][p];$('context').innerHTML=escape(snippet[0])+`<mark>${escape(clean(snippet[1]).replace(/^ +$/,'␣'))}</mark>`+`<span class="later-text">${escape(snippet[2])}</span>`;
  $('prefixNotice').textContent=label.same_token_prefix_as_dense?'Same token prefix as Dense at this checkpoint.':'Different token prefix from Dense: readout differences combine attention and changed text history.';
  $('positionInfo').textContent=`decode step ${label.step} · absolute position ${label.absolute_position} · ${label.reasoning?'reasoning':'prefill / non-reasoning'} · ${names[d.arms[a]]} · ${a===0?'Dense full history':sparseAt(d.arms[a],label)?'Sparse phase active':'Before sparse phase'}`;
  $('previous').disabled=state.visible.indexOf(p)===0;$('next').disabled=state.visible.indexOf(p)===state.visible.length-1;
  status(`Sample ${d.idx} · ${names[d.arms[a]]} · ${state.visible.length}/${d.P} checkpoints · 36 blocks · ${d.tracked.length.toLocaleString()} trackable tokens · Preparing ranks…`);
  try{
    const arrays=await Promise.all(state.pins.map(ranks));if(ticket!==state.ticket)return;
    const h=state.pins.indexOf(state.heat);drawGrid('rankGrid',h>=0?arrays[h]:null);chart('layerChart',true,arrays);chart('positionChart',false,arrays);
    status(`Sample ${d.idx} · ${names[d.arms[a]]} · ${state.visible.length}/${d.P} checkpoints · 36 blocks · ${d.tracked.length.toLocaleString()} trackable tokens · Ranks ready`);
  }catch(error){if(ticket===state.ticket)status(error.message,true);}
}
async function loadCase(idx){
  status('Loading local sample…');$('sample').disabled=true;
  try{
    await script(`case_${idx}.js`);state.data=window.FIGURE5_CASES[idx];
    state.pins=state.data.pins.slice(0,3);state.heat=state.pins[0];state.point=labelsFor().findIndex(l=>l.step===2560);if(state.point<0)state.point=0;
    $('prompt').textContent=state.data.prompt;state.textKey=null;
    $('compareMethod').innerHTML=state.data.arms.map(a=>`<option value="${a}">${names[a]}</option>`).join('');$('compareMethod').value='dense';
    $('promptCount').textContent=`· ${state.data.prompt_tokens} tokens`;
    $('provenance').textContent=`input SHA256 ${state.data.input_ids_sha256.slice(0,16)}…`;
    $('layer').innerHTML=Array.from({length:36},(_,l)=>`<option value="${l}">Block ${l}${l===35?' · Model output':''}</option>`).join('');
    visibility();axes();search();await render();scrollToPoint();
    document.querySelector(`[data-layer-row="${state.layer}"]`)?.scrollIntoView({block:'nearest'});
    document.querySelector(`[data-point-row="${state.point}"]`)?.scrollIntoView({block:'nearest'});
    window.scrollTo(0,0);
  }catch(error){status(error.message,true);}finally{$('sample').disabled=false;}
}
function togglePin(tid){if(!state.data.tracked.includes(tid))return;const i=state.pins.indexOf(tid);if(i>=0)state.pins.splice(i,1);else{if(state.pins.length===4){status('Four tokens are pinned. Unpin one before adding another.');return;}state.pins.push(tid);}render();}
document.addEventListener('click',event=>{
  const target=event.target.closest('[data-pin],[data-point],[data-layer]');if(!target)return;
  if(target.hasAttribute('data-pin'))togglePin(Number(target.dataset.pin));else select(target.hasAttribute('data-point')?Number(target.dataset.point):state.point,target.hasAttribute('data-layer')?Number(target.dataset.layer):state.layer,true);
});
$('sample').addEventListener('change',()=>loadCase(Number($('sample').value)));
$('arm').addEventListener('change',()=>{state.arm=state.data.arms.indexOf($('arm').value);visibility();render();scrollToPoint();});
$('position').addEventListener('change',()=>select(Number($('position').value),state.layer,true));
$('layer').addEventListener('change',()=>select(state.point,Number($('layer').value)));
$('previous').addEventListener('click',()=>select(state.visible[state.visible.indexOf(state.point)-1],state.layer,true));
$('next').addEventListener('click',()=>select(state.visible[state.visible.indexOf(state.point)+1],state.layer,true));
$('reasoningOnly').addEventListener('change',()=>{visibility();render();scrollToPoint();});
$('commonOnly').addEventListener('change',()=>{visibility();render();scrollToPoint();});
$('compareMethod').addEventListener('change',render);
$('search').addEventListener('input',search);
$('addPin').addEventListener('click',()=>{const val=$('searchResults').value;if(val!=='')togglePin(Number(val));});
$('clearPins').addEventListener('click',()=>{state.pins=[];render();});
$('pinCurrentTop').addEventListener('click',()=>togglePin(state.data.top_ids[offset(state.arm,state.point,state.layer)]));
$('baseline').addEventListener('change',render);
$('heatToken').addEventListener('change',()=>{state.heat=Number($('heatToken').value);render();});
for(const id of ['topGrid','rankGrid']){
  const canvas=$(id);
  function cell(event){const rect=canvas.getBoundingClientRect(),col=Math.floor((event.clientX-rect.left)/84),layer=35-Math.floor((event.clientY-rect.top-37)/18);return col>=0&&col<state.visible.length&&layer>=0&&layer<36?[state.visible[col],layer]:null;}
  canvas.addEventListener('click',event=>{const c=cell(event);if(c)select(...c);});
  canvas.addEventListener('mousemove',async event=>{
    const c=cell(event);if(!c){hideTooltip();return;}const [p,l]=c,d=state.data,a=state.arm,i=offset(a,p,l),label=labelsFor()[p],heat=state.heat,hover=++state.hoverTicket;
    let text=`step ${label.step} · abs ${label.absolute_position} · L${l}\n${label.same_token_prefix_as_dense?'Same prefix as Dense':'Different own prefix from Dense'}\n`;
    if(id==='rankGrid'){
      if(heat===null)text+='No token selected. Pin a readout token to view its ranks.';
      else{const array=await ranks(heat);if(hover!==state.hoverTicket)return;const [lo,hi]=interval(array,a,p,l);text+=`${clean(d.vocab[heat])} #${heat}\nFull-vocabulary rank ${rankText(lo,hi)}`;}
    }
    else text+=`${clean(d.vocab[d.top_ids[i]])} #${d.top_ids[i]}\nFull-vocabulary rank ${rankText(d.top_low[i],d.top_high[i])}\nDense: ${clean(d.vocab[d.top_ids[offset(0,p,l)]])}`;
    if(hover!==state.hoverTicket)return;
    const tip=$('tooltip');tip.textContent=text;tip.hidden=false;tip.style.left=Math.min(event.clientX+14,window.innerWidth-345)+'px';tip.style.top=Math.min(event.clientY+14,window.innerHeight-110)+'px';
  });canvas.addEventListener('mouseleave',hideTooltip);
}
let sync=false;for(const [a,b] of [['topScroll','rankScroll'],['rankScroll','topScroll']])$(a).addEventListener('scroll',()=>{hideTooltip();if(sync)return;sync=true;$(b).scrollLeft=$(a).scrollLeft;sync=false;});
loadCase(0);
