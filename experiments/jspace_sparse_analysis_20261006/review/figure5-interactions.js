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