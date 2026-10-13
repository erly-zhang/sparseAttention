"""CPU-only API check: instantiate tiny models and install, never forward."""
import json,torch
from transformers import LlamaConfig,LlamaForCausalLM
from adapters import install,METHODS
from pathlib import Path
HERE=Path(__file__).resolve().parent
cfg=LlamaConfig(hidden_size=512,intermediate_size=768,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,vocab_size=64)
cfg._attn_implementation='eager'
groups={'layers':{str(i):[{'representative':0,'members':[0,3]},{'representative':1,'members':[1]},{'representative':2,'members':[2]}] for i in range(2)}}
rows=[]
for method in METHODS:
    torch.manual_seed(42)
    model=LlamaForCausalLM(cfg).to(dtype=torch.bfloat16).eval()
    model,state=install(model,method,groups)
    assert len(model.model.layers)==2
    rows.append({'method':method,'installed':True,'forward_executed':False})
(HERE/'cpu_installers.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows))
