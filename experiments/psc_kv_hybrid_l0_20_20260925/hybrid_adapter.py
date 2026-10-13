"""C on zero-based layers0..20; preserve original FA2 forwards on21..31."""
import importlib.util
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]/'psc_kv_layer_outputs_20260925'
spec=importlib.util.spec_from_file_location('psc_original_routing_adapters',BASE/'adapters.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
ROOT=base.ROOT
METHOD='routing_c_l0_20_dense_l21_31'
METHODS=(METHOD,)

def install(model,method=METHOD,groups=None):
    if method!=METHOD:raise ValueError(method)
    layers=model.model.layers
    if len(layers)!=32:raise ValueError('This fixed ablation requires32 layers')
    original=[layer.self_attn.forward for layer in layers]
    model,state=base.install(model,'routing_c')
    for i in range(21,32):
        layers[i].self_attn.forward=original[i]
    state.update(method=method,routing_layers=list(range(21)),dense_layers=list(range(21,32)))
    assert all(layers[i].self_attn.forward==original[i] for i in range(21,32))
    assert all(layers[i].self_attn.forward!=original[i] for i in range(21))
    return model,state
