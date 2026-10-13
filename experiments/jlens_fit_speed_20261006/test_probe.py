"""Check sampled rows against the original full Jacobian estimator on CPU."""
import importlib.util
from pathlib import Path
import sys
import torch

from benchmark import probe_rows
from jlens.fitting import jacobian_for_prompt


def test_probe_matches_original():
    p=Path(__file__).resolve().parents[3]/'jacobian-lens/tests/tiny.py'
    spec=importlib.util.spec_from_file_location('_jlens_speed_tiny',p)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model=module.TinyDecoder(n_layers=4,d_model=8).requires_grad_(False)
    text='abcdefg '*8
    expected,_,_=jacobian_for_prompt(model,text,[0,1,2],dim_batch=2,max_seq_len=32,skip_first=2)
    actual,record=probe_rows(model,model.encode(text,max_length=32),dim_batch=2,batches=2,skip_first=2)
    for l in expected:
        torch.testing.assert_close(actual[l],expected[l][:4])
    assert record['measured_dimensions']==4
    assert record['full_dimensions']==8
