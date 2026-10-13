"""Dispatch to unchanged matched Dense or existing AutoBlock sparse kernel."""
import torch
from selection import build_index,sampled_member_coverage
from experiments.token_compacted_sparse import triton_token_compacted_prefill_attention
from experiments.shared_topp_mask_bench_20260930.attention import full_attention as historical_dense


def full_attention(q,k,v,positions,mode,head=None,layer=None,options=None,calibration_callback=None):
    qn,kn,vn=(x.transpose(1,2) for x in (q,k,v))
    if calibration_callback is not None:
        calibration_callback(layer,qn,kn,vn)
    if mode=='dense_full':
        return historical_dense(q,k,v,positions,mode='dense_full')
    expected=torch.arange(q.shape[2],device=q.device)[None]
    if not torch.equal(positions,expected):
        raise ValueError('sparse CSR uses original contiguous token positions')
    index,stats=build_index(qn,kn,vn,head,layer,mode,options)
    # Added coverage diagnostics are explicitly limited to the final 128 queries.
    stats.update(sampled_member_coverage(qn,kn,index))
    output=triton_token_compacted_prefill_attention(qn,kn,vn,index,
        token_chunk_size=options['token_chunk_size'],gqa_interleave=False)
    if not bool(torch.isfinite(output).all()):
        raise ValueError('nonfinite sparse output')
    return output.flatten(-2),None,stats
