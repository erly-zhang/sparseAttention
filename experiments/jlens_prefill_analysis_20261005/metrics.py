"""Exact full-vocabulary metrics, chunked by position. No display filtering."""
import torch


def ranks_of(logits, targets):
    """Original jlens.vis full-vocabulary argsort ranks, converted to one-based."""
    from jlens.vis import _ranks_of
    return _ranks_of(logits,targets) + 1


def distribution_metrics(dense, sparse, top_k=10):
    dense, sparse = dense.float(), sparse.float()
    if dense.shape != sparse.shape or dense.ndim != 2:
        raise ValueError('Expected matching [positions,vocabulary] logits')
    if not torch.isfinite(dense).all() or not torch.isfinite(sparse).all():
        raise ValueError('Nonfinite readout')
    log_d = dense.log_softmax(-1)
    log_s = sparse.log_softmax(-1)
    log_m = torch.logaddexp(log_d,log_s) - torch.log(torch.tensor(2.,device=dense.device))
    js = .5*((log_d.exp()*(log_d-log_m)).sum(-1) +
             (log_s.exp()*(log_s-log_m)).sum(-1))
    kl = (log_d.exp()*(log_d-log_s)).sum(-1)
    d_ids, s_ids = dense.topk(top_k,dim=-1).indices, sparse.topk(top_k,dim=-1).indices
    sparse_ranks = ranks_of(sparse,d_ids)
    intersection = (d_ids[:,:,None] == s_ids[:,None,:]).any(-1).sum(-1)
    return dict(js_nats=js.clamp_min(0), kl_dense_to_sparse_nats=kl.clamp_min(0),
                dense_top_ids=d_ids, sparse_top_ids=s_ids,
                dense_top_ranks_in_sparse=sparse_ranks,
                top_k_overlap=intersection.float()/top_k,
                dense_top1_rank_in_sparse=sparse_ranks[:,0])


def vector_metrics(dense,sparse):
    dense,sparse=dense.float(),sparse.float()
    if dense.shape != sparse.shape:
        raise ValueError('Mismatched activation shape')
    d=dense.flatten(1)
    s=sparse.flatten(1)
    norms=d.norm(dim=-1)
    return dict(relative_l2=(s-d).norm(dim=-1)/norms.clamp_min(1e-12),
                cosine=torch.nn.functional.cosine_similarity(d,s,dim=-1))


def to_lists(values):
    return {k:v.detach().cpu().tolist() for k,v in values.items()}


@torch.no_grad()
def layer_metrics(model,lens,layer,dense,sparse,top_k=10,chunk_size=16):
    device=model.hf.lm_head.weight.device
    d=dense.float().to(device)
    s=sparse.float().to(device)
    if layer in lens.jacobians:
        # The hook is the block output, exactly the position used by original fit.
        d_transport=lens.transport(d,layer)
        s_transport=lens.transport(s,layer)
    elif layer == model.n_layers-1:
        d_transport,s_transport=d,s  # final row J=I
    else:
        raise ValueError('No fitted Jacobian for layer')
    result={}
    for start in range(0,d.shape[0],chunk_size):
        sl=slice(start,start+chunk_size)
        part=to_lists(distribution_metrics(model.unembed(d_transport[sl]),
                        model.unembed(s_transport[sl]),top_k))
        for key,value in part.items(): result.setdefault(key,[]).extend(value)
    result['residual']=to_lists(vector_metrics(d,s))
    result['transported_residual']=to_lists(vector_metrics(d_transport,s_transport))
    return result
