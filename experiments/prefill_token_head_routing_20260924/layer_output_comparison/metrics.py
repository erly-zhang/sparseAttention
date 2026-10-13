"""Tokenwise feature-distribution diagnostics; not attention-probability KL.
Accumulate in bounded chunks; inputs are full o_proj outputs before residual.
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def compare_outputs(dense, candidate, valid, temperature=1.0, chunk_tokens=128):
    if dense.shape != candidate.shape or dense.ndim != 3:
        raise ValueError('outputs must have identical [batch,sequence,hidden] shapes')
    if valid.shape != dense.shape[:2] or valid.dtype != torch.bool:
        raise ValueError('valid must be a boolean [batch,sequence] padding mask')
    if temperature <= 0 or chunk_tokens <= 0:
        raise ValueError('positive temperature/chunk size required')
    d, c, mask = dense.flatten(0, 1), candidate.flatten(0, 1), valid.flatten()
    sums = torch.zeros(6, dtype=torch.float64, device=dense.device)
    for start in range(0, len(d), chunk_tokens):
        keep = mask[start:start+chunk_tokens]
        a = d[start:start+chunk_tokens][keep].float()
        b = c[start:start+chunk_tokens][keep].float()
        if not a.numel():
            continue
        if not bool(torch.isfinite(a).all() & torch.isfinite(b).all()):
            raise ValueError('nonfinite output')
        lp, lq = F.log_softmax(a/temperature, -1), F.log_softmax(b/temperature, -1)
        kl = (lp.exp()*(lp-lq)).sum(-1)
        aa, bb = a.square().sum(-1), b.square().sum(-1)
        both_zero = (aa == 0) & (bb == 0)
        cosine = F.cosine_similarity(a, b, dim=-1, eps=1e-12)
        cosine = torch.where(both_zero, torch.ones_like(cosine), cosine)
        sums += torch.stack((kl.double().sum(), cosine.double().sum(),
                             (a-b).double().square().sum(), a.double().square().sum(),
                             torch.tensor(float(len(a)), device=a.device, dtype=torch.float64),
                             ((aa == 0) | (bb == 0)).double().sum()))
    kl_sum, cos_sum, squared_error, squared_ref, count, zeros = sums.cpu().tolist()
    if not count:
        raise ValueError('no valid tokens')
    # Whole-layer Frobenius relative L2, not mean of per-token ratios.
    relative_l2 = (squared_error/squared_ref)**0.5 if squared_ref else (0.0 if not squared_error else None)
    return dict(valid_tokens=int(count), kl_dense_to_method=kl_sum/count,
                mean_token_cosine=cos_sum/count, relative_l2=relative_l2,
                squared_error=squared_error, squared_reference_norm=squared_ref,
                zero_norm_token_count=int(zeros), temperature=temperature,
                kl_units='nats', kl_axis='output_features',
                reference_zero_norm_policy='relative_l2=null for nonzero error; both-zero cosine=1')
