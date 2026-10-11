"""Explicit metrics and a nonnegative gradient-pursuit approximation.

The paper does not ship its exact pursuit implementation. This implements
positive-correlation atom selection plus projected, exact-line-search gradient
steps on the active support. All atoms compete globally. It is not a top-logit
projection or an exact solution of the combinatorial sparse-coding problem.
"""
from __future__ import annotations
import torch


def topk_recall(reference, candidate):
    return (reference[:, :, None] == candidate[:, None, :]).any(-1).float().mean(-1)


def js_divergence(logits_a, logits_b):
    la = logits_a.float().log_softmax(-1)
    lb = logits_b.float().log_softmax(-1)
    lm = torch.logaddexp(la, lb) - torch.log(torch.tensor(2.,device=la.device))
    return .5 * ((la.exp()*(la-lm)).sum(-1)+(lb.exp()*(lb-lm)).sum(-1))


def nonnegative_gradient_pursuit(targets, dictionary, k=25, inner_steps=10):
    """targets [n,d], unit-norm dictionary rows [v,d]; <=k positive atoms.

    The reported gain is uncentered reconstruction error reduction, NOT a
    corpus variance estimate. A same-size random dictionary is required for
    the excess-gain metric. Numerical comparisons use FP32 throughout.
    """
    targets = targets.float()
    dictionary = dictionary.float()
    n, d = targets.shape
    k = min(k, dictionary.shape[0])
    ids = torch.zeros(n,k,dtype=torch.long,device=targets.device)
    coefficients = torch.zeros(n,k,device=targets.device)
    reconstruction = torch.zeros_like(targets)
    residual = targets.clone()
    total_energy = targets.square().sum(-1).clamp_min(1e-12)
    gains = []
    for iteration in range(k):
        scores = residual @ dictionary.T
        if iteration:
            scores.scatter_(1,ids[:,:iteration],-float('inf'))
        best_score, chosen = scores.max(-1)
        ids[:,iteration] = chosen
        atoms = dictionary[ids[:,:iteration+1]]
        coeff = coefficients[:,:iteration+1].clone()
        for _ in range(inner_steps):
            gradient = torch.einsum('nkd,nd->nk',atoms,residual)
            gradient.masked_fill_((coeff <= 0) & (gradient < 0),0)
            direction = torch.einsum('nk,nkd->nd',gradient,atoms)
            alpha = (residual*direction).sum(-1)/direction.square().sum(-1).clamp_min(1e-20)
            alpha.clamp_(min=0)
            previous_error = residual.square().sum(-1)
            for _ in range(20):
                proposal = (coeff+alpha[:,None]*gradient).clamp_min(0)
                projected = torch.einsum('nk,nkd->nd',proposal,atoms)
                proposed_residual = targets-projected
                bad = proposed_residual.square().sum(-1)>previous_error+1e-6*total_energy
                if not bool(bad.any()):
                    break
                alpha = torch.where(bad,alpha*.5,alpha)
            good = proposed_residual.square().sum(-1)<=previous_error+1e-6*total_energy
            coeff = torch.where(good[:,None],proposal,coeff)
            reconstruction = torch.einsum('nk,nkd->nd',coeff,atoms)
            residual = targets-reconstruction
        coefficients[:,:iteration+1] = coeff
        gain = 1-residual.square().sum(-1)/total_energy
        gains.append(gain)
    return dict(ids=ids,coefficients=coefficients,reconstruction=reconstruction,
                gains=torch.stack(gains,dim=1),
                nonzero_atoms=(coefficients>1e-6).sum(-1))


def selfcheck():
    d = torch.eye(12)
    h = torch.zeros(3,12)
    h[0,[1,5,9]] = torch.tensor([3.,2.,1.])
    h[1,[0,2]] = torch.tensor([1.,4.])
    h[2,7] = 2.
    got = nonnegative_gradient_pursuit(h,d,k=6)
    torch.testing.assert_close(got['reconstruction'],h)
    assert bool((got['coefficients']>=0).all())
    assert bool((got['gains'][:,1:]>=got['gains'][:,:-1]-1e-6).all())
    torch.testing.assert_close(js_divergence(h,h),torch.zeros(3),atol=1e-7,rtol=0)
    torch.testing.assert_close(topk_recall(torch.tensor([[1,2]]),torch.tensor([[2,3]])),torch.tensor([.5]))
    print('PASS: pursuit support, positivity, reconstruction, monotone gain; JS and recall')


if __name__ == '__main__':
    selfcheck()
