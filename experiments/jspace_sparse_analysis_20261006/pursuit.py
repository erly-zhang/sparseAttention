"""Unit-atom nonnegative gradient pursuit, batched; no NNLS substitution."""
import torch

@torch.no_grad()
def pursuit(x, atoms, checkpoints=(16,25)):
    if x.ndim!=2 or atoms.ndim!=2 or x.shape[1]!=atoms.shape[1]:raise ValueError('Bad shapes')
    if not torch.isfinite(x).all() or not torch.isfinite(atoms).all():raise ValueError('Nonfinite input')
    n,d=x.shape;kmax=max(checkpoints)
    if kmax>len(atoms):raise ValueError('Too few atoms')
    support=torch.full((n,kmax),-1,device=x.device,dtype=torch.long)
    coefficients=x.new_zeros(n,kmax); selected=x.new_zeros(n,kmax,d)
    residual=x.clone();xnorm=x.norm(dim=-1);out={}
    row=torch.arange(n,device=x.device)
    for step in range(kmax):
        corr=residual@atoms.T
        if step:
            prev=support[:,:step];corr.scatter_(1,prev.clamp_min(0),
                torch.where(prev>=0,torch.full_like(prev,float('-inf'),dtype=x.dtype),corr.gather(1,prev.clamp_min(0))))
        value,index=corr.max(-1);valid=value>1e-7*xnorm
        support[:,step]=torch.where(valid,index,-1)
        selected[:,step]=atoms[index]*valid[:,None]
        active=selected[:,:step+1];c=coefficients[:,:step+1]
        gradient=torch.bmm(active,residual[:,:,None]).squeeze(-1)
        direction=gradient*torch.where(c>0,torch.ones_like(c),(gradient>0).to(c.dtype))
        projected=torch.bmm(direction[:,None,:],active).squeeze(1)
        eta=(projected*residual).sum(-1)/projected.square().sum(-1).clamp_min(1e-30)
        olderror=residual.square().sum(-1)
        accepted=torch.zeros(n,device=x.device,dtype=torch.bool)
        best=c.clone();bestres=residual.clone()
        for _ in range(20):
            candidate=(c+eta[:,None]*direction).clamp_min(0)
            r=x-torch.bmm(candidate[:,None,:],active).squeeze(1)
            take=(r.square().sum(-1)<=olderror+1e-6*olderror.clamp_min(1e-10))&~accepted
            best[take]=candidate[take];bestres[take]=r[take];accepted|=take
            if bool(accepted.all()):break
            eta*=.5
        coefficients[:,:step+1]=best;residual=bestres
        if step+1 in checkpoints:
            reconstruction=x-residual
            energy=x.square().sum(-1)
            out[step+1]=dict(support=support[:,:step+1].clone(),coefficients=best.clone(),
                reconstruction=reconstruction.clone(),remainder=residual.clone(),
                energy_reconstructed=torch.where(energy>0,(1-residual.square().sum(-1)/energy.clamp_min(1e-30)).clamp(0,1),torch.zeros_like(energy)),
                active_count=(best>1e-7*xnorm[:,None]).sum(-1))
    return out

@torch.no_grad()
def dense_span_energy(delta,atoms,support,coefficients):
    """Signed projection of a change onto Dense active directions, not a cone fit."""
    mask=(support>=0)&(coefficients>1e-7)
    a=atoms[support.clamp_min(0)]*mask[:,:,None]
    gram=a@a.transpose(1,2)
    vals,vecs=torch.linalg.eigh(gram)
    cutoff=vals.amax(-1,keepdim=True).clamp_min(1e-30)*1e-6
    inverse=torch.where(vals>cutoff,1/vals.clamp_min(1e-30),0)
    pinv=(vecs*inverse[:,None,:])@vecs.transpose(1,2)
    b=torch.bmm(a,delta[:,:,None]).squeeze(-1)
    pc=torch.bmm(pinv,b[:,:,None]).squeeze(-1)
    p=torch.bmm(pc[:,None,:],a).squeeze(1)
    energy=delta.square().sum(-1)
    fraction=(p.square().sum(-1)/energy.clamp_min(1e-30)).clamp(0,1)
    return torch.where(energy>0,fraction,torch.full_like(fraction,float('nan')))
