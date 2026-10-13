import argparse,time,os
from pathlib import Path
import torch
from safetensors import safe_open
from common import HERE,METHODS,TASKS,read,save,verify,sha
from pursuit import pursuit,dense_span_energy

def stat_check(path,expected):
    s=Path(path).stat()
    if s.st_size!=expected['bytes'] or s.st_mtime_ns!=expected['mtime_ns']:
        if not expected.get('sha256') or sha(path)!=expected['sha256']:
            raise ValueError('Frozen input content changed: '+str(path))

def means(value,positions,groups):
    lookup={p:i for i,p in enumerate(positions)}
    result={}
    for g,ps in groups.items():
        if ps:
            vals=value[[lookup[p] for p in ps]];finite=vals[torch.isfinite(vals)]
            if len(finite):result[g]=float(finite.mean())
    return result

@torch.no_grad()
def run(index):
    if not os.environ.get('SLURM_JOB_ID'):raise PermissionError('GPU analysis requires Slurm')
    manifest=verify();layer=index//4;task=TASKS[index%4]
    if not 0<=layer<32:raise ValueError('Invalid layer')
    d=HERE/'outputs'/('worker-'+str(index));d.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    device='cuda';free,_=torch.cuda.mem_get_info()
    if free<12*1024**3:raise RuntimeError('Insufficient free GPU memory')
    gen=torch.Generator().manual_seed(12)
    testatoms=torch.nn.functional.normalize(torch.randn(17,6,generator=gen),dim=-1)
    testx=torch.randn(4,6,generator=gen)
    cpu=pursuit(testx,testatoms,(3,5));gpu=pursuit(testx.cuda(),testatoms.cuda(),(3,5))
    error=float((cpu[5]['reconstruction']-gpu[5]['reconstruction'].cpu()).abs().max())
    if error>1e-4:raise ValueError('CPU/GPU pursuit disagreement')
    save(d/'gpu-check.json',dict(status='PASS',max_abs_error=error,float32=True,tf32=False))
    for p,e in manifest['metadata'].items():stat_check(p,e)
    model=Path(manifest['model']);weightmap=read(model/'model.safetensors.index.json')['weight_map']
    def weight(name):
        with safe_open(str(model/weightmap[name]),framework='pt',device='cpu') as f:return f.get_tensor(name).float().to(device)
    w=weight('lm_head.weight');gamma=weight('model.norm.weight')
    lens=torch.load(manifest['lens'],weights_only=True,map_location='cpu',mmap=True)
    atoms=w*gamma
    if layer!=31:atoms=atoms@lens['J'][layer].to(device).float()
    del w,lens
    norms=atoms.norm(dim=-1,keepdim=True)
    if not torch.isfinite(norms).all() or (norms==0).any():raise ValueError('Invalid dictionary')
    atoms=atoms/norms;del norms
    generator=torch.Generator(device=device).manual_seed(42+layer)
    q,r=torch.linalg.qr(torch.randn(4096,4096,device=device,generator=generator))
    q=q*torch.where(r.diagonal()<0,-1.,1.)[None,:];del r
    bykey={}
    for row in manifest['records']:
        if row['task']==task:bykey.setdefault(row['key'],{})[row['method']]=row
    for num,(key,rows) in enumerate(sorted(bykey.items())):
        out=d/(key+'.json');artifact=d/(key+'.pt')
        if out.exists() and artifact.exists():continue
        start=time.time();hs=[]
        for method in METHODS:
            row=rows[method];stat_check(row['artifact'],dict(bytes=row['artifact_bytes'],mtime_ns=row['artifact_mtime_ns'],sha256=row['artifact_sha256']))
            data=torch.load(row['artifact'],weights_only=True,map_location='cpu',mmap=True)
            if data['positions']!=row['positions']:raise ValueError('Position mismatch')
            hs.append(data['residuals'][layer].float().to(device));del data
        h=torch.cat(hs);n=len(hs[0]);fits=pursuit(h,atoms);random=pursuit(h@q.T,atoms)
        result=dict(key=key,task=task,layer=layer,methods={},seconds=time.time()-start)
        tensorout=dict(positions=rows['dense_full']['positions'],methods=METHODS,k={})
        for k,fit in fits.items():
            rec=fit['reconstruction'].reshape(7,n,4096);rem=fit['remainder'].reshape(7,n,4096)
            sup=fit['support'].reshape(7,n,k);coef=fit['coefficients'].reshape(7,n,k)
            frac=fit['energy_reconstructed'].reshape(7,n);control=random[k]['energy_reconstructed'].reshape(7,n)
            denseactive=(sup[0]>=0)&(coef[0]>1e-7*hs[0].norm(dim=-1)[:,None])
            tensorout['k'][k]={field:fit[field].cpu() for field in ['support','coefficients','energy_reconstructed','active_count']}
            tensorout['k'][k]['random_energy_reconstructed']=random[k]['energy_reconstructed'].cpu()
            tensorout['k'][k]['paired_position_metrics']={}
            for mi,method in enumerate(METHODS):
                delta=hs[mi]-hs[0]
                deltafrac=dense_span_energy(delta,atoms,sup[0],coef[0])
                active=(sup[mi]>=0)&(coef[mi]>1e-7*hs[mi].norm(dim=-1)[:,None])
                inter=((sup[0][:,:,None]==sup[mi][:,None,:])&denseactive[:,:,None]&active[:,None,:]).any(-1).sum(-1)
                recall=inter/denseactive.sum(-1).clamp_min(1)
                recall=torch.where(denseactive.sum(-1)>0,recall,torch.full_like(recall,float('nan')))
                metrics=dict(energy_reconstructed=frac[mi],random_energy_reconstructed=control[mi],
                    excess_energy_reconstructed=frac[mi]-control[mi],active_count=active.sum(-1).float(),
                    reconstruction_relative_l2=(rec[mi]-rec[0]).norm(dim=-1)/rec[0].norm(dim=-1).clamp_min(1e-12),
                    remainder_relative_l2=(rem[mi]-rem[0]).norm(dim=-1)/rem[0].norm(dim=-1).clamp_min(1e-12),
                    dense_active_support_recall=recall,delta_energy_in_dense_active_span=deltafrac)
                tensorout['k'][k]['paired_position_metrics'][method]={name:value.cpu() for name,value in metrics.items()}
                grouped={}
                for name,value in metrics.items():
                    if name=='dense_active_support_recall':
                        lookup={p:i for i,p in enumerate(rows[method]['positions'])}
                        grouped[name]={}
                        for g,ps in rows[method]['position_groups'].items():
                            vals=value[[lookup[p] for p in ps]]
                            finite=vals[torch.isfinite(vals)]
                            if len(finite):grouped[name][g]=float(finite.mean())
                    else:grouped[name]=means(value,rows[method]['positions'],rows[method]['position_groups'])
                result['methods'].setdefault(method,{})[str(k)]=grouped
        tmp=artifact.with_suffix('.tmp');torch.save(tensorout,tmp);tmp.replace(artifact)
        save(out,result);save(d/'progress.json',dict(completed=num+1,total=len(bykey),last_key=key))
        print('COMPLETE',index,key,round(time.time()-start,1),flush=True)
        del fits,random,h,hs,tensorout
    save(d/'complete.json',dict(status='complete',layer=layer,task=task,records=len(bykey)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);run(p.parse_args().index)
