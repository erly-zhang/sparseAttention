"""Original jlens.fit, on an unpatched, frozen Dense Llama model."""
import argparse
import gc
from common import HERE,config,read,save,sha,require_approval,runtime,load_model,Llama44LensModel,atomic_torch_save
from progress import PeriodicSaver


def fit_shard(shard):
    require_approval()
    import torch
    import jlens
    runtime()
    c=config()
    if not 0 <= shard < c['fit']['shards']: raise ValueError('Invalid shard')
    rows=read(HERE/'fit-prompts.json')[shard::c['fit']['shards']]
    directory=HERE/'outputs'/('fit-'+str(shard))
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'complete.json').exists():
        complete=read(directory/'complete.json')
        if complete['lens_sha256'] != sha(directory/'lens.pt'): raise ValueError('Shard changed')
        return
    with PeriodicSaver(directory):
        try:
            model,tokenizer=load_model()
            wrapper=Llama44LensModel(model,tokenizer)
            for row in rows:
                actual=wrapper.encode(row['text'],max_length=c['fit']['max_seq_len'])[0].tolist()
                if actual != row['input_ids']: raise ValueError('Fitting tokenizer changed')
            jlens.configure_logging()
            lens=jlens.fit(wrapper,[row['text'] for row in rows],
                source_layers=c['fit']['source_layers'],target_layer=c['fit']['target_layer'],
                dim_batch=c['fit']['dim_batch'],max_seq_len=c['fit']['max_seq_len'],
                skip_first=c['fit']['skip_first'],checkpoint_path=str(directory/'checkpoint.pt'),
                checkpoint_every=c['fit']['checkpoint_every'],resume=True)
            if lens.n_prompts != len(rows): raise ValueError('Fitting skipped prompts')
            if any(not torch.isfinite(j).all() for j in lens.jacobians.values()):
                raise ValueError('Nonfinite Jacobian')
            temp=directory/'lens.tmp.pt'
            lens.save(str(temp),dtype=torch.float32)
            temp.replace(directory/'lens.pt')
            save(directory/'complete.json',dict(status='complete',shard=shard,n_prompts=lens.n_prompts,
                 lens_sha256=sha(directory/'lens.pt'),config_sha256=sha(HERE/'config.json')))
        except Exception as exc:
            save(directory/'failure.json',dict(error=repr(exc)))
            raise


def merge():
    require_approval()
    import torch
    from jlens import JacobianLens
    c=config()
    total=0
    sums=None
    for shard in range(c['fit']['shards']):
        directory=HERE/'outputs'/('fit-'+str(shard))
        complete=read(directory/'complete.json')
        if complete['lens_sha256'] != sha(directory/'lens.pt'): raise ValueError('Invalid fit shard')
        lens=JacobianLens.load(str(directory/'lens.pt'))
        if lens.source_layers != c['fit']['source_layers'] or lens.d_model != 4096:
            raise ValueError('Incompatible fitted lens')
        if sums is None: sums={l:torch.zeros_like(j) for l,j in lens.jacobians.items()}
        for l,j in lens.jacobians.items(): sums[l].add_(j,alpha=lens.n_prompts)
        total += lens.n_prompts
        del lens
        gc.collect()
    if total != c['fit']['prompts']: raise ValueError('Wrong total fit count')
    for j in sums.values(): j.div_(total)
    lens=JacobianLens(sums,n_prompts=total,d_model=4096)
    lens.save(str(HERE/'outputs/lens.tmp.pt'),dtype=torch.float32)
    (HERE/'outputs/lens.tmp.pt').replace(HERE/'outputs/lens.pt')
    save(HERE/'outputs/lens-provenance.json',dict(n_prompts=total,
         lens_sha256=sha(HERE/'outputs/lens.pt'),config_sha256=sha(HERE/'config.json'),
         original_estimator=True,merge='prompt-count weighted arithmetic mean, streamed fp32'))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['fit','merge'])
    parser.add_argument('--shard',type=int)
    args=parser.parse_args()
    if args.action=='fit': fit_shard(args.shard)
    else: merge()
