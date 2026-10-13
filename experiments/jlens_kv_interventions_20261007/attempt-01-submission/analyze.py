"""Readouts and full-vocabulary J-space fits of measured intervention states."""
import argparse
from pathlib import Path
import sys
from prepare import HERE, read, sha
from run import gate, atomic_json


def main(index):
    gate()
    import torch
    from safetensors import safe_open
    from jlens.vis import _ranks_of
    c,p = read(HERE/'config.json'),read(HERE/'prepared.json')
    analysis = (HERE/c['source_analysis']).resolve()
    sys.path.insert(0,str(analysis))
    from pursuit import pursuit
    directory = HERE/'outputs'/f'trial-{index:03d}'
    result = read(directory/'result.json')
    assert result['artifact_sha256'] == sha(directory/'activations.pt')
    data = torch.load(directory/'activations.pt',weights_only=True,map_location='cpu',mmap=True)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = Path(p['model'])
    weight_map = read(model/'model.safetensors.index.json')['weight_map']
    def weight(name):
        with safe_open(str(model/weight_map[name]),framework='pt',device='cpu') as f:
            return f.get_tensor(name).to('cuda')
    w,gamma = weight('lm_head.weight'),weight('model.norm.weight')
    eps = read(model/'config.json')['rms_norm_eps']
    lens = torch.load(p['lens'],weights_only=True,map_location='cpu',mmap=True)
    targets = sorted(set(p['target_ids']))
    target_tensor = torch.tensor(targets,device='cuda')
    metrics,tensors = {},{}
    for layer in c['decomposition']['layers']:
        rows,annotations = [],[]
        sites = [result['input_tokens']-1]
        if not result['trial']['kind'].startswith('text_'):
            sites += [p['source_value_positions'][-1],p['sites']['source_after']]
        for pos in sorted(set(sites)):
            if pos in data['prefill_positions']:
                rows.append(data['prefill_residuals'][layer][data['prefill_positions'].index(pos)])
                annotations.append(dict(phase='prefill',position=pos))
        if layer in data['decode_residuals']:
            for step,h in enumerate(data['decode_residuals'][layer][:32]):
                rows.append(h)
                annotations.append(dict(phase='decode',position=result['input_tokens']+step,
                                        input_generated_token_id=data['generated_ids'][step],
                                        predicted_step=step+1))
        intervention = data['intervention']
        if intervention and intervention['layer'] == layer:
            rows += [intervention['before'],intervention['after']]
            annotations += [dict(phase='injection_before',position=intervention['position']),
                            dict(phase='injection_after',position=intervention['position'])]
        h = torch.stack(rows).float().cuda()
        j = lens['J'][layer].float().cuda() if layer != 31 else None
        transported = h if j is None else h@j.T
        transported = transported.to(w.dtype).float()
        normalized = gamma*(transported*torch.rsqrt(transported.square().mean(-1,keepdim=True)+eps)).to(w.dtype)
        logits = torch.nn.functional.linear(normalized,w).float()
        ranks = _ranks_of(logits,target_tensor)+1
        logp = logits.log_softmax(-1).index_select(1,target_tensor)
        atoms = w.float()*gamma.float()
        if j is not None:
            atoms = atoms@j
        atoms = torch.nn.functional.normalize(atoms,dim=-1)
        fits = pursuit(h,atoms,tuple(c['decomposition']['checkpoints']))
        metrics[str(layer)] = dict(annotations=annotations,target_ids=targets,
                                  target_ranks=ranks.cpu().tolist(),target_log_probabilities=logp.cpu().tolist(),
                                  jspace={str(k):dict(energy_reconstructed=v['energy_reconstructed'].cpu().tolist(),
                                                      active_count=v['active_count'].cpu().tolist(),
                                                      support=v['support'].cpu().tolist(),
                                                      coefficients=v['coefficients'].cpu().tolist()) for k,v in fits.items()})
        tensors[layer] = {k:{name:v[name].cpu() for name in ['support','coefficients','reconstruction','remainder']} for k,v in fits.items()}
        del h,j,transported,normalized,logits,atoms,fits
        torch.cuda.empty_cache()
    torch.save(tensors,directory/'decompositions.pt')
    atomic_json(directory/'analysis.json',dict(status='complete',trial=result['trial'],metrics=metrics,
                decomposition_sha256=sha(directory/'decompositions.pt'),
                rank_scope='full vocabulary; ranks are individual token probes, not UUID sequence probabilities',
                energy_scope='uncentered activation energy; not centered variance explained',
                generation_scope='first 32 saved decode positions; all decode residuals remain in activations.pt'))
    print('ANALYSIS COMPLETE',index)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index',type=int,required=True)
    main(parser.parse_args().index)
