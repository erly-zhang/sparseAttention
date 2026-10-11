"""One-input diagnostic: final kernel indices versus Shared on identical Q/K.

Run separately from timed evaluation. Never modify the selector's return value.
"""
import json
import os
from pathlib import Path
import numpy as np
import torch
from experiments.learned_mask_eval_20260914 import online
from experiments.learned_mask_eval_20260914 import selector as module

OUT = Path(os.environ['MASK_SHAPE_OUTPUT'])
OUT.mkdir(parents=True, exist_ok=False)
original_call = module.Selector.__call__
original_shared = module.shared_indices
active = None


def shared(*args, **kwargs):
    bundle = original_shared(*args, **kwargs)
    if active is not None:
        active['shared'].append(bundle['old'][active['tiles']].cpu().numpy())
    return bundle


def describe(keys, end):
    hist = np.histogram(keys, bins=np.linspace(0, end, 33))[0]
    runs = 1 + int(np.count_nonzero(np.diff(keys) > 1))
    return dict(position_hist=hist.tolist(), runs=runs,
                mean_run_length=len(keys)/runs, mean_position=float(keys.mean()/end),
                first_quarter=float(np.mean(keys < end/4)),
                last_quarter=float(np.mean(keys >= 3*end/4)))


@torch.no_grad()
def capture(self, q, *args, **kwargs):
    global active
    n, heads = q.shape[1:3]
    if n <= 10497:
        return original_call(self, q, *args, **kwargs)
    layer = self.current_layer.get()
    tiles = np.unique(np.linspace(80, (n+127)//128-1, 8).astype(int)).tolist()
    active = dict(tiles=tiles, shared=[])
    try:
        result = original_call(self, q, *args, **kwargs)
        final = result.token_indices.view(heads, (n+127)//128, min(n,10240))[:, tiles].cpu().numpy()
        baseline = np.stack(active['shared'])
    finally:
        active = None
    assert baseline.shape[0] == 3
    records = []
    for g, group in enumerate(self.layers[str(layer)]):
        rep = group['representative']
        for t, tile in enumerate(tiles):
            start, end = tile*128, min((tile+1)*128,n)
            protected = np.unique(np.r_[np.arange(min(128,end)),np.arange(start,end)])
            old = baseline[g,t]
            assert len(old)==len(np.unique(old))==min(10240,end)
            for h in group['members']:
                keys = final[h,t]
                assert len(keys)==len(np.unique(keys))==min(10240,end)
                assert (keys>=0).all() and (keys<end).all()
                assert np.isin(protected,keys).all()
                overlap = len(np.intersect1d(keys,old,assume_unique=True))
                if h==rep:
                    assert overlap == len(keys)
                peers = [x for x in group['members'] if x!=rep and x!=h]
                peer_overlap = [len(np.intersect1d(keys,final[x,t],assume_unique=True))/len(keys) for x in peers]
                records.append(dict(layer=layer,head=h,group=g,representative=rep,tile=tile,
                    tile_start=start,tile_end=end,budget=len(keys),is_representative=h==rep,
                    shared_overlap=overlap/len(keys),new_keys=len(keys)-overlap,
                    member_overlap=float(np.mean(peer_overlap)) if peer_overlap else None,
                    final=describe(keys,end),shared=describe(old,end)))
    np.savez_compressed(OUT/f'layer{layer:02d}.npz', final=final, shared=baseline,
                        tiles=np.array(tiles), n=np.array(n))
    (OUT/f'layer{layer:02d}.json').write_text(json.dumps(records))
    print('MASK_CAPTURE',self.predictor.variant,layer,n,flush=True)
    return result


module.shared_indices = shared
module.Selector.__call__ = capture

if __name__ == '__main__':
    online.base.runner.main()
