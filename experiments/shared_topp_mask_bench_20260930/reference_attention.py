"""Small literal reference with independent row-wise nucleus selection."""
import math
import torch


def full_attention(q,k,v,positions,mode='shared_topp',top_p=0.95):
    b,h,n,d = q.shape
    if mode not in ('shared_topp','dense_full') or not 0 < top_p <= 1 or n > 4096:
        raise ValueError('invalid reference parameters')
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    causal = positions[:,:,None] >= positions[:,None,:]
    scores = []
    for head in range(h):
        kv = head//(h//k.shape[1])
        s = q[:,head].to(dtype) @ k[:,kv].to(dtype).transpose(-1,-2) / math.sqrt(d)
        scores.append(s.masked_fill(~causal,float('-inf')))
    mask = torch.zeros_like(causal) if mode == 'shared_topp' else None
    if mask is not None:
        for batch in range(b):
            for row in range(n):
                head = int(positions[batch,row]) % h
                probabilities = scores[head][batch,row].softmax(-1)
                keys = [j for j in range(n) if causal[batch,row,j]]
                keys.sort(key=lambda j:(-float(probabilities[j]),int(positions[batch,j])))
                mass = 0.0
                for j in keys:
                    mask[batch,row,j] = True
                    mass += float(probabilities[j])
                    if top_p < 1 and mass >= top_p:
                        break
    output = []
    for head,s in enumerate(scores):
        kv = head//(h//k.shape[1])
        p = (s.masked_fill(~mask,float('-inf')) if mask is not None else s).softmax(-1)
        output.append((p @ v[:,kv].to(dtype)).to(q.dtype))
    return torch.stack(output,dim=2).flatten(-2),mask,{}
