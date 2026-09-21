"""Batched inference equivalent to the frozen training model's prefix inputs."""
import math
import torch


def protection(ends, starts, n, sink):
    ids = torch.arange(n, device=ends.device)[None]
    return (ids < ends[:, None]) & ((ids < sink) | (ids >= starts[:, None]))


def fixed_mask(scores, fixed, ends, budget):
    """Stable key-position tie break, including negative NN logits."""
    n = scores.shape[-1]
    ids = torch.arange(n, device=scores.device)[None]
    legal = ids < ends[:, None]
    ranking = scores.masked_fill(~legal | fixed, -torch.inf)
    order = torch.argsort(ranking, dim=-1, descending=True, stable=True)
    remaining = ends.clamp(max=budget) - fixed.sum(-1)
    chosen = torch.arange(n, device=scores.device)[None] < remaining[:, None]
    return fixed | torch.zeros_like(fixed).scatter(1, order, chosen)


def summaries(q, starts, ends, segments):
    # Prefix sums would change cancellation error. Match the training means.
    return torch.stack([
        torch.cat([q[a + i*(b-a)//segments:a + (i+1)*(b-a)//segments].float().mean(0)
                   if (i+1)*(b-a)//segments > i*(b-a)//segments
                   else torch.zeros_like(q[0], dtype=torch.float32)
                   for i in range(segments)])
        for a, b in zip(starts, ends)
    ])


def content_scores(net, z, summary, p, core, ends, head):
    context = core.float() @ z / core.sum(-1, keepdim=True)
    embedding = net.head_embedding.weight[head].expand(len(core), -1)
    u = net.query(torch.cat((summary, embedding, context), dim=-1))
    prior = torch.stack(((p.float().clamp_min(1e-20) * ends[:, None]).log(),
                         core.float()), dim=-1)
    return u @ z.T / math.sqrt(z.shape[-1]) + net.prior(prior).squeeze(-1)


def encode_masks(net, masks, ends):
    """Zero each padded prefix after BOTH convs, not only after the last conv."""
    ids = torch.arange(masks.shape[-1], device=masks.device)[None]
    legal = ids < ends[:, None]
    position = ids / (ends[:, None] - 1).clamp_min(1)
    x = torch.stack((masks.float(), position.masked_fill(~legal, 0)), dim=1)
    x = net.encoder[1](net.encoder[0](x)).masked_fill(~legal[:, None], 0)
    x = net.encoder[3](net.encoder[2](x)).masked_fill(~legal[:, None], 0)
    z = x.transpose(1, 2)
    context = torch.cat((z.sum(1)/ends[:, None],
                         (z*masks[:, :, None]).sum(1)/masks.sum(-1, keepdim=True)), -1)
    return z, context


def mask_scores(net, encoded, head):
    z, context = encoded
    embedding = net.head_embedding.weight[head].expand(len(z), -1)
    u = net.query(torch.cat((embedding, context), -1))
    return torch.einsum('bnd,bd->bn', z, u)/math.sqrt(z.shape[-1])


def indices(mask, width, sentinel):
    ids = torch.arange(mask.shape[-1], device=mask.device)[None].expand_as(mask)
    return ids.masked_fill(~mask, sentinel).sort(-1).values[:, :width].to(torch.int32)
