"""Offline learned-mask prototype. No patching of an inference runtime."""
from dataclasses import dataclass
import math

import torch
from torch import nn


VARIANTS = ("per_head", "per_cluster", "per_layer", "full_mask")


@dataclass(frozen=True)
class Config:
    head_dim: int = 128
    latent_dim: int = 16
    hidden_dim: int = 32
    head_embedding_dim: int = 8
    query_segments: int = 4
    tile_size: int = 128
    sink: int = 128
    budget: int = 10240
    core_budget: int = 6144


def validate_groups(groups):
    if sorted(map(int, groups)) != list(range(len(groups))):
        raise ValueError("Expected consecutive layer indices")
    heads = None
    for layer in groups.values():
        members = [h for group in layer for h in group["members"]]
        if sorted(members) != list(range(len(members))):
            raise ValueError("Groups must partition all Q heads exactly once")
        if heads is not None and len(members) != heads:
            raise ValueError("Inconsistent head counts")
        heads = len(members)
        if any(g["representative"] not in g["members"] for g in layer):
            raise ValueError("Representative is not a member")
    if not heads:
        raise ValueError("Empty groups")
    return heads


def protected_mask(n, start, end, config):
    if not (0 <= start < end <= n and end - start <= config.tile_size):
        raise ValueError("Invalid real query tile")
    ids = torch.arange(n)
    return (ids < end) & ((ids < config.sink) | (ids >= start))


def select_fixed(scores, protect, end, budget):
    """Stable position tie-break for this new prototype, not historical TopK ties."""
    n = scores.numel()
    if scores.ndim != 1 or protect.shape != scores.shape or not 0 < end <= n:
        raise ValueError("Invalid selection shapes")
    if not torch.isfinite(scores[:end]).all():
        raise ValueError("Nonfinite legal scores")
    protect = protect.to(device=scores.device, dtype=torch.bool)
    legal = torch.arange(n, device=scores.device) < end
    if (protect & ~legal).any() or int(protect.sum()) > min(budget, end):
        raise ValueError("Protection exceeds legal budget")
    selected = protect.clone()
    remaining = min(budget, end) - int(selected.sum())
    if remaining:
        ranking = scores.masked_fill(~legal | protect, -torch.inf)
        order = torch.argsort(ranking, descending=True, stable=True)
        selected[order[:remaining]] = True
    return selected


def shared_masks(shared_p, start, end, config):
    protect = protected_mask(len(shared_p), start, end, config).to(shared_p.device)
    core = select_fixed(shared_p, protect, end, config.core_budget)
    full = select_fixed(shared_p, protect, end, config.budget)
    return protect, core, full


def final_mask(scores, core, full, protect, end, config, variant, representative=False):
    if variant not in VARIANTS:
        raise ValueError(variant)
    if representative:
        return full.clone()
    fixed = protect if variant == "full_mask" else core
    return select_fixed(scores, fixed, end, config.budget)


def exact_probability_mean(q, k, start, query_chunk=8):
    """Offline teacher: every real query normalizes over ALL its legal keys."""
    if q.ndim != 2 or k.ndim != 2 or q.shape[1] != k.shape[1]:
        raise ValueError("Expected Q[m,d], correctly mapped K[n,d]")
    m, dim = q.shape
    if query_chunk <= 0 or m == 0 or not 0 <= start < start + m <= len(k):
        raise ValueError("Invalid query range")
    result = torch.zeros(len(k), device=q.device, dtype=torch.float32)
    keys = torch.arange(len(k), device=q.device)
    for lo in range(0, m, query_chunk):
        hi = min(lo + query_chunk, m)
        score = q[lo:hi].float() @ k.float().T / math.sqrt(dim)
        positions = torch.arange(start + lo, start + hi, device=q.device)
        score.masked_fill_(keys[None, :] > positions[:, None], -torch.inf)
        result += score.softmax(-1).sum(0) / m
    return result


def mapped_k(k, q_heads, head):
    """K[n,kv_heads,d], standard contiguous Llama GQA mapping."""
    if k.ndim != 3 or q_heads % k.shape[1] or not 0 <= head < q_heads:
        raise ValueError("Invalid GQA mapping")
    return k[:, head // (q_heads // k.shape[1])]


def query_summary(q, segments):
    if q.ndim != 2 or len(q) == 0:
        raise ValueError("Only real, non-padding queries are accepted")
    parts = []
    for i in range(segments):
        lo, hi = i * len(q) // segments, (i + 1) * len(q) // segments
        parts.append(q[lo:hi].mean(0) if hi > lo else torch.zeros_like(q[0]))
    return torch.cat(parts)


class Scorer(nn.Module):
    def __init__(self, config, num_heads):
        super().__init__()
        c = config
        self.config = c
        self.head_embedding = nn.Embedding(num_heads, c.head_embedding_dim)
        self.key = nn.Linear(c.head_dim, c.latent_dim, bias=False)
        self.query = nn.Sequential(
            nn.Linear(c.query_segments * c.head_dim + c.head_embedding_dim + c.latent_dim,
                      c.hidden_dim), nn.GELU(), nn.Linear(c.hidden_dim, c.latent_dim))
        self.prior = nn.Linear(2, 1, bias=False)

    def forward(self, q, k, shared_p, shared_mask, head):
        # Prototype recomputes projections; production must cache per KV/network.
        z = self.key(k.float())
        membership = shared_mask.to(device=z.device, dtype=torch.bool)
        context = z[membership].mean(0)
        embedding = self.head_embedding.weight[head]
        u = self.query(torch.cat((query_summary(q.float(), self.config.query_segments),
                                  embedding, context)))
        # The full positional membership vector is consumed, not only mask size.
        prior = torch.stack(((shared_p.float().clamp_min(1e-20) * len(k)).log(),
                             membership.float()), -1)
        return z @ u / math.sqrt(self.config.latent_dim) + self.prior(prior).squeeze(-1)


class MaskOnlyScorer(nn.Module):
    """Experiment4: positional shared mask + head identity, no Q/K or scores."""
    def __init__(self, config, num_heads):
        super().__init__()
        c = config
        self.head_embedding = nn.Embedding(num_heads, c.head_embedding_dim)
        self.encoder = nn.Sequential(
            nn.Conv1d(2, c.latent_dim, kernel_size=7, padding=3), nn.GELU(),
            nn.Conv1d(c.latent_dim, c.latent_dim, kernel_size=7, dilation=16, padding=48),
            nn.GELU())
        self.query = nn.Sequential(nn.Linear(c.head_embedding_dim + 2*c.latent_dim, c.hidden_dim),
                                   nn.GELU(), nn.Linear(c.hidden_dim, c.latent_dim))

    def forward(self, q, k, shared_p, shared_mask, head):
        # Normalized position is derived from the mask index, not model content.
        mask = shared_mask.float()
        position = torch.linspace(0, 1, len(mask), device=mask.device)
        z = self.encoder(torch.stack((mask, position))[None, :])[0].T
        context = torch.cat((z.mean(0), z[shared_mask].mean(0)))
        u = self.query(torch.cat((self.head_embedding.weight[head], context)))
        return z @ u / math.sqrt(z.shape[-1])


class Predictor(nn.Module):
    def __init__(self, groups, variant, config=Config()):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.num_heads = validate_groups(groups)
        self.config, self.groups, self.variant = config, groups, variant
        self.routing, self.representatives = {}, set()
        networks = {}
        for layer, clusters in groups.items():
            for group_id, group in enumerate(clusters):
                self.representatives.add((int(layer), group["representative"]))
                for head in group["members"]:
                    if head == group["representative"]:
                        continue
                    key = (f"l{layer}_h{head}" if variant == "per_head" else
                           f"l{layer}_g{group_id}" if variant == "per_cluster" else f"l{layer}")
                    self.routing[int(layer), head] = key
                    if key not in networks:
                        cls = MaskOnlyScorer if variant == "full_mask" else Scorer
                        networks[key] = cls(config, self.num_heads)
        self.networks = nn.ModuleDict(networks)

    def forward(self, q, k, shared_p, shared_mask, layer, head):
        return self.networks[self.routing[layer, head]](q, k, shared_p, shared_mask, head)


def teacher_loss(scores, teacher_p, fixed, end):
    candidate = (torch.arange(len(scores), device=scores.device) < end) & ~fixed
    if not candidate.any():
        return scores.sum() * 0
    mass = teacher_p[candidate].float()
    if mass.sum() <= 0:
        return scores.sum() * 0
    target = mass / mass.sum()
    return -(target * scores[candidate].log_softmax(0)).sum()


def selection_metrics(prediction, oracle, teacher_p, fixed, start, end):
    residual = prediction & ~fixed
    target = oracle & ~fixed
    ids = torch.arange(len(prediction), device=prediction.device)
    pairs = (end - torch.maximum(ids, torch.tensor(start, device=ids.device))).clamp_min(0)
    return dict(overlap=int((residual & target).sum()), target_count=int(target.sum()),
                recovered_mass=float(teacher_p[residual].sum()),
                oracle_mass=float(teacher_p[target].sum()),
                final_mass=float(teacher_p[prediction].sum()),
                selected_count=int(prediction.sum()),
                selected_pairs=int(pairs[prediction].sum()),
                causal_pairs=sum(i + 1 for i in range(start, end)))
