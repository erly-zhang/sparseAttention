"""Residual pre-o_proj correction; base model remains frozen."""
import torch
from torch import nn
from torch.nn import functional as F


class PreProjectionMLP(nn.Module):
    def __init__(self, width=4096, rank=128):
        super().__init__()
        if not 0 < rank <= width:
            raise ValueError('invalid bottleneck')
        self.down = nn.Linear(width, rank)
        self.up = nn.Linear(rank, width)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        y = x.to(self.down.weight.dtype)
        return (y + self.up(F.silu(self.down(y)))).to(x.dtype)


def loss(adapter, projection, x, target):
    if x.ndim != 2 or x.shape != target.shape or not len(x):
        raise ValueError('unaligned features')
    if any(p.requires_grad for p in projection.parameters()):
        raise ValueError('original o_proj must be frozen')
    if not bool(torch.isfinite(x).all() and torch.isfinite(target).all()):
        raise ValueError('nonfinite features')
    # FP32 feature loss; inference casts correction back to original BF16.
    corrected = adapter(x.detach().float())
    target = target.detach().float()
    raw = (corrected - target).square().sum(-1) / target.square().sum(-1).clamp_min(1e-8)
    weight = projection.weight.detach().float()
    bias = None if projection.bias is None else projection.bias.detach().float()
    student = F.linear(corrected, weight, bias)
    with torch.no_grad():
        teacher = F.linear(target, weight, bias)
    projected = (student - teacher).square().sum(-1) / teacher.square().sum(-1).clamp_min(1e-8)
    value = (raw + projected).mean()
    if not bool(torch.isfinite(value)):
        raise ValueError('nonfinite loss')
    return value


class CaptureComplete(Exception):
    """Intentional early exit at the current layer during feature collection."""


class ProjectionHooks:
    def __init__(self, model, layers):
        self.adapters = {}
        self.prefill = False
        self.enabled = False
        self.capture = set()
        self.indices = None
        self.stop = None
        self.features = {}
        self.calls = {}
        self.handles = [model.model.layers[i].self_attn.o_proj.register_forward_pre_hook(self.hook(i))
                        for i in layers]

    def hook(self, layer):
        def apply(module, args):
            if not self.prefill:
                return None
            x = args[0]
            if x.ndim != 3 or x.shape[0] != 1:
                raise ValueError('single unpadded sample required')
            if layer in self.capture:
                self.features[layer] = x[0, self.indices].detach().float().cpu().clone()
            if layer == self.stop:
                raise CaptureComplete()
            if self.enabled and layer in self.adapters:
                self.calls[layer] = self.calls.get(layer, 0) + 1
                adapter = self.adapters[layer]
                # Bound temporary activation memory for long prompts.
                y = torch.cat([adapter(chunk) for chunk in x.split(512, dim=1)], dim=1)
                if not bool(torch.isfinite(y).all()):
                    raise ValueError('nonfinite adapter output')
                return (y,) + args[1:]
            return None
        return apply

    def close(self):
        for handle in self.handles:
            handle.remove()
