"""Readout-only prototype; does not change attention, caches, or model weights."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class ResidualReadout(nn.Module):
    def __init__(self, hidden_size, rank=128, nonlinear=True):
        super().__init__()
        if not 0 < rank <= hidden_size:
            raise ValueError("rank must be in [1, hidden_size]")
        self.down = nn.Linear(hidden_size, rank)
        self.up = nn.Linear(rank, hidden_size)
        self.nonlinear = nonlinear
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden, *, enabled):
        # Explicit request mode covers both first-answer and cached-decode states.
        if not enabled:
            return hidden
        x = self.down(hidden.to(self.down.weight.dtype))
        if self.nonlinear:
            x = F.silu(x)
        return (hidden.to(x.dtype) + self.up(x)).to(hidden.dtype)


def paired_loss(adapter, lm_head, sparse_hidden, dense_hidden, *,
                weights=None, alpha=1.0, beta=1.0, temperature=1.0):
    """Loss over aligned answer-prediction states, excluding padding/post-EOS.

    Call with small feature microbatches: full vocabulary logits are materialized.
    Weights should balance complete prompts rather than over-weight long answers.
    """
    if sparse_hidden.ndim != 2 or sparse_hidden.shape != dense_hidden.shape:
        raise ValueError("expected aligned [positions, hidden_size] tensors")
    if sparse_hidden.shape[0] == 0:
        raise ValueError("empty feature batch")
    if not all(math.isfinite(v) for v in (alpha, beta, temperature)):
        raise ValueError("non-finite loss configuration")
    if temperature <= 0 or min(alpha, beta) < 0 or alpha + beta == 0:
        raise ValueError("invalid loss configuration")
    if any(p.requires_grad for p in lm_head.parameters()):
        raise ValueError("freeze LM head before calibration")
    if not (torch.isfinite(sparse_hidden).all() and torch.isfinite(dense_hidden).all()):
        raise ValueError("non-finite collected features")
    corrected = adapter(sparse_hidden.detach(), enabled=True)
    target = dense_hidden.detach().float()
    hidden_error = (corrected.float() - target).square().sum(-1)
    hidden_error = hidden_error / target.square().sum(-1).clamp_min(1e-8)
    dtype = lm_head.weight.dtype
    with torch.no_grad():
        teacher = lm_head(dense_hidden.detach().to(dtype)).float() / temperature
        teacher_logp = F.log_softmax(teacher, dim=-1)
    # Frozen head still participates in autograd on the student branch.
    student = lm_head(corrected.to(dtype)).float() / temperature
    student_logp = F.log_softmax(student, dim=-1)
    kl = (teacher_logp.exp() * (teacher_logp - student_logp)).sum(-1)
    loss = alpha * hidden_error + beta * temperature**2 * kl
    if weights is None:
        weights = torch.ones_like(loss)
    else:
        weights = weights.detach().to(device=loss.device, dtype=loss.dtype)
    if weights.shape != loss.shape or not torch.isfinite(weights).all():
        raise ValueError("invalid position weights")
    if bool((weights < 0).any()) or not bool(weights.sum() > 0):
        raise ValueError("weights must be nonnegative with positive sum")
    if not torch.isfinite(loss).all():
        raise ValueError("non-finite calibration loss")
    return (loss * weights).sum() / weights.sum()
