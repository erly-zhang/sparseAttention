"""Activation interventions in the effective token-direction convention.

Token atoms include final RMSNorm gains. A token span is a linear subspace,
not the sparse nonnegative J-space cone. Bag steering cannot encode order.
"""
import torch


def token_atoms(weight, gamma, jacobian, token_ids):
    rows = weight[token_ids].float() * gamma.float()
    if jacobian is not None:
        rows = rows @ jacobian.float()
    return torch.nn.functional.normalize(rows, dim=-1)


def span_basis(atoms):
    _, singular, vh = torch.linalg.svd(atoms.float(), full_matrices=False)
    cutoff = singular.max().clamp_min(1e-30) * 1e-6
    return vh[singular > cutoff]


def project(vector, basis):
    return (vector @ basis.T) @ basis


def change(hidden, *, kind, alpha, atoms, ordered_ids, unique_ids,
           wrong_atoms=None, donor=None, generator=None):
    h = hidden.float()
    if kind.startswith('steer_'):
        if kind == 'steer_first':
            direction = atoms[unique_ids.index(ordered_ids[0])]
        elif kind == 'steer_full':
            lookup = [unique_ids.index(t) for t in ordered_ids]
            direction = atoms[lookup].sum(0)  # multiplicity, without order
        elif kind == 'steer_wrong':
            direction = wrong_atoms.sum(0)
        elif kind == 'steer_random':
            direction = torch.randn(h.shape, generator=generator, device=h.device, dtype=h.dtype)
        else:
            raise ValueError(kind)
        direction = direction / direction.norm().clamp_min(1e-30)
        delta = float(alpha) * h.norm() * direction
    else:
        difference = donor.float() - h
        basis = span_basis(atoms)
        in_span = project(difference, basis)
        if kind == 'dense_value_span':
            delta = float(alpha) * in_span
        elif kind == 'dense_orthogonal_matched':
            remainder = difference - in_span
            if remainder.norm() < 1e-12 and in_span.norm() > 1e-12:
                raise ValueError('Cannot construct a nonzero norm-matched orthogonal control')
            delta = float(alpha) * remainder * (in_span.norm() / remainder.norm().clamp_min(1e-30))
        elif kind == 'dense_full':
            delta = float(alpha) * difference
        else:
            raise ValueError(kind)
    if not torch.isfinite(delta).all():
        raise ValueError('Nonfinite intervention')
    return h + delta, delta
