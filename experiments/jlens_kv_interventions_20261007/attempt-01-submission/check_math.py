"""CPU checks for causal-control geometry, not a model experiment."""
import json
import torch
from intervention_math import change, project, span_basis, token_atoms


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    h, donor = torch.randn(12), torch.randn(12)
    atoms = torch.nn.functional.normalize(torch.randn(4,12),dim=-1)
    common = dict(atoms=atoms,ordered_ids=[1,2,2,3,4],unique_ids=[1,2,3,4],
                  wrong_atoms=atoms.flip(0),donor=donor)
    basis = span_basis(atoms)
    patched, delta = change(h,kind='dense_value_span',alpha=1,**common)
    torch.testing.assert_close(project(patched,basis),project(donor,basis),atol=2e-6,rtol=2e-6)
    torch.testing.assert_close(patched-project(patched,basis),h-project(h,basis),atol=2e-6,rtol=2e-6)
    _, orth = change(h,kind='dense_orthogonal_matched',alpha=1,**common)
    torch.testing.assert_close(project(orth,basis),torch.zeros_like(orth),atol=2e-6,rtol=0)
    torch.testing.assert_close(orth.norm(),delta.norm(),atol=2e-6,rtol=2e-6)
    full, _ = change(h,kind='dense_full',alpha=1,**common)
    torch.testing.assert_close(full,donor)
    for kind in ['steer_first','steer_full','steer_wrong','steer_random','dense_value_span','dense_orthogonal_matched','dense_full']:
        unchanged, d = change(h,kind=kind,alpha=0,**common)
        assert torch.equal(unchanged,h) and torch.equal(d,torch.zeros_like(h))
        if kind.startswith('steer_'):
            _, d = change(h,kind=kind,alpha=.3,**common)
            torch.testing.assert_close(d.norm()/h.norm(),torch.tensor(.3))
    # Inserting the same token bag with a different order must be identical.
    bag, _ = change(h,kind='steer_full',alpha=.3,**common)
    reverse = dict(common,ordered_ids=list(reversed(common['ordered_ids'])))
    other, _ = change(h,kind='steer_full',alpha=.3,**reverse)
    torch.testing.assert_close(bag,other)
    w,g,j = torch.randn(7,12),torch.randn(12),torch.randn(12,12)
    expected = torch.nn.functional.normalize((w*g)@j,dim=-1)
    torch.testing.assert_close(token_atoms(w,g,j,list(range(7))),expected)
    print(json.dumps(dict(status='PASS',device='CPU',model_forward=False,
                         span_coordinate_restore=True,orthogonal_complement_preserved=True,
                         matched_control_norm=True,zero_strength_identity=True,
                         bag_order_invariance=True,effective_rms_gain_directions=True)))


if __name__ == '__main__':
    main()
