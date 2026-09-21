"""CPU correctness and tiny synthetic optimization, not Llama experiment scores."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch

from .model import (Config, Predictor, exact_probability_mean, final_mask, mapped_k,
                    shared_masks, teacher_loss, selection_metrics, VARIANTS)
from .data import export_record, validate_manifest, validate_group_provenance
from .train import main as train_main, cases


GROUPS = {"0": [{"representative": 0, "members": [0, 1, 2]},
                 {"representative": 3, "members": [3, 4, 5]}],
          "1": [{"representative": 0, "members": [0, 1, 2]},
                 {"representative": 3, "members": [3, 4, 5]}]}
C = Config(head_dim=4, latent_dim=4, hidden_dim=8, head_embedding_dim=2,
           query_segments=4, tile_size=4, sink=2, budget=8, core_budget=6)


class Tests(unittest.TestCase):
    def test_teacher_causal_gqa_and_partial(self):
        torch.manual_seed(1)
        q, k = torch.randn(3, 4), torch.randn(15, 2, 4)
        selected_k = mapped_k(k, 6, 4)
        torch.testing.assert_close(selected_k, k[:, 1])
        p = exact_probability_mean(q, selected_k, 9, query_chunk=2)
        expected = torch.zeros(15)
        for offset in range(3):
            score = (selected_k[:10 + offset].float() @ q[offset].float()) / 2
            expected[:10 + offset] += score.softmax(0) / 3
        torch.testing.assert_close(p, expected)
        torch.testing.assert_close(p.sum(), torch.tensor(1.))
        self.assertEqual(float(p[12:].sum()), 0.)
        changed = selected_k.clone()
        changed[12:] = 1e6
        torch.testing.assert_close(p, exact_probability_mean(q, changed, 9))

    def test_budget_and_representative_bypass(self):
        for end in (1, 3, 7, 8, 11, 13):
            start = max(0, end - 3)
            p = torch.arange(1., 16.)
            p[end:] = 0
            p /= p.sum()
            protect, core, full = shared_masks(p, start, end, C)
            for variant in VARIANTS:
                scores = torch.arange(15.).flip(0)
                result = final_mask(scores, core, full, protect, end, C, variant)
                self.assertEqual(int(result.sum()), min(8, end))
                self.assertFalse(result[end:].any())
                self.assertTrue(result[protect].all())
                if variant != "full_mask":
                    self.assertTrue(result[core].all())
                torch.testing.assert_close(full, final_mask(scores, core, full, protect,
                                                           end, C, variant, True))

    def test_full_mask_can_remove_core(self):
        p = torch.arange(1., 21.)
        p /= p.sum()
        protect, core, full = shared_masks(p, 17, 20, C)
        scores = -torch.arange(20.)
        residual = final_mask(scores, core, full, protect, 20, C, "per_layer")
        edited = final_mask(scores, core, full, protect, 20, C, "full_mask")
        self.assertTrue(residual[core].all())
        self.assertTrue((core & ~edited).any())
        self.assertTrue(edited[protect].all())

    def test_routing(self):
        counts = {"per_head": 8, "per_cluster": 4, "per_layer": 2, "full_mask": 2}
        for variant in VARIANTS:
            model = Predictor(GROUPS, variant, C)
            self.assertEqual(len(model.networks), counts[variant])
            self.assertNotIn((0, 0), model.routing)
        layer = Predictor(GROUPS, "per_layer", C)
        full = Predictor(GROUPS, "full_mask", C)
        self.assertEqual(layer.routing, full.routing)

    def test_full_mask_does_not_use_qk_or_probability(self):
        model = Predictor(GROUPS, "full_mask", C)
        q, k, p = torch.randn(3, 4), torch.randn(20, 4), torch.ones(20)/20
        _, _, full = shared_masks(p, 17, 20, C)
        first = model(q, k, p, full, 0, 1)
        second = model(q * 9, k + 30, p.flip(0) * .5, full, 0, 1)
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_training_all_variants(self):
        torch.set_num_threads(1)
        for variant in VARIANTS:
            torch.manual_seed(42)
            model = Predictor(GROUPS, variant, C)
            q, k = torch.randn(3, 4), torch.randn(20, 4)
            p = exact_probability_mean(q, k, 17)
            shared = torch.arange(1., 21.)
            shared /= shared.sum()
            protect, core, full = shared_masks(shared, 17, 20, C)
            fixed = protect if variant == "full_mask" else core
            conditioning = full if variant == "full_mask" else core
            optimizer = torch.optim.Adam(model.parameters(), lr=.02)
            history = []
            for _ in range(30):
                scores = model(q, k, shared, conditioning, 0, 1)
                loss = teacher_loss(scores, p, fixed, 20)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                history.append(float(loss.detach()))
            self.assertLess(history[-1], history[0])
            pred = final_mask(scores, core, full, protect, 20, C, variant)
            oracle = final_mask(p, core, full, protect, 20, C, variant)
            metrics = selection_metrics(pred, oracle, p, fixed, 17, 20)
            self.assertLessEqual(metrics["selected_pairs"], metrics["causal_pairs"])

    def test_manifest_and_training_cli(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "protected.jsonl"
            reference.write_text(json.dumps({"input_ids_sha256": "f" * 64}) + "\n")
            manifest = dict(schema="learned_mask_dense_teacher_v1", protected_reference=str(reference),
                            protected_reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                            splits={})
            for split in ("train", "validation", "test"):
                entries = []
                identity = hashlib.sha256(split.encode()).hexdigest()
                for layer in range(2):
                    entry = export_record(root / f"{split}-{layer}.pt", q_tile=torch.randn(3, 6, 4),
                                          k=torch.randn(10248, 2, 4), groups=GROUPS, layer=layer,
                                          start=10245, input_tokens=10248,
                                          input_ids_sha256=identity, family_id=split)
                    entries.append(entry)
                manifest["splits"][split] = entries
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest))
            validate_manifest(path)
            groups = root / "groups.json"
            groups.write_text(json.dumps({"layers": GROUPS}))
            command = ["learned_mask.train",
                       "--manifest", str(path), "--groups", str(groups), "--epochs", "1",
                       "--output", str(root/"run")]
            with patch("sys.argv", command):
                train_main()
            for variant in VARIANTS:
                result = json.loads((root/"run"/variant/"heldout.json").read_text())
                self.assertEqual(result["cases"], 8)
                self.assertEqual(result["raw_counts"]["selected_count"], 8 * 10240)
            # Changing only the family identifier must still trigger leakage checks.
            manifest["splits"]["validation"][0]["family_id"] = "train"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "leaks across splits"):
                validate_manifest(path)

    def test_fold_rebinds_representative_probabilities(self):
        model = Predictor(GROUPS, "per_layer", C)
        teacher = torch.rand(6, 20)
        teacher /= teacher.sum(-1, keepdim=True)
        record = dict(layer=0, start=17, q=torch.randn(3, 6, 4), k=torch.randn(20, 2, 4),
                      teacher_p=teacher, shared_p=torch.zeros(2, 20))
        for head, _, _, p_rep, *_ in cases(model, record):
            representative = 0 if head < 3 else 3
            torch.testing.assert_close(p_rep, teacher[representative])

    def test_fixed_group_protocol_keeps_nn_holdout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root/'reference.jsonl'
            reference.write_text(json.dumps({'input_ids_sha256': 'f'*64})+'\n')
            entries = []
            for i in range(3):
                entries.append(export_record(root/f'{i}.pt', q_tile=torch.randn(3, 6, 4),
                    k=torch.randn(20, 2, 4), groups=GROUPS, layer=0, start=17, input_tokens=20,
                    input_ids_sha256=hashlib.sha256(str(i).encode()).hexdigest(), family_id=f'kv-{i}'))
            groups = root/'groups.json'
            groups.write_text(json.dumps(dict(layers=GROUPS, calibration_sample_ids=['kv-0','kv-1','kv-2'])))
            manifest = dict(schema='learned_mask_dense_teacher_v1',
                protocol='leave_one_input_out_fixed_groups',
                grouping_scope='historical_fixed_all_calibration', grouping_independent_of_validation=False,
                groups_sha256=hashlib.sha256(groups.read_bytes()).hexdigest(),
                protected_reference=str(reference), protected_reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                splits=dict(train=entries[:2], validation=entries[2:]))
            path = root/'manifest.json'
            path.write_text(json.dumps(manifest))
            checked = validate_manifest(path)
            self.assertEqual(validate_group_provenance(checked, groups), GROUPS)
            manifest['grouping_independent_of_validation'] = True
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'independent grouping'):
                validate_manifest(path)
            manifest['grouping_independent_of_validation'] = False
            manifest['splits']['validation'][0]['input_ids_sha256'] = entries[0]['input_ids_sha256']
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'leaks across splits'):
                validate_manifest(path)

    def test_cv_no_fake_test_and_group_fit_leakage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "reference.jsonl"
            reference.write_text(json.dumps({"input_ids_sha256": "f" * 64}) + "\n")
            entries = []
            for i in range(3):
                entries.append(export_record(root/f"{i}.pt", q_tile=torch.randn(3, 6, 4),
                    k=torch.randn(20, 2, 4), groups=GROUPS, layer=0, start=17, input_tokens=20,
                    input_ids_sha256=hashlib.sha256(str(i).encode()).hexdigest(), family_id=f"kv-{i}"))
            groups = root / "groups.json"
            payload = dict(layers=GROUPS, calibration_sample_ids=["kv-0", "kv-1"])
            groups.write_text(json.dumps(payload))
            manifest = dict(schema="learned_mask_dense_teacher_v1", protocol="leave_one_input_out",
                protected_reference=str(reference),
                protected_reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                group_fit_family_ids=["kv-0", "kv-1"],
                groups_sha256=hashlib.sha256(groups.read_bytes()).hexdigest(),
                splits=dict(train=entries[:2], validation=entries[2:]))
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest))
            validate_manifest(path)
            validate_group_provenance(manifest, groups)
            payload["calibration_sample_ids"].append("kv-2")
            groups.write_text(json.dumps(payload))
            manifest["groups_sha256"] = hashlib.sha256(groups.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "held-out input"):
                validate_group_provenance(manifest, groups)
            manifest["splits"]["test"] = entries[2:]
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "Split names"):
                validate_manifest(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
