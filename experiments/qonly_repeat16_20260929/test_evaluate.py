import copy
import unittest
from config import METHODS, SPARSE_LAYERS
from evaluate import selected, validate_rows, projection


class EvaluationTests(unittest.TestCase):
    def sample(self):
        item = dict(source_index=331,input_tokens=7,input_ids_sha256='x',answers=['abc'],raw_row_sha256='y')
        rows = [dict(**item,method=m,generated_ids=[2],generated_tokens=1,full_kv_cache=True,
            finite_all_steps=True,dense_decode=True,historical_token_ids_equal=True,
            routed_layer_eligible_pairs={} if m=='dense' else {i:28 for i in SPARSE_LAYERS}) for m in METHODS]
        return item,rows

    def test_identity_count_and_layer_negative_gates(self):
        item,rows = self.sample()
        validate_rows(rows,[item])
        for field,value in [('input_ids_sha256','bad'),('generated_tokens',2),
                            ('finite_all_steps',False),('routed_layer_eligible_pairs',{})]:
            wrong=copy.deepcopy(rows)
            wrong[1][field]=value
            with self.assertRaises((ValueError,AssertionError)):
                validate_rows(wrong,[item])
        with self.assertRaises(AssertionError):
            validate_rows(rows[:-1]+[rows[0]],[item])

    def test_smoke_disjoint_remaining(self):
        manifest=dict(records=[dict(source_index=i) for i in [331,61,16,7,9]])
        self.assertEqual([r['source_index'] for r in selected(manifest,'smoke')],[331,61,16])
        self.assertEqual([r['source_index'] for r in selected(manifest,'remaining')],[7,9])
        self.assertEqual(selected(manifest,None),manifest['records'])

    def test_budget_uses_worst_per_method_and_reserve(self):
        rows=[dict(method=m,prefill_seconds=s,decode_seconds=2) for m in METHODS for s in (1,3)]
        self.assertEqual(projection(100,rows),100+1.5*47*15+1200)


if __name__=='__main__':
    unittest.main(verbosity=2)
