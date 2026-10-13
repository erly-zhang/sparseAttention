import copy
import tempfile
import unittest
from pathlib import Path
import protocol as p


class ProtocolTests(unittest.TestCase):
    def rows(self):
        return [dict(s, input_tokens=s['length_bin'], input_ids_sha256=str(i),
                     pairs=[[f'key{i}', f'value{i}']], target_row=0,
                     target_key=f'key{i}', answer=f'value{i}',
                     feature_positions=p.positions(s['length_bin']))
                for i, s in enumerate(p.specs())]

    def test_fresh_split(self):
        rows = self.rows()
        p.validate_inputs(rows)
        self.assertEqual(len(p.LAYERS), 16)
        self.assertEqual([sum(r['split']==s for r in rows) for s in p.SPLITS], [32,8,8])
        with self.assertRaises(AssertionError):
            p.validate_inputs(rows, excluded_keys=['key0'])
        with self.assertRaises(AssertionError):
            p.validate_inputs(rows, excluded_hashes=['0'])
        rows[-1]['pairs'] = rows[0]['pairs']
        with self.assertRaises(AssertionError):
            p.validate_inputs(rows)

    def test_positions(self):
        for n in (256,4096,16384):
            ix = p.positions(n)
            self.assertEqual(len(set(ix)),256)
            self.assertEqual(ix[-64:],list(range(n-64,n)))
            self.assertTrue(all(0<=i<n for i in ix))

    def test_selection_identity_and_budget(self):
        history = [dict(epoch=0,validation_loss=1),dict(epoch=1,validation_loss=1)]
        self.assertEqual(p.pick_checkpoint(history)['epoch'],0)
        history.append(dict(epoch=2,validation_loss=float('nan')))
        with self.assertRaises(ValueError):
            p.pick_checkpoint(history)
        self.assertGreater(p.projection(100,10,100,100,10),p.DEADLINE)
        self.assertLess(p.projection(100,1,1,1,1),p.DEADLINE)

    def test_results_identity(self):
        manifest=self.rows()
        rows=[]
        for item in manifest:
            if item['split']!='heldout':
                continue
            for method in p.METHODS:
                rows.append(dict(id=item['id'],method=method,input_ids_sha256=item['input_ids_sha256'],
                    correct=1,prediction=item['answer'],generated_ids=[1],full_kv_cache=True,
                    finite_all_steps=True,dense_decode=True,pairs={} if method=='dense' else
                    {i:item['input_tokens']*(item['input_tokens']+1)//2 for i in p.LAYERS}))
        p.validate_results(rows,manifest)
        bad=copy.deepcopy(rows)
        bad[-1]=bad[0]
        with self.assertRaises(AssertionError):
            p.validate_results(bad,manifest)

    def test_exclusive_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'result.json'
            p.save(path,dict(status='passed'))
            with self.assertRaises(FileExistsError):
                p.save(path,{})


if __name__=='__main__':
    unittest.main()
