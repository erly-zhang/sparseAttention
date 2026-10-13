"""Exercise the actual pinned official scorer against every historical result."""
import ast
import collections
import unittest
from support import HERE,read,official_scorer

class ScoringInterface(unittest.TestCase):
    def test_all_historical_scores(self):
        counts=collections.Counter()
        for reference in read(HERE/'prepared.json')['references']:
            row=read(reference['path'])
            score=official_scorer(row['task'])(row['prediction'],row['answers'],'llama')
            self.assertAlmostEqual(float(score),float(row['score']),places=12)
            counts[row['task']]+=1
        self.assertEqual(dict(counts),{task:350 for task in read(HERE/'config.json')['tasks']})

    def test_generation_and_history_call_validated_scorer(self):
        tree=ast.parse((HERE/'execute.py').read_text())
        for name in ['history','generate']:
            function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
            calls=[n for n in ast.walk(function) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='official_scorer']
            self.assertEqual(len(calls),1)
            self.assertEqual(len(calls[0].args),1)
        unsafe=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='official' and not n.args]
        self.assertFalse(unsafe)

if __name__=='__main__':unittest.main(verbosity=2)
