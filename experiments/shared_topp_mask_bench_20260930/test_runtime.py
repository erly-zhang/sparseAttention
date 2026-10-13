import unittest
from evaluate import projection
from protocol import METHODS


class RuntimeTests(unittest.TestCase):
    def test_projection_uses_both_methods(self):
        rows=[dict(method=m,prefill_seconds=100,input_tokens=1000,decode_seconds=10,generated_ids=[1]*11) for m in METHODS]
        value=projection(rows,[dict(input_tokens=2000,max_new_tokens=21)],500)
        self.assertEqual(value,500+1.5*(2*(400+20))+1800)

    def test_projection_does_not_assume_short_generation(self):
        rows=[dict(method=m,prefill_seconds=100,input_tokens=1000,decode_seconds=1,generated_ids=[1]) for m in METHODS]
        self.assertGreater(projection(rows,[dict(input_tokens=1000,max_new_tokens=50)],500),
                           projection(rows,[dict(input_tokens=1000,max_new_tokens=5)],500))

    def test_script_resource_bounds(self):
        from pathlib import Path
        script=(Path(__file__).resolve().parent/'run.sbatch').read_text()
        for option in ('--array=0-3%4','--gres=gpu:h100-80:1','--cpus-per-task=8','--mem=128G','--time=24:00:00','--no-requeue'):
            self.assertIn('#SBATCH '+option,script)


if __name__=='__main__':
    unittest.main()
