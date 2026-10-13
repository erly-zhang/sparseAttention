"""CPU retry policy checks; no submission, GPU import or inference."""
import unittest
from monitor import retry_candidates

class RetryPolicy(unittest.TestCase):
    def test_only_transient_workers_retry(self):
        rows=[['12_0','TIMEOUT','0:0'],['12_1','FAILED','1:0'],['12_2','NODE_FAIL','1:0'],['12_3','CANCELLED','0:0']]
        self.assertEqual(retry_candidates(rows,[],{},set()),([0,2],[]))
        self.assertEqual(retry_candidates(rows,[],{}, {0,2}),([],[]))

    def test_active_retry_and_bound(self):
        old=[['12_0','TIMEOUT','0:0']]
        self.assertEqual(retry_candidates(old,[['13_0','RUNNING','0:0']],{'0':['13']},set()),([],[]))
        self.assertEqual(retry_candidates(old,[['13_0','TIMEOUT','0:0']],{'0':['13']},set()),([0],[]))
        self.assertEqual(retry_candidates(old,[['14_0','TIMEOUT','0:0']],{'0':['13','14']},set()),([],[0]))
        self.assertEqual(retry_candidates(old,[['13_0','FAILED','1:0']],{'0':['13']},set()),([],[0]))

if __name__=='__main__':unittest.main(verbosity=2)
