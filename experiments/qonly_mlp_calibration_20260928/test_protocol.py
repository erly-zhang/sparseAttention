import copy
import json
import tempfile
from pathlib import Path
import unittest
from protocol import specs, make_prompt, validate_inputs, exact_match, save


class SyntheticTests(unittest.TestCase):
    def test_split_seeds_lengths_depths(self):
        rows = specs()
        self.assertEqual(len(rows), 48)
        self.assertEqual(len({r['seed'] for r in rows}), 48)
        for split, count in [('train', 32), ('validation', 8), ('heldout', 8)]:
            group = [r for r in rows if r['split'] == split]
            self.assertEqual(len(group), count)
            self.assertEqual(sum(r['length_bin'] == 4096 for r in group), count // 2)
            self.assertEqual(len({(r['length_bin'], r['depth']) for r in group}), 8)

    def test_determinism_and_exclusion(self):
        # A character tokenizer exercises packing and identity gates, not Llama parity.
        encode = lambda text: list(text.encode())
        rows = []
        for spec in specs():
            row = make_prompt(spec, encode)
            self.assertEqual(row, make_prompt(spec, encode))
            row = json.loads(json.dumps(row))
            row['input_tokens'] = len(row.pop('ids'))
            row['input_ids_sha256'] = row['id']
            rows.append(row)
        self.assertTrue(validate_inputs(rows))
        with self.assertRaises(AssertionError):
            validate_inputs(rows, [rows[0]['input_ids_sha256']])
        bad = copy.deepcopy(rows)
        bad[-1]['pairs'][0] = bad[0]['pairs'][0]
        with self.assertRaises(AssertionError):
            validate_inputs(bad)

    def test_exact_match_no_substring_credit(self):
        self.assertEqual(exact_match(' "abc"\n', 'abc'), 1)
        self.assertEqual(exact_match('abcabc', 'abc'), 0)
        self.assertEqual(exact_match('The answer is abc', 'abc'), 0)

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'a.json'
            save(path, {'x': 1})
            with self.assertRaises(FileExistsError):
                save(path, {'x': 2})
            self.assertEqual(json.loads(path.read_text()), {'x': 1})


if __name__ == '__main__':
    unittest.main()
