import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import prepare
import protocol
from test_protocol import fixtures


class ReuseTests(unittest.TestCase):
    def test_only_methods_and_provenance_may_change(self):
        old,_,_=fixtures()
        old['methods']=['dense_full','shared_map']
        new=dict(old,methods=list(protocol.METHODS),preparation_sources={},
            reused_input_manifest_sha256=protocol.ORIGIN_MANIFEST,origin_audit_sha256=protocol.ORIGIN_AUDIT)
        protocol.validate_reuse(new,old)
        for field in ('input_ids_sha256','answers','source_index'):
            bad=copy.deepcopy(new)
            bad['records'][0][field]='changed'
            with self.assertRaises(AssertionError): protocol.validate_reuse(bad,old)

    def test_clone_bytes_and_exclusive_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            src=root/'origin'
            src.mkdir()
            for name in ('inputs','data','official'):
                (src/name).mkdir()
            old,_,_=fixtures()
            old['methods']=['dense_full','shared_map']
            for row in old['records']:
                for suffix,field in {'ids':'input_ids_sha256','pos':'position_ids_sha256','mask':'attention_mask_sha256',
                    'prompt':'rendered_prompt_sha256','raw':'raw_row_sha256'}.items():
                    p=src/'inputs'/(row['key']+'.'+suffix)
                    p.write_bytes((row['key']+suffix).encode())
                    row[field]=protocol.sha(p)
            (src/'inputs/manifest.json').write_text(json.dumps(old))
            (src/'independent-audit.json').write_text('{}')
            mh=protocol.sha(src/'inputs/manifest.json')
            ah=protocol.sha(src/'independent-audit.json')
            with patch.object(prepare,'ORIGIN_MANIFEST',mh),patch.object(protocol,'ORIGIN_MANIFEST',mh),\
                 patch.object(prepare,'ORIGIN_AUDIT',ah),patch.object(protocol,'ORIGIN_AUDIT',ah):
                dest=root/'new'
                result=prepare.clone_inputs(src,dest)
                self.assertEqual(result['records'],old['records'])
                for p in (src/'inputs').iterdir():
                    if p.name!='manifest.json':
                        self.assertEqual(p.read_bytes(),(dest/'inputs'/p.name).read_bytes())
                with self.assertRaises(FileExistsError): prepare.clone_inputs(src,dest)

    def test_origin_hash_mismatch_rejected_before_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'inputs').mkdir()
            (root/'inputs/manifest.json').write_text('{}')
            with self.assertRaises(AssertionError): prepare.clone_inputs(root,root/'new')
            self.assertFalse((root/'new').exists())


if __name__=='__main__': unittest.main()
