import copy
import json
import struct
import unittest
from pathlib import Path
from scripts.x369_semantic_cap_geometric_section import encode,reopen,certificate

PATH=Path(__file__).resolve().parents[1]/'docs/receipts/x369_original_native_minimal_consumer_20261128.json'


class SemanticCapBidi(unittest.TestCase):
    def test_real_receipt_all_histories(self):
        result=certificate(PATH.read_bytes())
        self.assertEqual(result['histories'],48)
        self.assertEqual(result['original_float_scalars'],96)
        self.assertEqual(result['free_rational_residual_coordinates'],76)
        self.assertEqual(result['sphere_sign_bits'],8)
        self.assertTrue(result['exact_source_byte_roundtrip'])
        self.assertFalse(result['bit_rate_reduction_certified'])
        self.assertFalse(result['consumer_minimality_certified'])

    def test_float32_and_float64_bits_including_negative_zero(self):
        for dtype,fmt,words in (
            ('float32','I',(0,0x80000000,1,0x80000001,0x7f7fffff,0xff7fffff,0x3f800000)),
            ('float64','Q',(0,0x8000000000000000,1,0x8000000000000001,0x7fefffffffffffff,
                             0xffefffffffffffff,0x3ff0000000000000))):
            cfg=dict(experts=4,width=12,dtype=dtype)
            h=struct.pack('<96'+fmt,*(words[i%len(words)] for i in range(96)))
            p=encode(h,cfg)
            self.assertEqual(reopen(json.loads(json.dumps(p))),h)
            bad=copy.deepcopy(p);bad['original_sha256']='0'*64
            with self.assertRaises(ValueError):reopen(bad)

    def test_nonfinite_and_malformed_sections_refused(self):
        cfg=dict(experts=4,width=12,dtype='float32')
        with self.assertRaises(ValueError):
            encode(struct.pack('<96f',float('nan'),*([0]*95)),cfg)
        p=encode(bytes(384),cfg);p['schema']=9
        with self.assertRaises(ValueError):reopen(p)


if __name__=='__main__':
    unittest.main()
