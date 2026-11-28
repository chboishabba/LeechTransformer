"""Tests for exact cap fibre reflections and native scope guards."""
import struct
import unittest
from pathlib import Path

from scripts.x369_original_capacity_fibre_reflections import reflect_triads, certificate
from scripts.x369_original_receipt_multichart_weld import chart_values
from scripts.x369_source_acceptance import parse_args, stage_plan

ROOT=Path(__file__).resolve().parents[1]/'docs/receipts'
ORIG=ROOT/'x369_original_native_minimal_consumer_20261128.json'
TARGET=ROOT/'x369_original_targeted_semantic_20261128.json'
CAP='semantic_plus_cap_profile'


class CapacityReflections(unittest.TestCase):
    def test_signed_triads_keep_full_cap_and_reopen(self):
        for dtype,fmt in (('float32','f'),('float64','d')):
            cfg=dict(experts=4,width=12,dtype=dtype)
            h=struct.pack('<96'+fmt,*([1.,-2.,3.]*32))
            q=chart_values(h,cfg)[CAP]
            aliases=list(reflect_triads(h,cfg))
            self.assertEqual(len(aliases),26)
            for ref,b in aliases:
                self.assertNotEqual(b,h)
                self.assertEqual(chart_values(b,cfg)[CAP],q)
                self.assertEqual(dict(reflect_triads(b,cfg))[ref],h)
            self.assertEqual(list(reflect_triads(bytes(len(h)),cfg)),[])

    def test_real_receipt_aliases_do_not_promote_reachability(self):
        r=certificate(ORIG.read_bytes(),TARGET.read_bytes())
        self.assertEqual(r['observed_histories'],107)
        self.assertGreater(r['exact_reflection_pairs'],0)
        self.assertEqual(r['aliases_with_recorded_source_path'],0)
        self.assertTrue(r['chart_preserved_exactly'])
        self.assertTrue(r['constructed_alias_reachability_not_certified'])
        self.assertFalse(r['consumer_descent_certified'])

    def test_native_scope_requires_both_receipts(self):
        base=['--checkpoint','not-read.pt','--sha256','0'*64,'--source-head','0'*40,
              '--output','not-written.json','--fibre-budget','189',
              '--only-stages','admission','indexed_fibre',
              '--receipt-capacity-fibre',str(TARGET)]
        with self.assertRaises(SystemExit):parse_args(base)
        a=parse_args(base+['--receipt-targeted-multichart',str(ORIG)])
        self.assertEqual(stage_plan(a),('admission','indexed_fibre'))
        with self.assertRaises(SystemExit):
            parse_args(base+['--receipt-targeted-multichart',str(ORIG),'--source-witness-seeds','4'])


if __name__=='__main__':
    unittest.main()
