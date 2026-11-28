"""Source-free orbit proof obligations and genuine path scope guards."""
import struct
import unittest
from pathlib import Path

from models.geometric_reflection_fibre import GROUP, mul, inv
from scripts.x369_reachable_capacity_orbit_weld import (
    act_bytes, numeric_key, conjugate, stabilizer, public_literal, certificate)
from scripts.x369_source_acceptance import parse_args, stage_plan

ROOT=Path(__file__).resolve().parents[1]/'docs/receipts'
ORIG=ROOT/'x369_original_native_minimal_consumer_20261128.json'
TARGET=ROOT/'x369_original_targeted_semantic_20261128.json'


class ReachableCapacityOrbit(unittest.TestCase):
    def test_exact_group_law_and_signed_zero(self):
        for dtype,fmt,word in (('float32','f','I'),('float64','d','Q')):
            cfg=dict(experts=4,width=12,dtype=dtype)
            h=struct.pack('<96'+fmt,*([0.,-0.,1.,-2.,3.,4.]*16))
            for g in GROUP:
                self.assertEqual(act_bytes(inv(g),act_bytes(g,h,cfg),cfg),h)
                for k in GROUP:
                    self.assertEqual(act_bytes(g,act_bytes(k,h,cfg),cfg),
                                     act_bytes(mul(g,k),h,cfg))
                    for u in ((8,0),(4,1),(10,2),(1,3)):
                        self.assertEqual(conjugate(g,conjugate(k,u)),
                                         conjugate(mul(g,k),u))
            zero=bytes(len(h))
            signed=act_bytes(GROUP[1],zero,cfg)
            self.assertNotEqual(zero,signed)
            self.assertEqual(numeric_key(zero,cfg),numeric_key(signed,cfg))
            self.assertEqual(stabilizer(zero,cfg),tuple(range(12)))

    def test_real_source_orbit_candidates(self):
        r=certificate(ORIG.read_bytes(),TARGET.read_bytes())
        self.assertEqual((r['observed_histories'],r['observed_transitions']),(107,103))
        self.assertEqual(r['stabilizer_size_histogram'],{1:84,2:12,3:8,6:3})
        self.assertEqual(r['nontrivial_numeric_orbit_actions'],1134)
        self.assertEqual(r['exact_cap_preserving_nontrivial_orbit_actions'],8)
        self.assertTrue(r['all_observed_stabilizer_inclusions'])
        self.assertEqual(r['scope_counts'].get('fixed',0),0)
        self.assertEqual(sum(r['scope_counts'].values()),8)
        self.assertFalse(r['conjugate_path_equivariance_certified'])
        self.assertFalse(r['reachable_cap_descent_certified'])
        for a in r['aliases']:
            self.assertEqual(a['scope']=='public_literal',
                             all(public_literal(u) for u in a['transformed_word']))

    def test_original_literal_vs_expanded_controls(self):
        self.assertTrue(public_literal((4,0)))
        self.assertTrue(public_literal((8,1)))
        self.assertTrue(public_literal((10,2)))
        self.assertFalse(public_literal((2,2)))
        self.assertFalse(public_literal((6,2)))

    def test_native_scope_guard(self):
        args=['--checkpoint','not-read.pt','--sha256','0'*64,'--source-head','0'*40,
              '--output','not-written.json','--fibre-budget','189',
              '--only-stages','admission','indexed_fibre',
              '--receipt-reachable-capacity-orbit',str(TARGET)]
        with self.assertRaises(SystemExit):parse_args(args)
        args+=['--receipt-targeted-multichart',str(ORIG)]
        self.assertEqual(stage_plan(parse_args(args)),('admission','indexed_fibre'))
        with self.assertRaises(SystemExit):parse_args(args+['--receipt-capacity-fibre',str(TARGET)])


if __name__=='__main__':
    unittest.main()
