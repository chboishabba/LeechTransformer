"""Native orbit receipts: scope-aware lower bounds and lossless ranked section."""
import copy
import json
import struct
import unittest
from pathlib import Path
from scripts.x369_reachable_capacity_residual_bounds import (
    certificate, receipt_union, encode, reopen, exact_step, orientation)

ROOT=Path(__file__).resolve().parents[1]/'docs/receipts'
FILES=('x369_original_native_minimal_consumer_20261128.json',
       'x369_original_targeted_semantic_20261128.json',
       'x369_reachable_capacity_orbit_native.json')


class ReachableCapacityResidualBounds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw=tuple((ROOT/p).read_bytes() for p in FILES)
        cls.cert=certificate(*cls.raw)

    def test_three_receipt_union(self):
        r=self.cert
        self.assertEqual((r['observed_states'],r['observed_edges']),(171,159))
        self.assertEqual((r['exact_payload_edges'],r['digest_only_edges']),(137,22))
        c=r['native_counts']
        self.assertEqual((c['numeric_prediction_matches'],c['byte_prediction_matches'],
                          c['numerically_distinct_anchor_alias']),(8,2,8))
        self.assertEqual((c['consumer_separation'],c['successor_chart_separation']),(32,31))
        self.assertEqual((c['ids'],c['weights'],c['local_signs']),(11,20,1))
        for f in ('logits_0','logits_1','logits_2','output'):self.assertEqual(c[f],32)
        self.assertFalse(r['transformed_path_intermediates_independently_replayed'])
        self.assertFalse(r['original_native_gpu_newly_replayed'])
        self.assertFalse(r['finite_source_graph_closure_certified'])

    def test_scope_bounds_and_orientation_rank(self):
        r=self.cert
        rows=r['scope_bounds']
        self.assertEqual([x['observed_states'] for x in rows],[131,151,171])
        self.assertEqual([x['cap_fibres'] for x in rows],[131,147,162])
        self.assertEqual([x['same_cap_pairs'] for x in rows],[0,4,10])
        self.assertEqual([x['exact_witness_pairs'] for x in rows],[0,4,9])
        self.assertEqual([x['minimum_required_residual_labels_lower_bound'] for x in rows],[1,2,3])
        self.assertEqual([x['fixed_width_residual_bits_lower_bound'] for x in rows],[0,1,2])
        for row in rows:
            self.assertTrue(row['sample_injective_rank'])
            self.assertTrue(row['exact_sample_label_optimality'])
            self.assertFalse(row['universal_source_sufficiency_certified'])
        self.assertTrue(r['observed_cap_fibres_share_one_numeric_orbit'])
        self.assertTrue(r['exact_group_orbit_rank_only_sample_injective'])
        self.assertEqual(r['maximum_observed_fibre_orbit_size'],3)
        self.assertEqual(r['observed_orientation_ranks'],[0,1,2])
        self.assertTrue(r['all_observed_orbit_section_roundtrips'])

    def test_exact_fallback_and_signed_zero(self):
        cfg=dict(experts=4,width=12,dtype='float32')
        h=struct.pack('<96f',*([0.,-0.,1.,-2.,3.,4.]*16))
        n=struct.pack('<96f',*([1.,-0.,0.,2.,-3.,4.]*16))
        p=encode(h,cfg)
        self.assertEqual(reopen(json.loads(json.dumps(p))),h)
        calls=[]
        def step(x,u):
            calls.append((x,u))
            return ('actual observer',),n
        o,q=exact_step(p,(8,0),step)
        self.assertEqual(calls,[(h,(8,0))])
        self.assertEqual(o,('actual observer',))
        self.assertEqual(reopen(q),n)
        bad=copy.deepcopy(p);bad['orientation_rank']=12
        with self.assertRaises(ValueError):reopen(bad)

    def test_pinned_receipt_tamper(self):
        with self.assertRaises(ValueError):
            receipt_union(self.raw[0],self.raw[1],self.raw[2]+b' ')


if __name__=='__main__':
    unittest.main()
