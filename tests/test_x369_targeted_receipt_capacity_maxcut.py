"""Source-free acceptance of the expanded native receipt and cap max-cut."""
import unittest
from pathlib import Path
from scripts.x369_targeted_receipt_capacity_maxcut import (
    certificate, minimum_projections, union_receipts)

ROOT=Path(__file__).resolve().parents[1]/'docs/receipts'
ORIG=ROOT/'x369_original_native_minimal_consumer_20261128.json'
TARGET=ROOT/'x369_original_targeted_semantic_20261128.json'


class TargetedCapacityMaxcut(unittest.TestCase):
    def test_combined_native_receipts(self):
        r=certificate(ORIG.read_bytes(),TARGET.read_bytes())
        self.assertEqual((r['observed_states'],r['observed_edges']),(107,103))
        self.assertEqual(r['native_comparisons']['constructed_semantic_six'],36)
        c=r['native_separation_counts']
        self.assertEqual(c['consumer_separation'],36)
        self.assertEqual(c['successor_chart_separation'],12)
        self.assertEqual(c['ids'],25)
        self.assertEqual(c['local_signs'],16)
        for f in ('logits_0','logits_1','logits_2','output','weights'):
            self.assertEqual(c[f],36)
        p=r['minimum_sample_injective_cap_projection']
        self.assertEqual(p['minimum_extra_coordinates'],1)
        self.assertEqual(p['minimal_subsets'],[(12,),(13,)])
        self.assertEqual(p['same_six_pairs'],169)
        self.assertEqual(p['maximum_six_fibre'],9)
        self.assertEqual(p['pair_difference_masks'],{15360:79,13056:57,16128:33})
        self.assertTrue(r['all_observed_histories_reopen'])
        self.assertFalse(r['full_source_sufficiency_certified'])
        self.assertFalse(r['finite_source_graph_closure_certified'])

    def test_pinned_receipt_tamper(self):
        a,b=ORIG.read_bytes(),TARGET.read_bytes()
        for x,y in ((a+b' ',b),(a,b+b' ')):
            with self.assertRaises(ValueError):union_receipts(x,y)

    def test_true_capacity_alias_blocks_any_coordinate_subset(self):
        a=((0,)*6,((0,)*6,((1,2),(3,4))))
        r=minimum_projections({'a':a,'b':a})
        self.assertFalse(r['sample_injective'])
        self.assertIsNone(r['minimum_extra_coordinates'])
        self.assertEqual(r['minimal_subsets'],[])


if __name__=='__main__':
    unittest.main()
