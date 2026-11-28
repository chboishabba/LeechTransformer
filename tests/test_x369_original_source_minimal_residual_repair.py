"""Consumer-specific minimal residual repair uses original exact refiner."""
import json
import unittest
from scripts.x369_original_source_minimal_residual_repair import (
    residual_repair,repair_given_stable_partition,certificate)

STATES=tuple(bytes((i,)) for i in range(4))
SEEDS=(STATES[0],STATES[1])

def source(h,u):
    if u!='tick':raise ValueError('original control')
    i=h[0]
    return (('original_consumer',i),),STATES[(i+2)%4]

class OriginalMinimalResidualRepair(unittest.TestCase):
    def test_minimal_one_bit_residual_within_each_coarse_fibre(self):
        r=residual_repair(STATES,SEEDS,('tick',),lambda h:h[0]%2,source)
        self.assertEqual(r['source_closed_finite_domain'],4)
        self.assertEqual(r['original_coarse_fibres'],2)
        self.assertEqual(r['exact_minimal_future_consumer_classes'],4)
        self.assertEqual(r['maximal_required_residual_classes_in_one_fibre'],2)
        self.assertEqual(r['minimal_worst_case_extra_binary_bits'],1)
        self.assertEqual(r['original_source_replay']['edges'],4)
        self.assertTrue(r['original_source_replay']['minimal'])
        self.assertEqual(len(r['exact_source_fibre_rows']),4)
        self.assertFalse(r['actual_original_history_bytes_losslessly_recovered'])
        self.assertFalse(r['original_unrestricted_float_source_certified'])
        # Must be suitable for the original immutable native JSON runner.
        json.dumps(r,sort_keys=True)

    def test_full_original_bytes_requires_zero_additional_residual(self):
        r=residual_repair(STATES,SEEDS,('tick',),lambda h:h,source)
        self.assertEqual(r['original_coarse_fibres'],4)
        self.assertEqual(r['maximal_required_residual_classes_in_one_fibre'],1)
        self.assertEqual(r['minimal_worst_case_extra_binary_bits'],0)

    def test_constant_coarse_requires_two_extra_bits(self):
        r=residual_repair(STATES,SEEDS,('tick',),lambda h:0,source)
        self.assertEqual(r['original_coarse_fibres'],1)
        self.assertEqual(r['maximal_required_residual_classes_in_one_fibre'] if
            'maximal_required_residual_classes_in_one_fibre' in r else
            r['maximal_required_residual_classes_in_one_fibre'],4)
        self.assertEqual(r['minimal_worst_case_extra_binary_bits'],2)

    def test_unclosed_original_source_refused(self):
        with self.assertRaises(ValueError):
            residual_repair(STATES[:2],SEEDS,('tick',),
                            lambda h:h[0]%2,source)
        with self.assertRaises(ValueError):
            repair_given_stable_partition(STATES,
               {STATES[0]:0},{STATES[0]:0})

    def test_existing_exact_fixture_receipt(self):
        r=certificate()
        self.assertEqual(r['minimum_extra_bits_per_fibre_worst_case'],1)
        self.assertFalse(r['source_neural_checkpoint_executed'])

if __name__=='__main__':
    unittest.main()
