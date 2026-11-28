"""Verified original AMD gfx803 receipt proves 4-way consumer residual lower bound."""
import copy
import json
import unittest
from pathlib import Path
from scripts.x369_original_receipt_four_way_residual_bound import (
    certify_four_way_original_source_residual)

PATH=Path(__file__).resolve().parents[1]/'docs'/'receipts'/(
    'x369_original_native_minimal_consumer_20261128.json')

class ActualNativeFourWayResidual(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.receipt=json.loads(PATH.read_text())

    def test_all_six_original_source_pairs_separate(self):
        cert=certify_four_way_original_source_residual(self.receipt)
        self.assertEqual(cert['original_source_history_count'],4)
        self.assertEqual(cert['all_pairs_checked'],6)
        self.assertEqual(cert['at_least_residual_labels_in_unmarked_fibre'],4)
        self.assertEqual(cert['at_least_fixed_binary_residual_bits_in_unmarked_fibre'],2)
        self.assertEqual(cert['chronological_controls'],[[8,0],[4,1]])
        self.assertEqual(cert['distinct_original_readouts_by_consumer']['logits_0'],4)
        self.assertEqual(cert['distinct_original_readouts_by_consumer']['ids'],2)
        self.assertEqual(cert['distinct_original_readouts_by_consumer']['local_signs'],1)
        self.assertTrue(all('logits_0' in p['differing_original_consumers']
                            for p in cert['actual_source_pairwise_readout_separators']))
        self.assertTrue(cert['lower_bound_relative_to_this_coarse_chart'])
        self.assertFalse(cert['full_source_global_residual_minimum_claim'])
        self.assertFalse(cert['original_gpu_model_newly_replayed'])
        json.dumps(cert,sort_keys=True)

    def test_tampered_original_second_step_history_refused(self):
        copy_receipt=copy.deepcopy(self.receipt)
        rows=copy_receipt['stages']['indexed_fibre'][
            'original_arbitrary_word_source_residual'][
            'exact_source_word_transport_receipts']
        rows[2]['receipt']['original_step_receipts'][1][
            'input_source_sha256']='0'*64
        with self.assertRaises(ValueError):
            certify_four_way_original_source_residual(copy_receipt)

    def test_original_consumer_portfolio_missing_refused(self):
        copy_receipt=copy.deepcopy(self.receipt)
        rows=copy_receipt['stages']['indexed_fibre'][
            'original_arbitrary_word_source_residual'][
            'exact_source_word_transport_receipts']
        del rows[2]['receipt']['original_step_receipts'][1][
            'original_source_observations']['logits_0']
        with self.assertRaises(ValueError):
            certify_four_way_original_source_residual(copy_receipt)

if __name__=='__main__':
    unittest.main()
