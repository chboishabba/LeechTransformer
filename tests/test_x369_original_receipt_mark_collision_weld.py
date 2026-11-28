"""Regression: real committed native gfx803 receipt refutes coarse mark-only source consumer."""
import copy
import json
import unittest
from pathlib import Path
from scripts.x369_original_receipt_mark_collision_weld import (
    extract_original_mark_collision)

PATH=Path(__file__).resolve().parents[1]/'docs'/'receipts'/(
    'x369_original_native_minimal_consumer_20261128.json')

class ActualOriginalGfx803ReceiptCollision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original=json.loads(PATH.read_text())

    def test_original_source_non_descent_without_new_gpu_call(self):
        witness=extract_original_mark_collision(self.original)
        self.assertEqual(witness['status'],'receipt_verified_original_coarse_non_descent')
        self.assertEqual(witness['same_coarse_fibre'],'unmarked_original_recurrent_history')
        self.assertNotEqual(witness['original_first_source']['post_first_sha256'],
                            witness['original_second_source']['post_first_sha256'])
        self.assertEqual(witness['first_original_control'],[8,0])
        self.assertEqual(witness['second_original_control'],[4,1])
        self.assertIn('logits_0',witness['original_consumers_differ'])
        self.assertIn('ids',witness['original_consumers_differ'])
        self.assertNotIn('local_signs',witness['original_consumers_differ'])
        self.assertEqual(witness['original_source_bytes_each'],384)
        self.assertTrue(witness['original_consumer_factorization_through_mark_only_refuted'])
        self.assertFalse(witness['newly_replayed_original_native_gpu'])
        self.assertFalse(witness['unrestricted_float_neural_source_not_certified'] is False)
        self.assertFalse(witness['source_training_provenance_authenticated'])
        json.dumps(witness,sort_keys=True)

    def test_forged_admission_or_consumer_observation_refused(self):
        tampered=copy.deepcopy(self.original)
        tampered['stages']['admission']['native_gfx803_verified']=False
        with self.assertRaises(ValueError):
            extract_original_mark_collision(tampered)
        tampered=copy.deepcopy(self.original)
        a=tampered['stages']['indexed_fibre']['original_arbitrary_word_source_residual']
        a['exact_source_word_transport_receipts'][0]['receipt'][
            'original_step_receipts'][1]['input_source_sha256']='0'*64
        with self.assertRaises(ValueError):
            extract_original_mark_collision(tampered)

if __name__=='__main__':
    unittest.main()
