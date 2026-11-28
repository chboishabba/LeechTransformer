"""Original native recurrent-history signed task residual; not target leakage."""
import unittest
from models.geometric_reflection_fibre import GROUP
from scripts.x369_original_marked_seed_fabric import marked_seed_records
from scripts.x369_original_source_signed_defect_transport import (
    marked_original_source_defects,source_step_exact,
    original_source_defect_packet,reopen_source_defect,certificate)

MEANS=((0,0,0),(-1,0,0),(1,0,0))
DELTAS=((0,0,0),(-1,0,0),(1,0,0))
def seed(m,d):
    return bytes(v+1 for v in m+d)
def marks():
    return marked_seed_records(MEANS,DELTAS,seed)

class OriginalSignedDefectTransport(unittest.TestCase):
    def test_original_identity_source_has_zero_bit_residual(self):
        u=(GROUP.index((0,0,0)),0)
        step=lambda h,u:((('logits_0',h),('ids',0)),h)
        r=marked_original_source_defects(marks(),step,seed,(u,),
                                         seed_budget=9,depth=2)
        self.assertEqual(r['checked_first_source_edges'],9)
        self.assertEqual(r['original_source_nonzero_canonical_defect_edges'],0)
        self.assertEqual(r['checked_two_step_paths'],9)
        self.assertEqual(r['nonzero_two_step_canonical_defects'],0)
        self.assertTrue(r['original_source_history_bytes_reopened_exact'])
        self.assertFalse(r['original_learned_equivariance_certified'])

    def test_nontrivial_original_action_does_not_assume_learned_step(self):
        u=(GROUP.index((1,1,0)),2)
        def learned(h,u):
            return (('logits_0',h),('local_signs',b'\x80')),h
        r=marked_original_source_defects(marks(),learned,seed,(u,),
                                         seed_budget=9,depth=2)
        self.assertGreater(r['original_source_nonzero_canonical_defect_edges'],0)
        self.assertIsNotNone(r['first_nonzero_original_neural_defect'])
        for row in r['full_original_byte_defect_receipts']:
            self.assertEqual(reopen_source_defect(row['first_defect']),
                             bytes.fromhex(row['first_defect']['actual_source_hex']))
            self.assertTrue(row['two_step_original_byte_reopen_exact'])
            self.assertEqual(reopen_source_defect(row['second_defect']),
                             bytes.fromhex(row['second_defect']['actual_source_hex']))

    def test_bitwise_signed_zero_is_not_erased(self):
        canonical=bytes.fromhex('0000000000000000')
        actual=bytes.fromhex('0000008000000000')
        receipt=original_source_defect_packet(
            actual,canonical,{'logits_0':b'raw'},(0,0))
        self.assertEqual(receipt['signed_nonzero_residual_bytes'],1)
        self.assertFalse(receipt['learned_step_equals_independent_task_target'])
        self.assertEqual(reopen_source_defect(receipt),actual)
        bad=dict(receipt,exact_xor_residual_hex=canonical.hex())
        with self.assertRaises(ValueError):
            reopen_source_defect(bad)

    def test_native_source_repeatability_gate(self):
        def nondeterministic(h,u):
            nondeterministic.calls+=1
            return (('obs',nondeterministic.calls),),h
        nondeterministic.calls=0
        with self.assertRaises(RuntimeError):
            source_step_exact(nondeterministic,b'\x01',(0,0))
        with self.assertRaises(ValueError):
            original_source_defect_packet(b'a',b'bc',{},(0,0))

    def test_control_and_certificate_guards(self):
        from scripts.x369_original_source_signed_defect_transport import signed_action_of_control
        self.assertEqual(signed_action_of_control((0,0)),GROUP[0])
        for bad in (3,(12,0),(0,4),(0,0,0),(-1,0)):
            with self.assertRaises(ValueError):signed_action_of_control(bad)
        self.assertTrue(certificate()['two_step_original_source_reopened'])
        with self.assertRaises(ValueError):
            marked_original_source_defects(marks(),lambda h,u:((),h),
                seed,((0,0),),seed_budget=10)
        with self.assertRaises(ValueError):
            marked_original_source_defects(marks(),lambda h,u:((),h),
                seed,((0,0),),seed_budget=1,depth=3)

if __name__=='__main__':
    unittest.main()
