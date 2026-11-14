"""Fail-closed F2 source-token action and label-claim regressions."""
import unittest
import numpy as np
from scripts.certify_c3_task_orbits_original_tokens import (
    cyclic_token_action,certify_orbit_action,RULE)

def specimen(seed):
    base=np.array([seed,seed+1,seed+2,seed+3,seed+4,seed+5],dtype=np.int64)
    p=[1,2,3]
    a=cyclic_token_action(base,p)
    b=cyclic_token_action(a,p)
    return np.stack((base,a,b))

def source():
    fit=np.stack([specimen(i*10) for i in range(3)])
    ev=np.stack([specimen(i*10+100) for i in range(3)])
    cert={'rule':RULE,'output_rule':'plus_one_mod_three_v1',
          'token_positions':[1,2,3],
          'fit_phase_labels':[[0,1,2],[1,2,0],[2,0,1]],
          'eval_phase_labels':[[0,1,2]]*3}
    return fit,ev,{'machine_action_certificate':cert}

class C3ActionCertificateTests(unittest.TestCase):
    def test_valid_structural_only(self):
        fit,ev,manifest=source()
        r=certify_orbit_action(fit,ev,manifest)
        self.assertTrue(r['machine_action_structurally_verified'])
        self.assertEqual(r['source_orbit_groups_checked'],6)
        self.assertFalse(r['semantic_label_correctness_independently_verified'])
        self.assertFalse(r['learned_original_model_c3_circuit_proved'])
    def test_absent_checker_remains_unverified(self):
        fit,ev,_=source()
        self.assertFalse(certify_orbit_action(fit,ev,{})['machine_action_structurally_verified'])
    def test_forged_input_action_fails(self):
        fit,ev,manifest=source()
        ev[1,2,4]+=1
        with self.assertRaisesRegex(ValueError,'does not close'):
            certify_orbit_action(fit,ev,manifest)
    def test_forged_label_action_fails(self):
        fit,ev,manifest=source()
        manifest['machine_action_certificate']['eval_phase_labels'][1]=[0,1,1]
        with self.assertRaisesRegex(ValueError,'output phase action'):
            certify_orbit_action(fit,ev,manifest)
    def test_duplicate_positions_fail(self):
        fit,ev,manifest=source()
        manifest['machine_action_certificate']['token_positions']=[1,1,2]
        with self.assertRaisesRegex(ValueError,'distinct'):
            certify_orbit_action(fit,ev,manifest)
    def test_trivial_orbit_fails(self):
        fit,ev,manifest=source()
        ev[0]=np.repeat(np.array([[4,2,2,2,8,9]]),3,axis=0)
        with self.assertRaisesRegex(ValueError,'trivial'):
            certify_orbit_action(fit,ev,manifest)

if __name__=='__main__':unittest.main()
