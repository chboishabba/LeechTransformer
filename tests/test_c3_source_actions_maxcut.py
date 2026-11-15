"""C3/C3² source-action negative-control regressions, not learned-model proofs."""
import unittest
import torch
from scripts.audit_c3_source_actions_maxcut import report,c3_relational_controls,c3_square_controls
class C3SourceActionTests(unittest.TestCase):
    def test_simultaneous_action_and_nontrivial_relative_witness(self):
        x=torch.zeros(1,24,dtype=torch.float64);h=x.clone();x[...,0]=1;h[...,0]=1
        d=c3_relational_controls(x,h)
        self.assertLess(d["diagonal_c3_max_feature_defect"],1e-12)
        self.assertGreater(d["relative_c3_max_feature_difference"],.5)
        self.assertLess(d["relative_rotation_input_norm_defect"],1e-12)
    def test_nine_group_actions_are_diagonal_invariants(self):
        x=torch.arange(1,25,dtype=torch.float64).reshape(1,24)
        h=torch.flip(x,[-1])
        d=c3_square_controls(x,h)
        self.assertEqual(d["nine_group_elements_evaluated"],9)
        self.assertLess(d["diagonal_c3_square_max_feature_defect"],1e-10)
        self.assertGreater(d["one_sided_c3_square_max_feature_difference"],1e-3)
        self.assertLess(d["wave_fourier_pair_norm_defect"],1e-10)
    def test_no_learned_claim(self):
        r=report()
        self.assertFalse(r["original_model_checkpoint_loaded"])
        self.assertFalse(r["causal_ce_measured"])
    def test_invalid_source_dimensions(self):
        with self.assertRaises(ValueError):
            c3_relational_controls(torch.zeros(1,22),torch.zeros(1,22))
        with self.assertRaises(ValueError):
            c3_square_controls(torch.zeros(1,24),torch.zeros(1,48))
if __name__=="__main__":unittest.main()
