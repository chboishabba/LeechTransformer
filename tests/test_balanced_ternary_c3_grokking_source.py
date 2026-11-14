"""C3 grokking, balanced trit / inversion, actual source parity diagnostic controls."""
import unittest
import torch
from models.model import LeechConfig,LeechGPT
from scripts.audit_balanced_ternary_c3_grokking_fourier import (
    report,eq_defect,memorizer,phase,multiply_phase,invert_phase,rotate_phase,
    central_antipodal,dist)
from scripts.audit_c2_parity_c3_fourier_teacher_inputs import local_probe

class FourierGrokkingOriginalSourceTests(unittest.TestCase):
    def test_all_nine_exact_modular_fourier_pairs(self):
        r=report()
        self.assertEqual(len(r["finite_cases"]),9)
        self.assertLess(r["claims"]["max_fourier_multiplication_residual"],1e-12)

    def test_character_rule_not_merely_training_fit(self):
        self.assertEqual(eq_defect(lambda a,b:(a+b)%3),0)
        self.assertEqual(eq_defect(memorizer),2)
        self.assertEqual(memorizer(2,2),0)
        self.assertNotEqual(memorizer(2,2),(2+2)%3)

    def test_balanced_inversion_is_reflection_not_central_minus_I(self):
        z=phase(1)
        self.assertLess(dist(invert_phase(z),phase(-1)),1e-12)
        self.assertLess(dist(invert_phase(rotate_phase(invert_phase(z))),
                             rotate_phase(rotate_phase(z))),1e-12)
        self.assertLess(dist(central_antipodal(rotate_phase(z)),
                             rotate_phase(central_antipodal(z))),1e-12)
        self.assertGreater(dist(invert_phase(z),central_antipodal(z)),.5)

    def test_original_layer_source_even_odd_paired_identity(self):
        torch.manual_seed(97)
        cfg=LeechConfig(vocab_size=32,d_model=48,n_heads=2,n_layers=2,
            block_size=8,dropout=0.,moe_experts=4,moe_top_k=2,
            moe_expert_width=24,moe_even_width=24,
            moe_guarded_ffn=True,moe_deployed_parity=True,
            moe_route_geometry="c2_invariant")
        model=LeechGPT(cfg).eval()
        x=torch.randn(5,48)
        r=local_probe(model.blocks[0],x)
        self.assertLess(r["source_odd_parity_max_abs"],1e-5)
        self.assertLess(r["source_even_parity_max_abs"],1e-7)
        self.assertLess(abs(r["paired_cross_mean"]),1e-5)
        self.assertTrue(0 <= r["c3_wave_rotating_residual_energy_fraction"] <= 1)
        self.assertFalse(r["triad_action_is_semantic"])
if __name__=="__main__":unittest.main()
