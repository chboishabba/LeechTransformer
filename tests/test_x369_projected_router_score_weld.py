"""X369 finite polynomial -> actual learned head -> top-k source bridge."""
import unittest
import torch

from models.history_orbit_moe import HistoryOrbitMoE
from scripts.audit_x369_projected_router_score_weld import (
    projected_original_c3_router_weld)


class X369ProjectedOriginalScoreWeld(unittest.TestCase):
    def sample(self):
        g=torch.Generator().manual_seed(369)
        x=torch.randn(9,24,generator=g,dtype=torch.float64)
        h=torch.randn(9,4,24,generator=g,dtype=torch.float64)
        w=torch.randn(48,generator=g,dtype=torch.float64)
        b=torch.randn(4,generator=g,dtype=torch.float64)
        return x,h,w,b

    def test_source_polynomial_projection_and_strict_topk(self):
        x,h,w,b=self.sample()
        audit=projected_original_c3_router_weld(
            x,h,w,expert_bias=b,top_k=2)
        self.assertLess(
            audit["relative_c3_directed_feature_energy_max_abs_error"],1e-10)
        self.assertLess(audit["directed_feature_zero_sum_max_abs_error"],1e-10)
        self.assertLess(audit["weight_projection_score_max_abs_error"],1e-10)
        self.assertLess(audit["full_source_logit_projection_max_abs_error"],1e-10)
        self.assertLess(audit["source_score_bound_max_violation"],1e-10)
        self.assertGreater(
            audit["source_phase_feature_displacement_mean"],.1)
        self.assertTrue(
            audit["same_topk_set_does_not_imply_equal_mixture_weights"])
        self.assertFalse(audit["grokking_semantic_action_verified"])
        self.assertFalse(audit["sophontic_intervention_verified"])
        self.assertFalse(audit["triadic_face_inferred"])

    def test_large_feature_energy_can_be_invisible_to_learned_router(self):
        x,h,w,_=self.sample()
        w[:]=0.
        # Each triad's (self,cyclic,anti) head has only its DC mode:
        # delta T is nonzero but orthogonal to this entire head.
        w[24:]=1.
        bias=torch.tensor([20.,10.,0.,-10.],dtype=torch.float64)
        a=projected_original_c3_router_weld(
            x,h,w,expert_bias=bias,top_k=2)
        self.assertGreater(a["source_phase_feature_displacement_mean"],.1)
        self.assertLess(a["source_phase_projected_logit_delta_mean_abs"],1e-10)
        self.assertEqual(
            a["fraction_strict_topk_set_stability_certified"],1.)
        self.assertEqual(
            a["fraction_original_topk_set_actually_changed"],0.)

    def test_one_triad_changes_expert_and_margin_predicts_not_certified(self):
        x=torch.tensor([[1.,0.,0.]],dtype=torch.float64)
        h=torch.tensor([[[1.,0.,0.],[0.,1.,0.]]],
                       dtype=torch.float64)
        w=torch.tensor([0.,0.,0.,1.,0.,0.],dtype=torch.float64)
        a=projected_original_c3_router_weld(x,h,w,top_k=1)
        self.assertAlmostEqual(
            a["fraction_original_topk_set_actually_changed"],1.)
        self.assertAlmostEqual(
            a["fraction_strict_topk_set_stability_certified"],0.)
        # A sufficiently separated expert offset prevents TopK changes;
        # it does not imply constant softmax weights at top_k>1.
        b=projected_original_c3_router_weld(
            x,h,w,expert_bias=torch.tensor([10.,0.],dtype=torch.float64),
            top_k=1)
        self.assertEqual(
            b["fraction_strict_topk_set_stability_certified"],1.)
        self.assertEqual(
            b["fraction_original_topk_set_actually_changed"],0.)

    def test_original_history_orbit_moe_route_selection_weld(self):
        torch.manual_seed(41)
        moe=HistoryOrbitMoE(d_model=24,num_experts=3,top_k=1,
                            route_geometry="c2c3_wave").double().eval()
        g=torch.Generator().manual_seed(98)
        x=torch.randn(12,24,generator=g,dtype=torch.float64)
        h=torch.randn(12,3,24,generator=g,dtype=torch.float64)
        old,_=moe.select_source_experts(x,h)
        with torch.no_grad():
            moe.route_probe_action="c3_relative_shift"
            new,_=moe.select_source_experts(x,h)
            moe.route_probe_action=None
        a=projected_original_c3_router_weld(
            x,h,moe.router.weight.detach(),
            expert_bias=moe.expert_bias.detach(),top_k=1)
        changed=(old!=new).any(-1).double().mean()
        self.assertAlmostEqual(
            a["fraction_original_topk_set_actually_changed"],
            float(changed),places=12)

    def test_invalid_source_and_bias_rejected(self):
        x,h,w,b=self.sample()
        with self.assertRaises(ValueError):
            projected_original_c3_router_weld(x,h,w[:-1],expert_bias=b)
        with self.assertRaises(ValueError):
            projected_original_c3_router_weld(x,h,w,expert_bias=b[:-1])
        with self.assertRaises(ValueError):
            projected_original_c3_router_weld(x,h,w,top_k=4)
        with self.assertRaises(ValueError):
            projected_original_c3_router_weld(x,h[...,1:],w)
        with self.assertRaises(ValueError):
            projected_original_c3_router_weld(
                x,torch.full_like(h,float("nan")),w)


if __name__=="__main__":
    unittest.main()
