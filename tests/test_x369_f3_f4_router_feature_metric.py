"""Original shared-router geometry and nonlinear F3 null confound tests."""
import unittest
import torch
from scripts.audit_x369_f3_f4_router_feature_metric import (
    original_router_features,source_f3_f4_router_geometry_audit)

class X369F3F4RouterMetricTests(unittest.TestCase):
    def sample(self):
        g=torch.Generator().manual_seed(42)
        x=torch.randn(12,24,generator=g,dtype=torch.float64)
        h=torch.randn(12,4,24,generator=g,dtype=torch.float64)
        w=torch.randn(48,generator=g,dtype=torch.float64)
        return x,h,w

    def test_shared_head_unary_cancels_for_every_expert(self):
        x,h,w=self.sample()
        r=source_f3_f4_router_geometry_audit(x,h,w)
        self.assertLess(r["common_x_unary_logit_spread_c2_max"],1e-12)
        self.assertLess(r["common_x_unary_logit_spread_c3_max"],1e-12)
        self.assertGreater(r["original_history_routing_score_gap_rms"],.01)
        self.assertGreater(r["expert_relative_logits_geometry_gap_rms"],.01)
        self.assertTrue(r["same_2d_router_head_across_all_experts"])
        self.assertFalse(r["geometry_comparison_same_initial_function_established"])

    def test_first_zero_history_gate_has_common_logits(self):
        x,h,w=self.sample()
        h=torch.zeros_like(h)
        for geometry in ("c2_invariant","c2c3_wave"):
            features=original_router_features(x,h,geometry)
            logits=features@w
            torch.testing.assert_close(logits,logits[:,:1].expand_as(logits),
                                       rtol=0,atol=1e-12)

    def test_hidden_space_displacement_matching_not_router_feature_matching(self):
        x,h,w=self.sample()
        r=source_f3_f4_router_geometry_audit(x,h,w)
        self.assertLess(
            r["source_x_displacement_rel_vs_random_max_abs_difference"],1e-10)
        self.assertLess(r["relative_C3_unary_feature_delta_max"],1e-10)
        self.assertGreater(r["random_unary_feature_delta_mean"],.01)
        self.assertLess(
            r["fraction_original_source_rows_feature_displacement_matched_5pct"],
            1.)
        self.assertTrue(r["source_activation_matched_is_not_router_feature_matched"])

    def test_history_sign_independence_is_lost_by_c3_directed_features(self):
        x,h,w=self.sample()
        c2a=original_router_features(x,h,"c2_invariant")
        c2b=original_router_features(x,-h,"c2_invariant")
        torch.testing.assert_close(c2a,c2b,rtol=0,atol=0)
        c3a=original_router_features(x,h,"c2c3_wave")
        c3b=original_router_features(x,-h,"c2c3_wave")
        d=x.shape[1]
        torch.testing.assert_close(c3a[...,:d],c3b[...,:d],
                                   rtol=0,atol=0)
        torch.testing.assert_close(c3a[...,d:],-c3b[...,d:],
                                   rtol=0,atol=1e-12)

    def test_invalid_shapes_and_weight_refuse(self):
        x,h,w=self.sample()
        with self.assertRaises(ValueError):
            original_router_features(x,h[:,:1],"c2_invariant")
        with self.assertRaises(ValueError):
            source_f3_f4_router_geometry_audit(x,h,w[:-1])
        with self.assertRaises(ValueError):
            source_f3_f4_router_geometry_audit(x,h,w[None].repeat(4,1))
        with self.assertRaises(ValueError):
            original_router_features(x,h,"unsupported")

class BoundedLiveCaptureTests(unittest.TestCase):
    def test_actual_routing_method_preserved_and_restored(self):
        from types import SimpleNamespace
        from scripts.audit_x369_f3_live_router_feature_metric import (
            bounded_router_state_capture)
        class MockMOE:
            route_geometry="c2c3_wave"
            def select_source_experts(self,x,h):
                return x.sum(),h.sum()
        moe=MockMOE()
        model=SimpleNamespace(
            cfg=SimpleNamespace(d_model=24,moe_experts=4),
            blocks=[SimpleNamespace(moe=moe)])
        x=torch.randn(2,24,dtype=torch.float64)
        h=torch.randn(2,4,24,dtype=torch.float64)
        before=moe.select_source_experts(x,h)
        with bounded_router_state_capture(
                model,seed=7,max_states_per_layer=4) as (saved,seen):
            for i in range(7):
                after=moe.select_source_experts(x+i,h)
                expected=(x+i).sum(),h.sum()
                self.assertEqual(after,expected)
            self.assertEqual(seen,[14])
            self.assertEqual(len(saved[0]),4)
            self.assertTrue(all(t[0].shape==(24,) and
                                t[1].shape==(4,24) for t in saved[0]))
        self.assertEqual(moe.select_source_experts(x,h),before)

    def test_capture_fail_closed_budget_and_bad_live_shape(self):
        from types import SimpleNamespace
        from scripts.audit_x369_f3_live_router_feature_metric import (
            bounded_router_state_capture)
        class MockMOE:
            route_geometry="c2c3_wave"
            def select_source_experts(self,x,h):
                return x
        moe=MockMOE()
        model=SimpleNamespace(
            cfg=SimpleNamespace(d_model=24,moe_experts=4),
            blocks=[SimpleNamespace(moe=moe)])
        with self.assertRaises(ValueError):
            with bounded_router_state_capture(model,max_states_per_layer=1):
                pass
        with bounded_router_state_capture(model,max_states_per_layer=4):
            with self.assertRaises(ValueError):
                moe.select_source_experts(torch.randn(2,24),torch.randn(2,3,24))
        self.assertEqual(moe.select_source_experts(
            torch.zeros(1,24),torch.zeros(1,3,24)).shape,(1,24))

if __name__=="__main__":
    unittest.main()
