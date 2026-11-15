"""Finite C2 history stabilisation controls on source MoE.

These are semantic CPU tests, not RX580 CE or a global contraction proof.
"""
import unittest
import torch
from models.history_orbit_moe import HistoryOrbitMoE
from scripts.screen_c2_history_even_repair import (
    bounded_transition,interventions,cases)

class C2HistoryInterventionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026)
        self.moe=HistoryOrbitMoE(d_model=24,num_experts=4,
            top_k=2,expert_width=16,route_geometry="c2_invariant").eval()

    def test_radial_clip_is_odd_and_bounded(self):
        y=torch.randn(9,24)*200
        u=bounded_transition(y,cap=4.,scale=1.)
        v=bounded_transition(-y,cap=4.,scale=1.)
        self.assertTrue(torch.allclose(u,-v,atol=1e-5,rtol=1e-6))
        self.assertLessEqual(float(torch.linalg.vector_norm(u.double(),dim=-1).max()),4.00001)

    def test_actual_recurrence_has_hard_selected_history_bound(self):
        from contextlib import contextmanager
        @contextmanager
        def capped_moe(cap):
            handles=[layer.register_forward_hook(
                lambda module,inp,out:bounded_transition(out,cap,1.))
                for layer in self.moe.transitions]
            try:yield
            finally:
                for handle in handles:handle.remove()
        x=torch.randn(1,24)
        h=torch.zeros(1,4,24)
        with capped_moe(2.):
            for _ in range(25):
                _,h,_,_,_=self.moe.step(x,h)
                self.assertLessEqual(
                    float(torch.linalg.vector_norm(h.double(),dim=-1).max()),2.00001)

    def test_finite_c2_sign_transport_with_clip(self):
        x=torch.randn(2,24)
        h=torch.randn(2,4,24)
        handles=[layer.register_forward_hook(
            lambda module,inp,out:bounded_transition(out,2.,1.))
            for layer in self.moe.transitions]
        try:
            a,ha,ida,sa,wa=self.moe.step(x,h)
            b,hb,idb,sb,wb=self.moe.step(-x,-h)
            self.assertTrue(torch.equal(ida,idb))
            self.assertTrue(torch.allclose(wa,wb,atol=1e-6))
            self.assertTrue(torch.allclose(a,-b,atol=1e-6))
            self.assertTrue(torch.allclose(ha,-hb,atol=1e-6))
        finally:
            for layer in handles:layer.remove()

    def test_controls_include_identity_and_fail_closed(self):
        self.assertEqual(cases()[0]["name"],"baseline")
        self.assertEqual(cases()[0]["even_scales"],{})
        with self.assertRaises(ValueError):
            bounded_transition(torch.zeros(1,24),-1.,1.)
        # Validation of caps is at the context manager boundary.
        from models.model import LeechConfig,LeechGPT
        cfg=LeechConfig(vocab_size=32,d_model=48,n_heads=2,n_layers=12,
            block_size=8,dropout=0.,moe_experts=4,moe_top_k=2,
            moe_expert_width=24,moe_even_width=24,
            moe_guarded_ffn=True,moe_deployed_parity=True)
        model=LeechGPT(cfg).eval()
        with self.assertRaises(ValueError):
            with interventions(model,late_cap=-1):pass

if __name__=="__main__":unittest.main()
