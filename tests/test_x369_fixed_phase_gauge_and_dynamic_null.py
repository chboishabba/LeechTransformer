"""X369 maximum cut: fixed phase is merely a weight-permutation gauge.

The new q-gated mixed null changes orientation PER INPUT, while retaining
signed C2 symmetry and exact 2q(x)q(h) feature-energy matching.
These tests certify source geometry, NOT empirical C3 semantics.
"""
import unittest
import torch

from models.history_orbit_moe import HistoryOrbitMoE
from models.x369_fair_c3_residual import fixed_mixed_c3_weight_gauge
from models.trialectic_wave_prior import (
    rotate3, trialectic_router_invariants, directed_trialectic_invariants,
    q_gated_mixed_c3_mask,q_gated_mixed_c3_x)


def feats(x,h):
    return torch.cat((
        trialectic_router_invariants(x),
        directed_trialectic_invariants(x,h)),dim=-1)


class X369GaugeAndGatedNull(unittest.TestCase):
    def sample(self):
        g=torch.Generator().manual_seed(369)
        x=torch.randn(8,4,24,generator=g,dtype=torch.float64)
        h=torch.randn(8,4,24,generator=g,dtype=torch.float64)
        base=torch.tensor([True,False,False,True,False,True,False,True])
        return x,h,base

    def test_fixed_coherent_vs_mixed_is_exact_weight_permutation(self):
        x,h,base=self.sample()
        triads=x.reshape(8,4,8,3)
        fixed=torch.where(
            base.reshape(1,1,8,1),
            triads.roll(-1,-1),triads.roll(1,-1)).reshape_as(x)
        coherent=rotate3(x)
        w=torch.randn(1,48,dtype=torch.float64,
                      generator=torch.Generator().manual_seed(12))
        wfixed=fixed_mixed_c3_weight_gauge(w,base)
        torch.testing.assert_close(
            (feats(coherent,h)*w).sum(-1),
            (feats(fixed,h)*wfixed).sum(-1),
            rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(w.norm(),wfixed.norm(),rtol=0,atol=0)
        self.assertFalse(torch.equal(w,wfixed))

    def test_gated_null_is_c2_and_c3_mask_invariant_but_source_dependent(self):
        x,h,base=self.sample()
        mask=q_gated_mixed_c3_mask(x,base)
        self.assertTrue(torch.equal(mask,q_gated_mixed_c3_mask(-x,base)))
        self.assertTrue(torch.equal(mask,q_gated_mixed_c3_mask(rotate3(x),base)))
        # This is the decisive condition absent from old fixed-mixed:
        # there is no one coordinate permutation for ALL these inputs.
        self.assertTrue(bool((mask[0]!=mask[1]).any()))
        xplus=q_gated_mixed_c3_x(x,base)
        torch.testing.assert_close(
            q_gated_mixed_c3_x(-x,base),-xplus,
            rtol=0,atol=0)
        self.assertFalse(torch.equal(xplus,rotate3(x)))
        self.assertFalse(torch.equal(xplus,rotate3(rotate3(x))))

    def test_gated_null_preserves_every_expert_directed_energy(self):
        x,h,base=self.sample()
        coher=feats(rotate3(x),h)-feats(x,h)
        mixed=feats(q_gated_mixed_c3_x(x,base),h)-feats(x,h)
        torch.testing.assert_close(
            coher.square().sum(-1),mixed.square().sum(-1),
            rtol=1e-12,atol=1e-12)
        self.assertGreater(float((coher-mixed).abs().max()),.01)

    def test_actual_nonzero_residual_keeps_source_signed_c2(self):
        torch.manual_seed(40)
        moe=HistoryOrbitMoE(d_model=24,num_experts=4,top_k=2,
            route_geometry="c2_c3_mixed_null_residual").double().eval()
        with torch.no_grad():
            moe.c3_residual_weight.copy_(
                torch.randn_like(moe.c3_residual_weight))
        x,h,_=self.sample()
        # Under simultaneous x,h sign reversal, mixed gate q is
        # unchanged and directed channels are bilinear.
        a,weights=moe.select_source_experts(x[:,0,:],h)
        b,reversed_weights=moe.select_source_experts(-x[:,0,:],-h)
        self.assertTrue(torch.equal(a,b))
        torch.testing.assert_close(weights,reversed_weights,
                                   rtol=1e-12,atol=1e-12)

    def test_malformed_masks_fail_closed(self):
        x,h,base=self.sample()
        with self.assertRaises(ValueError):
            q_gated_mixed_c3_x(x,base[:-1])
        with self.assertRaises(ValueError):
            q_gated_mixed_c3_x(x,base.to(torch.int64))
        with self.assertRaises(ValueError):
            q_gated_mixed_c3_x(torch.full_like(x,float("nan")),base)
        with self.assertRaises(ValueError):
            fixed_mixed_c3_weight_gauge(
                torch.zeros(1,48),base[:-1])


if __name__=="__main__":
    unittest.main()
