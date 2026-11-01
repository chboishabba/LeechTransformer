"""Executed E8 root mathematics on Torch, not only dimension numerology."""
import unittest
import torch
from models.e8_root_prior import (
    e8_roots,reflect_e8,e8_router_invariants,e8_geometry_receipt)
from models.history_orbit_moe import HistoryOrbitMoE


class E8RootPriorTests(unittest.TestCase):
    def test_exact_240_roots_112_plus_128_and_opposites(self):
        roots=e8_roots(dtype=torch.float64)
        self.assertEqual(tuple(roots.shape),(240,8))
        self.assertTrue(torch.equal((roots**2).sum(-1),
                                    torch.full((240,),2.,dtype=torch.float64)))
        self.assertEqual(len(set(map(tuple,roots.tolist()))),240)
        self.assertEqual(int((roots.abs()==1.).sum(-1).eq(2).sum()),112)
        self.assertEqual(int((roots.abs()==.5).all(-1).sum()),128)
        universe=set(map(tuple,roots.tolist()))
        self.assertTrue(all(tuple(-v for v in r) in universe for r in universe))

    def test_384D_Leech24_block_alignment_without_lattice_identity(self):
        receipt=e8_geometry_receipt(384,8)
        self.assertEqual(receipt["e8_charts"],48)
        self.assertEqual(receipt["e8_charts_per_head"],6)
        self.assertEqual(receipt["e8_charts_per_24d_block"],3)
        self.assertTrue(receipt["Leech_requires_independent_gluing_proof"])

    def test_weyl_reflection_is_involutive_and_closed_on_roots(self):
        roots=e8_roots(dtype=torch.float64)
        universe=set(map(tuple,roots.tolist()))
        # Select spanning root types: two-sparse and half-integer.
        for root_index in (0,17,111,112,160,239):
            rotated=reflect_e8(roots,root_index)
            self.assertTrue(torch.allclose(rotated.square().sum(-1),
                                            roots.square().sum(-1),atol=1e-12))
            self.assertTrue(torch.allclose(reflect_e8(rotated,root_index),
                                            roots,atol=1e-12))
            self.assertTrue(all(tuple(float(round(z*2)/2) for z in row)
                                in universe for row in rotated.tolist()))

    def test_e8_router_features_jointly_reflection_and_sign_invariant(self):
        torch.manual_seed(42)
        x=torch.randn(2,48,dtype=torch.float64)
        h=torch.randn(2,48,dtype=torch.float64)
        baseline=e8_router_invariants(x,h)
        self.assertEqual(tuple(baseline.shape),tuple(x.shape))
        self.assertTrue(torch.allclose(baseline,
            e8_router_invariants(-x,-h),atol=1e-12,rtol=1e-12))
        for i in (0,111,112,239):
            f=e8_router_invariants(reflect_e8(x,i),reflect_e8(h,i))
            self.assertTrue(torch.allclose(baseline,f,atol=3e-12,rtol=3e-12))
        self.assertTrue(torch.isfinite(e8_router_invariants(
            1e10*x,1e10*h)).all())

    def test_e8_learned_router_vs_expert_output_boundary(self):
        torch.manual_seed(21)
        model=HistoryOrbitMoE(48,4,2,24,
            route_geometry="e8_root_invariant",transition_rank=24).eval()
        x=torch.randn(2,48)
        h=torch.randn(2,4,48)
        a=model.step(x,h)
        flipped=model.step(-x,-h)
        self.assertEqual(model.router.in_features,48)
        self.assertTrue(torch.equal(a[2],flipped[2]))
        self.assertTrue(torch.allclose(a[4],flipped[4],atol=1e-6))
        self.assertTrue(torch.allclose(a[0],-flipped[0],atol=2e-6))
        for index in (0,112):
            reflected=model.step(reflect_e8(x,index),
                reflect_e8(h,index))
            self.assertTrue(torch.equal(a[2],reflected[2]))
            self.assertTrue(torch.allclose(a[4],reflected[4],atol=2e-5))
        # The expert FFN is NOT required to be a Weyl intertwiner.


if __name__=="__main__":
    unittest.main()
