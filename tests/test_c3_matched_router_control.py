"""Geometric F3 null matches source C3 displacement, not just norm."""
import unittest
import torch
from models.trialectic_wave_prior import rotate3
from models.c3_matched_router_control import matched_displacement_random_x,matched_mixed_orientation_c3_x

class MatchedRandomControlTests(unittest.TestCase):
    def test_norm_and_displacement_same_as_relative_c3(self):
        x=torch.randn(8,24,dtype=torch.float64,
                      generator=torch.Generator().manual_seed(72))
        shifted=rotate3(x)
        random=matched_displacement_random_x(x,seed=13)
        self.assertLess(float((x.norm(dim=-1)-random.norm(dim=-1)).abs().max()),1e-12)
        self.assertLess(float(((shifted-x).norm(dim=-1)-
                               (random-x).norm(dim=-1)).abs().max()),1e-12)
        self.assertGreater(float((shifted-random).abs().max()),.1)
    def test_float32_and_zero_source(self):
        x=torch.randn(3,384,generator=torch.Generator().manual_seed(5))
        x[1]=0
        random=matched_displacement_random_x(x,seed=100)
        self.assertEqual(random.dtype,torch.float32)
        self.assertEqual(float(random[1].abs().max()),0.)
        self.assertLess(float((x.norm(dim=-1)-random.norm(dim=-1)).abs().max()),2e-5)
        self.assertLess(float(((rotate3(x)-x).norm(dim=-1)-
                               (random-x).norm(dim=-1)).abs().max()),2e-5)
    def test_seed_reproducibility(self):
        x=torch.randn(3,24)
        self.assertTrue(torch.equal(matched_displacement_random_x(x,seed=1),
                                    matched_displacement_random_x(x,seed=1)))
        self.assertFalse(torch.equal(matched_displacement_random_x(x,seed=1),
                                     matched_displacement_random_x(x,seed=2)))
    def test_invalid_shapes_and_nonfinite(self):
        with self.assertRaises(ValueError):
            matched_displacement_random_x(torch.zeros(3,22))
        with self.assertRaises(ValueError):
            matched_displacement_random_x(torch.tensor([[float("nan")]*24]))


class MixedOrientationTwoMetricTests(unittest.TestCase):
    def test_exact_activation_and_router_feature_displacements(self):
        from models.trialectic_wave_prior import (
            trialectic_router_invariants,directed_trialectic_invariants,
            directed_c3_displacement_energy_exact)
        rng=torch.Generator().manual_seed(79)
        x=torch.randn(12,384,generator=rng,dtype=torch.float64)
        h=torch.randn(12,4,384,generator=rng,dtype=torch.float64)
        plus=rotate3(x)
        mixed=matched_mixed_orientation_c3_x(x,seed=29)
        self.assertFalse(torch.equal(mixed,plus))
        torch.testing.assert_close(x.norm(dim=-1),mixed.norm(dim=-1),
                                   rtol=1e-13,atol=1e-13)
        torch.testing.assert_close((plus-x).norm(dim=-1),
                                   (mixed-x).norm(dim=-1),
                                   rtol=1e-13,atol=1e-13)
        def features(z):
            y=z[:,None,:].expand_as(h)
            return torch.cat((trialectic_router_invariants(y),
                              directed_trialectic_invariants(y,h)),dim=-1)
        f0=features(x)
        f1=features(plus)
        fm=features(mixed)
        torch.testing.assert_close(fm[...,:384],f0[...,:384],
                                   rtol=1e-12,atol=1e-12)
        # Stronger than an average: EACH expert and EACH original row
        # has EXACT source router-feature displacement norm.
        torch.testing.assert_close(
            (fm-f0).norm(dim=-1),(f1-f0).norm(dim=-1),
            rtol=5e-12,atol=5e-12)
        closed=directed_c3_displacement_energy_exact(x,h)
        torch.testing.assert_close(
            (f1-f0).square().sum(dim=-1),closed,
            rtol=1e-11,atol=1e-11)
        torch.testing.assert_close(
            (fm-f0).square().sum(dim=-1),closed,
            rtol=1e-11,atol=1e-11)
        self.assertGreater(float((fm-f1).abs().max()),.1)

    def test_nontrivial_mixed_order_three_group_permutation(self):
        x=torch.arange(24,dtype=torch.float64).reshape(1,24)
        out=matched_mixed_orientation_c3_x(x,seed=41)
        original=x.reshape(-1,3)
        transformed=out.reshape(-1,3)
        left=original.roll(-1,-1)
        right=original.roll(+1,-1)
        plus=(transformed==left).all(dim=-1)
        minus=(transformed==right).all(dim=-1)
        self.assertTrue(bool((plus|minus).all()))
        self.assertTrue(bool(plus.any() and minus.any()))
        # The selected blockwise 3-cycles close after three
        # repetitions; this is NOT a learned task group action.
        second=torch.where(plus[:,None],transformed.roll(-1,-1),
                           transformed.roll(+1,-1))
        third=torch.where(plus[:,None],second.roll(-1,-1),
                          second.roll(+1,-1))
        torch.testing.assert_close(third,original,rtol=0,atol=0)

    def test_exact_feature_match_survives_zero_history_and_float32(self):
        from models.trialectic_wave_prior import directed_trialectic_invariants
        x=torch.randn(8,24,generator=torch.Generator().manual_seed(34))
        h=torch.randn(8,4,24,generator=torch.Generator().manual_seed(35))
        for value in (h,torch.zeros_like(h)):
            plus=rotate3(x)
            mix=matched_mixed_orientation_c3_x(x,seed=3)
            torch.testing.assert_close(
                (directed_trialectic_invariants(
                    plus[:,None,:].expand_as(value),value)-
                 directed_trialectic_invariants(
                    x[:,None,:].expand_as(value),value)).norm(dim=-1),
                (directed_trialectic_invariants(
                    mix[:,None,:].expand_as(value),value)-
                 directed_trialectic_invariants(
                    x[:,None,:].expand_as(value),value)).norm(dim=-1),
                rtol=3e-6,atol=3e-6)

    def test_reproducible_and_rejects_unmixable_single_triad(self):
        x=torch.randn(4,24)
        self.assertTrue(torch.equal(
            matched_mixed_orientation_c3_x(x,seed=11),
            matched_mixed_orientation_c3_x(x,seed=11)))
        with self.assertRaises(ValueError):
            matched_mixed_orientation_c3_x(torch.randn(4,3))
        with self.assertRaises(ValueError):
            matched_mixed_orientation_c3_x(torch.empty(0,24))
        with self.assertRaises(ValueError):
            matched_mixed_orientation_c3_x(
                torch.full((4,24),float("nan")))

    def test_q_factorization_closed_form_zero_and_nan_refusal(self):
        from models.trialectic_wave_prior import (
            directed_c3_displacement_energy_exact)
        x=torch.zeros(3,24,dtype=torch.float64)
        h=torch.randn(3,4,24,generator=torch.Generator().manual_seed(71),
                      dtype=torch.float64)
        e=directed_c3_displacement_energy_exact(x,h)
        self.assertTrue(torch.equal(e,torch.zeros_like(e)))
        x[:,0:3]=torch.tensor([1.,0.,0.],dtype=torch.float64)
        e=directed_c3_displacement_energy_exact(x,h)
        # For (1,0,0), q(x)=1 so per-expert energy=2q(h) at first triad.
        a,b,c=h[:,:,0],h[:,:,1],h[:,:,2]
        expected=2*(a*a+b*b+c*c-a*b-b*c-c*a)
        torch.testing.assert_close(e,expected,rtol=1e-12,atol=1e-12)
        with self.assertRaises(ValueError):
            directed_c3_displacement_energy_exact(x,h[:,:,:-1])
        with self.assertRaises(ValueError):
            directed_c3_displacement_energy_exact(
                torch.full_like(x,float("nan")),h)

if __name__=="__main__":
    unittest.main()
