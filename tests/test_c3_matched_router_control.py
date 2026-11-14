"""Geometric F3 null matches source C3 displacement, not just norm."""
import unittest
import torch
from models.trialectic_wave_prior import rotate3
from models.c3_matched_router_control import matched_displacement_random_x

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

if __name__=="__main__":
    unittest.main()
