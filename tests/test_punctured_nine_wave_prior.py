"""Regression for DASHI's actual punctured T² nine-sheet / 384D prior."""
import unittest
import torch
from models.punctured_nine_wave_prior import (
    SITES,FREQUENCY_PAIRS,real_fourier9_basis,puncture8_to_wave8,
    wave8_to_puncture8,wave8_to_nine_zero_mean,translate_wave8,
    invert_sites_wave8,pair_wave_router_features,punctured_wave_geometry,
    punctured_blocks_to_wave_blocks,wave_blocks_to_punctured_blocks,
    translate_punctured_blocks,nine_to_wave_and_centre,wave_and_centre_to_nine)
from models.history_orbit_moe import HistoryOrbitMoE


class PuncturedNineLearnerTests(unittest.TestCase):
    def test_prime_shape_and_four_antipodal_pairs(self):
        shape=punctured_wave_geometry(384,8)
        self.assertEqual(shape["blocks24"],16)
        self.assertEqual(shape["head_blocks24"],2)
        self.assertEqual(shape["punctured_points_per_lane"],8)
        self.assertEqual(len(SITES),9)
        self.assertEqual(len(FREQUENCY_PAIRS),4)
        self.assertEqual(set(SITES[1:]),{
            (-1,0),(1,0),(0,-1),(0,1),
            (-1,-1),(1,1),(-1,1),(1,-1)})
        for point in SITES[1:]:
            self.assertIn(tuple(-z for z in point),SITES[1:])

    def test_spatial_puncture_fourier_bridge_is_invertible(self):
        torch.manual_seed(3)
        x=torch.randn(3,16,3,8,dtype=torch.float64)
        y=puncture8_to_wave8(x)
        self.assertTrue(torch.allclose(wave8_to_puncture8(y),x,atol=1e-12))
        basis=real_fourier9_basis()
        self.assertTrue(torch.allclose(basis.T@basis,torch.eye(8,dtype=basis.dtype),atol=1e-12))
        self.assertLess(float(basis.sum(0).abs().max()),1e-12)
        full=wave8_to_nine_zero_mean(y)
        self.assertLess(float(full.sum(-1).abs().max()),1e-12)
        self.assertTrue(torch.allclose(
            wave_blocks_to_punctured_blocks(
                punctured_blocks_to_wave_blocks(x.reshape(3,384))),
            x.reshape(3,384),atol=1e-12))
        # 8 point amplitudes != an isometric 8-wave coordinate system.
        self.assertFalse(torch.allclose(x.square().sum(),y.square().sum()))

    def test_independent_centre_side_information_roundtrip(self):
        full=torch.randn(3,16,3,9,dtype=torch.float64)
        waves,centre=nine_to_wave_and_centre(full)
        self.assertTrue(torch.allclose(
            wave_and_centre_to_nine(waves,centre),full,atol=1e-12))
        self.assertFalse(torch.allclose(
            wave_and_centre_to_nine(waves,torch.zeros_like(centre)),full))
        self.assertEqual(tuple(waves.shape),(3,16,3,8))

    def test_f3_squared_phase_translation_law_and_conjugation(self):
        waves=torch.randn(5,8,dtype=torch.float64)
        a,b=(1,2),(2,1)
        moved=translate_wave8(translate_wave8(waves,a),b)
        combined=translate_wave8(waves,((a[0]+b[0])%3,(a[1]+b[1])%3))
        self.assertTrue(torch.allclose(moved,combined,atol=1e-12))
        self.assertTrue(torch.allclose(translate_wave8(waves,(3,3)),waves,atol=1e-12))
        self.assertTrue(torch.equal(invert_sites_wave8(invert_sites_wave8(waves)),waves))
        self.assertFalse(torch.equal(invert_sites_wave8(waves),-waves))
        sheet=wave8_to_nine_zero_mean(waves)
        shift=(1,2)
        shifted=wave8_to_nine_zero_mean(translate_wave8(waves,shift))
        for index,(x,y) in enumerate(SITES):
            source=(((x-shift[0]+1)%3)-1,((y-shift[1]+1)%3)-1)
            self.assertTrue(torch.allclose(shifted[:,index],
                                            sheet[:,SITES.index(source)],atol=1e-12))

    def test_ordered_dot_wedge_not_semantic_antipode(self):
        x=torch.zeros(1,24,dtype=torch.float64)
        h=torch.zeros_like(x)
        x[...,0]=1
        h[...,1]=1
        lhs=pair_wave_router_features(x,h)
        rhs=pair_wave_router_features(h,x)
        self.assertEqual(lhs.shape,(1,48))
        self.assertEqual(float(lhs[0,25]),1.)
        self.assertEqual(float(rhs[0,25]),-1.)
        self.assertTrue(torch.equal(lhs,pair_wave_router_features(-x,-h)))
        self.assertFalse(torch.equal(lhs,rhs))

    def test_router_joint_inversion_and_translations(self):
        torch.manual_seed(4)
        model=HistoryOrbitMoE(48,4,2,24,route_geometry="punctured9_wave").eval()
        x=torch.randn(2,48)
        h=torch.randn(2,4,48)
        result=model.step(x,h)
        flipped=model.step(-x,-h)
        self.assertTrue(torch.equal(result[2],flipped[2]))
        self.assertTrue(torch.allclose(result[4],flipped[4],atol=2e-6))
        self.assertTrue(torch.allclose(result[0],-flipped[0],atol=3e-6))
        self.assertTrue(torch.allclose(result[1],-flipped[1],atol=3e-6))
        shifted=model.step(translate_punctured_blocks(x,(1,1)),
                           translate_punctured_blocks(h,(1,1)))
        self.assertTrue(torch.equal(result[2],shifted[2]))
        self.assertTrue(torch.allclose(result[4],shifted[4],atol=1e-5,rtol=1e-5))
        # Output and history are not claimed translation-equivariant:
        # only the learned *router* has the group invariance.


if __name__=="__main__":
    unittest.main()
