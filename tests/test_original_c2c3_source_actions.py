"""Finite source C2, C3, S3 controls and actual router semantics.

No trained checkpoint is asserted C3-equivariant by these constructed tests.
"""
import unittest
import torch
from models.history_orbit_moe import HistoryOrbitMoE
from models.trialectic_wave_prior import (
    rotate3,phase_rotation_residual,trialectic_router_invariants,
    directed_trialectic_invariants)
from scripts.audit_original_c2c3_router_actions import (
    reflect3,group_witness,feature_contract,inspect_router)

class C2C3GeometryContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(117)
        self.x=torch.randn(5,24)
        self.h=torch.randn(5,4,24)

    def test_finite_c3_s3_and_central_sign(self):
        result=group_witness(self.x)
        self.assertTrue(all(result.values()))
        self.assertLess(float(phase_rotation_residual(self.x).abs().max()),1e-5)

    def test_feature_c2_c3_and_reflection_negative_control(self):
        report=feature_contract(self.x,self.h[:,0],1e-5)
        self.assertTrue(report["antipodal"]["within_tolerance"])
        self.assertTrue(report["cyclic"]["within_tolerance"])
        # S3 reflection exchanges two directed interaction channels, so
        # directed features are NOT generally reflection-invariant.
        self.assertFalse(report["reflect"]["within_tolerance"])

    def test_actual_c2c3_router_invariance(self):
        moe=HistoryOrbitMoE(d_model=24,num_experts=4,top_k=2,
                            expert_width=16,route_geometry="c2c3_wave").eval()
        r=inspect_router(moe,self.x,self.h,1e-5)
        self.assertTrue(r["antipodal"]["expert_ids_identical"])
        self.assertTrue(r["antipodal"]["weights_within_tolerance"])
        self.assertTrue(r["cyclic"]["expert_ids_identical"])
        self.assertTrue(r["cyclic"]["weights_within_tolerance"])
        # Reflection must be *measured*, not assumed symmetric.

    def test_c2_only_does_not_force_c3_router(self):
        moe=HistoryOrbitMoE(d_model=24,num_experts=4,top_k=2,
                            expert_width=16,route_geometry="c2_invariant").eval()
        r=inspect_router(moe,self.x,self.h,1e-5)
        self.assertTrue(r["antipodal"]["weights_within_tolerance"])
        # Even if an accidental random router happens to preserve this
        # sample's chosen IDs, its logits/weights should generally change.
        self.assertGreater(r["cyclic"]["max_topk_weight_abs_delta"],1e-7)

    def test_phase_relation_not_a_semantic_proof(self):
        # Unary feature C3 invariance does not mean every neuron output is
        # S3-equivariant: no trained expert/observer action constructed.
        self.assertEqual(trialectic_router_invariants(self.x).shape,self.x.shape)
        self.assertEqual(directed_trialectic_invariants(
            self.x,self.h[:,0]).shape,self.x.shape)

if __name__=="__main__":
    unittest.main()
