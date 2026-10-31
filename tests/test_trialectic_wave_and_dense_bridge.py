import unittest
import torch
from torch import nn
from models.trialectic_wave_prior import (
    wave3,unwave3,rotate3,trialectic_router_invariants,
    phase_rotation_residual,geometry_shape)
from models.dense_to_signed_experts import DenseToSignedExperts


class TrialecticWavePriorTests(unittest.TestCase):
    def test_384_24_3_rechart(self):
        self.assertEqual(geometry_shape(384,8,12)["leech24_blocks"],16)
        self.assertEqual(geometry_shape(384,8,12)["trialectic_triples_per_layer"],128)
        torch.manual_seed(4)
        x=torch.randn(3,384)
        self.assertTrue(torch.allclose(unwave3(wave3(x)),x,atol=1e-6))
        self.assertLess(float(phase_rotation_residual(x).abs().max()),2e-6)

    def test_c2_and_c3_invariant_router_features(self):
        x=torch.randn(3,384)
        f=trialectic_router_invariants
        self.assertTrue(torch.equal(f(x),f(-x)))
        self.assertTrue(torch.allclose(f(x),f(rotate3(x)),atol=2e-5,rtol=2e-5))

    def test_signed_wave_router_leaves_inverted_history_compatible(self):
        from models.history_orbit_moe import HistoryOrbitMoE
        torch.manual_seed(3)
        m=HistoryOrbitMoE(48,4,2,24,route_geometry="c2c3_wave")
        x=torch.randn(2,48);h=torch.randn(2,4,48)
        p=m.step(x,h); n=m.step(-x,-h)
        self.assertTrue(torch.equal(p[2],n[2]))
        self.assertTrue(torch.equal(p[4],n[4]))
        self.assertTrue(torch.allclose(p[0],-n[0],atol=2e-6))
        self.assertTrue(torch.allclose(p[1],-n[1],atol=2e-6))

    def test_dense_teacher_exact_parity_and_trainable_odd_student(self):
        torch.manual_seed(3)
        teacher=nn.Sequential(nn.Linear(48,96),nn.GELU(),nn.Linear(96,48))
        bridge=DenseToSignedExperts(teacher,48,4,2,24)
        x=torch.randn(2,4,48)
        obj=bridge(x,blend=0.)
        self.assertTrue(torch.allclose(obj["output"],obj["reference"],atol=2e-6))
        with torch.no_grad():
            obj_flip=bridge(-x,blend=0.)
            self.assertTrue(torch.allclose(obj["even"],obj_flip["even"],atol=2e-6))
            self.assertTrue(torch.allclose(obj["odd"],-obj_flip["odd"],atol=2e-6))
        bridge.train()
        bridge.zero_grad()
        bridge(x,blend=0.)["odd_distillation_loss"].backward()
        self.assertTrue(any(p.grad is not None for p in bridge.student.parameters()))
        self.assertTrue(all(not p.requires_grad for p in bridge.teacher.parameters()))

    def test_deployable_even_odd_students_and_exact_residual(self):
        from models.dense_to_signed_experts import canonical_even_input
        torch.manual_seed(17)
        teacher=nn.Sequential(nn.Linear(48,96),nn.GELU(),nn.Linear(96,48))
        bridge=DenseToSignedExperts(teacher,48,4,2,24,transition_rank=24).eval()
        x=torch.randn(2,3,48)
        self.assertTrue(torch.equal(canonical_even_input(x),
                                    canonical_even_input(-x)))
        positive=bridge.predict_only(x)
        negative=bridge.predict_only(-x)
        self.assertTrue(torch.allclose(positive["even_student"],
                                       negative["even_student"],atol=2e-6))
        self.assertTrue(torch.allclose(positive["odd_student"],
                                       -negative["odd_student"],atol=2e-6))
        record=bridge(x,blend=1.)
        self.assertTrue(torch.allclose(record["reference"],
            record["student_output"]+record["exact_residual"],atol=2e-6))
        # The deployed prediction uses neither teacher forward nor weights.
        bridge.teacher.forward=lambda x: (_ for _ in ()).throw(
            AssertionError("teacher consulted by deployment"))
        bridge.predict_only(x)

    def test_rank24_history_factorization_reduces_parameters(self):
        from models.history_orbit_moe import HistoryOrbitMoE
        full=HistoryOrbitMoE(384,4,2,192,route_geometry="raw")
        reduced=HistoryOrbitMoE(384,4,2,192,
                                route_geometry="c2c3_wave",transition_rank=24)
        count=lambda m:sum(p.numel() for p in m.transitions.parameters())
        self.assertLess(count(reduced),count(full))
        self.assertEqual(reduced.transition_rank,24)

    def test_independent_face_context_not_deduced_from_boundary(self):
        teacher=nn.Sequential(nn.Linear(48,48))
        bridge=DenseToSignedExperts(teacher,48,2,1,32)
        x=torch.randn(1,2,48)
        self.assertFalse(bridge(x)["face_supplied"])
        self.assertTrue(bridge(x,face_context=x)["face_supplied"])
        with self.assertRaises(ValueError):
            bridge(x,face_context=torch.zeros(1,2,24))


if __name__=="__main__":
    unittest.main()
