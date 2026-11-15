"""CPU contracts for R4 C2-preserving residual and capacity-matched C3 null.

These tests do not claim real-data CE, learned C3 action or RX580 acceptance.
"""
import unittest
import torch

from models.history_orbit_moe import HistoryOrbitMoE
from models.model import LeechConfig, LeechGPT
from models.trialectic_wave_prior import (
    directed_trialectic_invariants, rotate3, trialectic_router_invariants,
)
from models.x369_fair_c3_residual import (
    assert_identical_initial_student_rollout,
    clone_c2_with_zero_c3_residual,
)


class X369FairC3ResidualTests(unittest.TestCase):
    def source(self):
        torch.manual_seed(41)
        cfg = LeechConfig(
            vocab_size=128, d_model=48, n_layers=2, n_heads=2,
            block_size=8, dropout=0.0, moe_experts=4, moe_top_k=2,
            moe_expert_width=24, moe_even_width=24,
            moe_guarded_ffn=True, moe_deployed_parity=True,
            moe_route_geometry="c2_invariant")
        return LeechGPT(cfg).eval()

    def test_full_student_rollout_is_bitwise_identical_at_step_zero(self):
        source = self.source()
        coherent = clone_c2_with_zero_c3_residual(source, "c2_c3_residual")
        null = clone_c2_with_zero_c3_residual(
            source, "c2_c3_mixed_null_residual")
        c2 = clone_c2_with_zero_c3_residual(source, "c2_c2_residual")
        inputs = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long)
        receipt = assert_identical_initial_student_rollout(
            source, coherent, null, inputs, c2_control=c2)
        self.assertTrue(receipt["source_c2_equals_coherent_logits_bitwise"])
        self.assertTrue(receipt["source_c2_equals_mixed_null_logits_bitwise"])
        self.assertTrue(receipt["source_c2_equals_trainable_c2_control_bitwise"])
        self.assertEqual(receipt["residual_trainable_parameters_per_arm"], 192)
        self.assertFalse(receipt["f5_opened"])
        for i in range(2):
            self.assertTrue(torch.equal(
                source.blocks[i].moe.router.weight,
                coherent.blocks[i].moe.router.weight))
            self.assertEqual(
                coherent.blocks[i].moe.c3_residual_weight.count_nonzero().item(),
                0)

    def test_c3_control_has_exact_matched_feature_displacement(self):
        torch.manual_seed(42)
        module = HistoryOrbitMoE(
            d_model=48, num_experts=4, top_k=2,
            route_geometry="c2_c3_mixed_null_residual").double()
        x = torch.randn(3, 48, dtype=torch.float64)
        h = torch.randn(3, 4, 48, dtype=torch.float64)
        original = x[:, None, :].expand(-1, 4, -1)
        triads = original.reshape(3, 4, 16, 3)
        mask = module._c3_mixed_null_mask.view(1, 1, 16, 1)
        mixed = torch.where(
            mask, triads.roll(-1, -1), triads.roll(1, -1)
        ).reshape_as(original)
        coherent = rotate3(original)
        def features(z):
            return torch.cat((
                trialectic_router_invariants(z),
                directed_trialectic_invariants(z, h)), -1)
        original_features = features(original)
        cohesive_delta = (features(coherent) - original_features).square().sum(-1)
        mixed_delta = (features(mixed) - original_features).square().sum(-1)
        torch.testing.assert_close(
            cohesive_delta, mixed_delta, rtol=1e-12, atol=1e-10)
        self.assertTrue(bool(module._c3_mixed_null_mask.any()))
        self.assertTrue(bool((~module._c3_mixed_null_mask).any()))
        self.assertNotEqual(module._c3_mixed_null_mask.ndim, 0)

    def test_zero_residual_branch_has_nonzero_learnable_gradient(self):
        torch.manual_seed(43)
        module = HistoryOrbitMoE(
            d_model=48, num_experts=4, top_k=2,
            route_geometry="c2_c3_residual")
        x = torch.randn(5, 48)
        h = torch.randn(5, 4, 48)
        ids, weights = module.select_source_experts(x, h)
        self.assertEqual(tuple(ids.shape), (5, 2))
        (weights.square().sum()).backward()
        g = module.c3_residual_weight.grad
        self.assertIsNotNone(g)
        self.assertTrue(torch.isfinite(g).all().item())
        self.assertGreater(g.abs().sum().item(), 0.0)

    def test_fail_closed_on_wrong_source_and_missing_zero_mask(self):
        source = self.source()
        for bad in ("c2c3_wave", "raw", "c2_invariant"):
            with self.assertRaises(ValueError):
                clone_c2_with_zero_c3_residual(source, bad)
        with self.assertRaises(ValueError):
            HistoryOrbitMoE(
                d_model=3, num_experts=4,
                route_geometry="c2_c3_mixed_null_residual")
        coherent = clone_c2_with_zero_c3_residual(source, "c2_c3_residual")
        null = clone_c2_with_zero_c3_residual(
            source, "c2_c3_mixed_null_residual")
        with self.assertRaises(ValueError):
            assert_identical_initial_student_rollout(
                source, coherent, null, torch.tensor([[-1]]))
        with torch.no_grad():
            coherent.blocks[0].moe.c3_residual_weight.fill_(0.25)
        with self.assertRaises(AssertionError):
            assert_identical_initial_student_rollout(
                source, coherent, null,
                torch.tensor([[1, 2, 3, 4]], dtype=torch.long))


if __name__ == "__main__":
    unittest.main()
