"""Parity is an *inductive bias on learning*, not a moral/opponent label."""
import unittest
import torch
from models.history_orbit_moe import HistoryOrbitMoE
from models.model import LeechConfig, LeechGPT


class SignedExpertGeometryTests(unittest.TestCase):
    def _expert(self, k=2):
        torch.manual_seed(42)
        return HistoryOrbitMoE(d_model=6,num_experts=4,top_k=k,
                               expert_width=12,route_geometry="c2_invariant").eval()

    def test_equivariance_all_top_k_and_nonzero_history(self):
        for k in (1,2,4):
            model=self._expert(k)
            x=torch.randn(4,6)
            history=torch.randn(4,4,6)
            output,updated,ids,signs,weights=model.step(x,history)
            inverted=model.step(-x,-history)
            self.assertTrue(torch.equal(ids,inverted[2]))
            self.assertTrue(torch.equal(weights,inverted[4]))
            self.assertTrue(torch.allclose(output,-inverted[0],atol=2e-6,rtol=2e-6))
            self.assertTrue(torch.allclose(updated,-inverted[1],atol=2e-6,rtol=2e-6))
            self.assertTrue(torch.equal(signs,-inverted[3]))

    def test_zero_orientation_score_uses_inversion_consistent_tiebreak(self):
        model=self._expert(2)
        with torch.no_grad():model.orientation.weight.zero_()
        x=torch.tensor([[0.,0.,-2.,0.,0.,0.]])
        h=torch.zeros(1,4,6)
        out=model.step(x,h)
        flipped=model.step(-x,-h)
        self.assertTrue(torch.equal(out[3],-flipped[3]))
        self.assertTrue(torch.allclose(out[0],-flipped[0],atol=2e-6))
        self.assertTrue(torch.allclose(out[1],-flipped[1],atol=2e-6))

    def test_zero_fixed_point_remains_fixed(self):
        model=self._expert()
        out=model.step(torch.zeros(1,6),torch.zeros(1,4,6))
        self.assertTrue(torch.equal(out[0],torch.zeros_like(out[0])))
        self.assertTrue(torch.equal(out[1],torch.zeros_like(out[1])))
        self.assertTrue(torch.equal(out[3],torch.zeros_like(out[3])))

    def test_full_sequence_inversion_with_retained_histories(self):
        model=self._expert()
        x=torch.randn(2,5,6)
        h=torch.randn(2,4,6)
        positive=model(x,h)
        negative=model(-x,-h)
        self.assertTrue(torch.equal(positive[2],negative[2]))
        self.assertTrue(torch.equal(positive[4],negative[4]))
        self.assertTrue(torch.allclose(positive[0],-negative[0],atol=3e-6,rtol=3e-6))
        self.assertTrue(torch.allclose(positive[1],-negative[1],atol=3e-6,rtol=3e-6))
        # Same semantic response annotations would NOT imply this check;
        # it explicitly applies a geometric action to inputs AND histories.

    def test_strict_default_config_preserved(self):
        self.assertEqual(LeechConfig(vocab_size=40).moe_route_geometry,"raw")
        kwargs=dict(vocab_size=40,d_model=48,n_layers=1,n_heads=2,
                    block_size=12,dropout=0.,lambda_geo=0.,
                    moe_experts=3,moe_top_k=2,moe_expert_width=16)
        raw=LeechGPT(LeechConfig(**kwargs))
        signed=LeechGPT(LeechConfig(**kwargs,moe_route_geometry="c2_invariant"))
        signed.load_state_dict(raw.state_dict(),strict=True)
        self.assertEqual(signed.blocks[0].moe.route_geometry,"c2_invariant")

    def test_default_raw_router_not_forced_to_be_invariant(self):
        model=HistoryOrbitMoE(2,2,1,4,route_geometry="raw")
        with torch.no_grad():
            model.router.weight.zero_()
            model.router.bias.zero_()
            model.router.weight[0,2]=1
            model.expert_bias.zero_()
        x=torch.zeros(1,2)
        h=torch.tensor([[[1.,0.],[-1.,0.]]])
        self.assertNotEqual(model.step(x,h)[2].item(),
                            model.step(-x,-h)[2].item())


if __name__=="__main__":
    unittest.main()
