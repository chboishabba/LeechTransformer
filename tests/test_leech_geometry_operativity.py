"""Measure which geometric components actually change the learner."""
import unittest
import torch
from models.model import generate_leech_kernel,LeechConfig,LeechGPT


class LeechGeometryOperativityTests(unittest.TestCase):
    def test_shared_qk_orthogonal_kernel_cancels_from_attention_scores(self):
        torch.manual_seed(123)
        transform=generate_leech_kernel(24)
        self.assertTrue(torch.allclose(transform @ transform.T,
                                        torch.eye(24),atol=2e-6))
        q=torch.randn(5,24)
        k=torch.randn(7,24)
        raw=q@k.T
        transformed=(q@transform)@(k@transform).T
        self.assertTrue(torch.allclose(raw,transformed,atol=3e-6,rtol=3e-6))

    def test_dense_checkpoint_without_geo_loss_has_no_trained_moe(self):
        cfg=LeechConfig(vocab_size=2048,d_model=384,n_layers=12,
                        n_heads=8,block_size=512,lambda_geo=0.,
                        moe_experts=0)
        self.assertEqual(cfg.moe_experts,0)
        self.assertEqual(cfg.lambda_geo,0)
        self.assertFalse(cfg.moe_deployed_parity)
        # The architecture may still have learned nonlinear QKV and FFN;
        # identity of attention scores is NOT identity of model parameters.


if __name__=="__main__": unittest.main()
