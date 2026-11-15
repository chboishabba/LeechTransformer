"""CPU-only checks for real-module stage instrumentation.

These assertions do not create gfx803 hardware receipts or allow training.
"""
import unittest
from unittest import mock

import torch

from models.model import LeechConfig, LeechGPT
from scripts.diagnose_gfx803_leech_operator_boundary import (
    first_failed_stage, named_stages, recorded_forward, run_differential)


class Gfx803LeechOperatorBoundaryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.net=LeechGPT(LeechConfig(
            vocab_size=37,d_model=48,n_heads=2,n_layers=2,
            block_size=8,dropout=0.,lambda_geo=0.)).eval()

    def test_stage_order_reuses_original_forward_modules(self):
        stages=[name for name,_ in named_stages(self.net,1)]
        self.assertEqual(stages[:5],[
            "tok_emb","block0.ln1","block0.qkv",
            "block0.attn.out","block0.attn"])
        self.assertEqual(stages[-2:],["final_norm","head"])
        self.assertIn("block0.ffn.linear_in",stages)
        self.assertIn("block0.ffn.linear_out",stages)
        self.assertIn("block0.output",stages)
        self.assertNotIn("block1.qkv",stages)
        with self.assertRaises(ValueError):
            list(named_stages(self.net,0))
        with self.assertRaises(ValueError):
            list(named_stages(self.net,3))

    def test_hooks_preserve_cpu_output_and_remove_after_use(self):
        ids=torch.randint(0,37,(2,5))
        with torch.no_grad():
            expected=self.net(ids)[0].detach().clone()
            snapshots,actual=recorded_forward(self.net,ids,1)
        self.assertTrue(torch.equal(expected,actual))
        self.assertEqual(snapshots["block0.qkv"].shape,(2,5,144))
        self.assertEqual(snapshots["block0.ffn.linear_in"].shape,(2,5,192))
        self.assertEqual(snapshots["head"].shape,(2,5,37))
        self.assertFalse(self.net.blocks[0].attn.qkv._forward_hooks)

    def test_first_failure_does_not_skip_prior_stages(self):
        checks={"embedding":{"passed":True},
                "block0.qkv":{"passed":False},
                "head":{"passed":False}}
        self.assertEqual(first_failed_stage(checks),"block0.qkv")
        self.assertIsNone(first_failed_stage({"all":{"passed":True}}))
        self.assertEqual(first_failed_stage({"missing":{}}),"missing")

    def test_rocm_hardware_diagnostic_refuses_cpu_only(self):
        with mock.patch.object(torch.cuda,"is_available",return_value=False):
            with self.assertRaisesRegex(RuntimeError,"ROCm"):
                run_differential(self.net,torch.randint(0,37,(2,5)).numpy())


if __name__=="__main__":
    unittest.main()
