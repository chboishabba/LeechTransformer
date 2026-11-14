"""Matched original bounded C2 vs C2xC3 experimental admission guards."""
import unittest
import numpy as np
import torch
from models.model import LeechConfig,LeechGPT
from scripts.screen_bounded_c2_vs_c2c3_router_fresh import (
    validate_fresh,clone_with_geometry,router_params_only,train_arm)

class BoundedC2C3MatchedTests(unittest.TestCase):
    def test_reject_exact_and_sliding_window_overlap(self):
        original=np.array([list(range(24))],dtype=np.int64)
        heldout=np.array([list(range(100,124))],dtype=np.int64)
        fresh=np.array([list(range(200,224))],dtype=np.int64)
        self.assertGreater(validate_fresh(fresh,original,heldout),0)
        with self.assertRaises(ValueError):
            validate_fresh(original,original,heldout)
        near=np.array([list(range(8,32))],dtype=np.int64)
        with self.assertRaises(ValueError):
            validate_fresh(near,original,heldout)

    def test_matched_router_parameters_and_exact_source_state(self):
        torch.manual_seed(41)
        cfg=LeechConfig(vocab_size=256,d_model=48,n_layers=12,n_heads=2,
          block_size=24,dropout=0.,moe_experts=4,moe_top_k=2,
          moe_expert_width=24,moe_even_width=24,
          moe_guarded_ffn=True,moe_deployed_parity=True,
          moe_route_geometry="c2_invariant",moe_late_history_cap=64.,
          moe_even_layer0_scale=.75,moe_even_layer1_scale=1.)
        source=LeechGPT(cfg).eval()
        left=clone_with_geometry(source,"c2_invariant")
        right=clone_with_geometry(source,"c2c3_wave")
        pa=router_params_only(left)
        pb=router_params_only(right)
        self.assertEqual(sum(p.numel() for p in pa),sum(p.numel() for p in pb))
        self.assertEqual(len(pa),len(pb))
        for a,b in zip(left.blocks,right.blocks):
            self.assertTrue(torch.equal(a.moe.router.weight,b.moe.router.weight))
            self.assertEqual(a.moe.history_cap,b.moe.history_cap)
        self.assertEqual(right.cfg.moe_route_geometry,"c2c3_wave")
        self.assertEqual(left.cfg.moe_route_geometry,"c2_invariant")

    def test_invalid_fresh_token_width_fail_closed(self):
        old=np.arange(24).reshape(1,24)
        other=np.arange(100,124).reshape(1,24)
        new=np.arange(200,225).reshape(1,25)
        with self.assertRaises(ValueError):
            validate_fresh(new,old,other)

if __name__=="__main__":unittest.main()
