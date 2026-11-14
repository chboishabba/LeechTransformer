"""F6 three-arm source-clone capacity contract; not F6 predictive evidence."""
import unittest
import torch
from models.model import LeechConfig, LeechGPT
from scripts.screen_bounded_c2_vs_c2c3_router_fresh import (
    clone_with_geometry,router_params_only,validate_fresh)
class F6SourceThreeArmTests(unittest.TestCase):
    def test_24d_matched_checkpoint_tensor_count_and_source_state(self):
        torch.manual_seed(41)
        cfg=LeechConfig(vocab_size=256,d_model=48,n_layers=12,n_heads=2,
            block_size=24,dropout=0.,moe_experts=4,moe_top_k=2,
            moe_expert_width=24,moe_even_width=24,moe_guarded_ffn=True,
            moe_deployed_parity=True,moe_route_geometry="c2_invariant",
            moe_late_history_cap=64.)
        source=LeechGPT(cfg).eval()
        clones=[clone_with_geometry(source,g) for g in
            ("c2_invariant","c2c3_wave","punctured9_wave_matched")]
        counts=[sum(p.numel() for p in router_params_only(m)) for m in clones]
        self.assertEqual(counts,[counts[0]]*3)
        for model in clones:
            self.assertEqual(model.blocks[0].moe.router.in_features,2*cfg.d_model)
            self.assertEqual(model.blocks[10].moe.history_cap,64.)
            self.assertTrue(torch.equal(model.blocks[0].moe.router.weight,
                                        source.blocks[0].moe.router.weight))
        x=torch.randn(2,48)
        h=torch.randn(2,4,48)
        ids,weights=clones[2].blocks[0].moe.select_source_experts(x,h)
        self.assertEqual(tuple(ids.shape),(2,2))
        self.assertEqual(tuple(weights.shape),(2,2))
    def test_old_development_and_lockbox_excluded(self):
        a=torch.arange(24).numpy().reshape(1,24)
        b=torch.arange(100,124).numpy().reshape(1,24)
        prev=torch.arange(200,224).numpy().reshape(1,24)
        locked=torch.arange(300,324).numpy().reshape(1,24)
        excluded=__import__("numpy").concatenate((a,prev,locked))
        with self.assertRaises(ValueError):
            validate_fresh(prev,excluded,b)
        with self.assertRaises(ValueError):
            validate_fresh(locked,excluded,b)
if __name__=="__main__":unittest.main()
