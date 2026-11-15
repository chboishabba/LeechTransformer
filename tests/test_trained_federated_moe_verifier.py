"""End-to-end verification harness smoke with an explicitly saved model."""
import json
import tempfile
import unittest
from pathlib import Path
import torch
from models.model import LeechGPT,LeechConfig
from scripts.verify_trained_federated_moe import checkpoint_parts,infer_config,evaluate


class TrainedVerifierTests(unittest.TestCase):
    def test_save_load_actual_weights_and_replay(self):
        cfg=LeechConfig(vocab_size=31,d_model=48,n_layers=1,n_heads=2,
                        block_size=12,dropout=0.,lambda_geo=0.,
                        moe_experts=3,moe_top_k=2,moe_expert_width=32)
        model=LeechGPT(cfg).eval()
        with tempfile.TemporaryDirectory() as d:
            ckpt=Path(d)/"weights.pt"
            torch.save({"model_state_dict":model.state_dict(),
                        "config":{"n_heads":2,"moe_top_k":2}},ckpt)
            state,saved=checkpoint_parts(ckpt)
            restored_cfg=infer_config(state,saved,None)
            restored=LeechGPT(restored_cfg).eval()
            restored.load_state_dict(state,strict=True)
            torch.manual_seed(5)
            receipt=evaluate(restored,torch.randint(0,31,(1,3)),rtt_ms=5)
            self.assertTrue(receipt["cache_parity_pass"])
            self.assertTrue(receipt["moe_enabled"])
            self.assertEqual(receipt["wire"]["adaptive"]["sha256"],
                             receipt["wire"]["raw"]["sha256"])
            self.assertLessEqual(receipt["wire"]["adaptive"]["wire_bytes"],
                                 receipt["wire"]["raw"]["wire_bytes"])
            self.assertEqual(len(receipt["geometry_diagnostics"]),1)

    def test_missing_uninferable_hyperparameters_rejected(self):
        cfg=LeechConfig(vocab_size=31,d_model=48,n_layers=1,n_heads=2,
                        block_size=12,dropout=0.,moe_experts=3)
        state=LeechGPT(cfg).state_dict()
        with self.assertRaisesRegex(ValueError,"top-k"):
            infer_config(state,{},None)

if __name__=="__main__":
    unittest.main()
