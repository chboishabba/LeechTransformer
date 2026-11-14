"""Teacher-free checkpoint extraction and final-quality gate regressions."""
import unittest

import numpy as np
import torch

from models.model import LeechConfig, LeechGPT
from scripts.promote_guarded_moe import (
    extract_teacher_free, evaluate_teacher_free)
from scripts.train_guarded_moe_continuation import build_guarded_hybrid


class TeacherFreePromotionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(421)
        base=dict(vocab_size=41,d_model=48,n_layers=2,n_heads=2,
                  block_size=12,dropout=0.,lambda_geo=0.)
        self.teacher=LeechGPT(LeechConfig(**base)).eval()
        self.student=LeechGPT(LeechConfig(
            **base,moe_experts=3,moe_top_k=2,moe_expert_width=24,
            moe_transition_rank=24,moe_route_geometry="e8_root_invariant",
            moe_deployed_parity=True,moe_even_width=24)).eval()
        state=self.student.state_dict()
        for k,v in self.teacher.state_dict().items():
            if k in state and ".ffn." not in k:
                state[k]=v
        self.student.load_state_dict(state,strict=True)
        self.hybrid=build_guarded_hybrid(self.teacher,self.student).eval()

    def test_zero_and_partial_gates_cannot_strip_teacher(self):
        with self.assertRaisesRegex(ValueError,"gates"):
            extract_teacher_free(self.hybrid)
        with torch.no_grad():
            self.hybrid.blocks[0].quality_gate.fill_(1.)
            self.hybrid.blocks[1].quality_gate.fill_(.5)
        with self.assertRaisesRegex(ValueError,"gates"):
            extract_teacher_free(self.hybrid)

    def test_full_gates_remove_all_dense_tensors_and_preserve_logits(self):
        with torch.no_grad():
            for block in self.hybrid.blocks:
                block.quality_gate.fill_(1.)
        independent=extract_teacher_free(self.hybrid)
        self.assertFalse(independent.cfg.moe_guarded_ffn)
        self.assertFalse(any(".ffn." in key or key.endswith(".quality_gate")
                             for key in independent.state_dict()))
        self.assertLess(sum(p.numel() for p in independent.parameters()),
                        sum(p.numel() for p in self.hybrid.parameters()))
        ids=torch.randint(0,41,(2,7))
        with torch.no_grad():
            a=self.hybrid(ids)[0]
            b=independent(ids)[0]
        self.assertTrue(torch.allclose(a,b,atol=1e-4,rtol=1e-4))
        receipt=evaluate_teacher_free(
            self.teacher,self.hybrid,independent,
            np.random.default_rng(7).integers(0,41,(3,7),dtype=np.int64),
            ce_budget_nats=100.,max_logit_delta=1e-3)
        self.assertTrue(receipt["accepted"])
        self.assertTrue(receipt["all_dense_ffn_tensors_removed"])
        self.assertGreater(receipt["removed_parameter_count"],0)
        self.assertLessEqual(
            receipt["max_guarded_to_standalone_logit_abs_delta"],1e-3)

    def test_invalid_gate_is_rejected_even_if_clamped_to_one(self):
        with torch.no_grad():
            for block in self.hybrid.blocks:
                block.quality_gate.fill_(1.)
            self.hybrid.blocks[0].quality_gate.fill_(2.)
        with self.assertRaisesRegex(ValueError,"gates"):
            extract_teacher_free(self.hybrid)


if __name__ == "__main__":
    unittest.main()
