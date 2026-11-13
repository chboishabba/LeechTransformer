"""Guarded frozen-teacher source identity and genuine student-only rollout.

Tiny constructed CPU weights are semantic unit tests, not native GPU quality
measurements and not trained LILA release receipts.
"""
import dataclasses
import unittest
import torch
from models.model import LeechConfig, LeechGPT
from scripts.train_guarded_moe_continuation import build_guarded_hybrid

class OriginalTeacherIdentityTests(unittest.TestCase):
    def test_hybrid_restores_teacher_ffn_even_if_student_ffn_changed(self):
        torch.manual_seed(71)
        dense_config=LeechConfig(vocab_size=32,d_model=48,n_layers=2,
                                 n_heads=2,block_size=8,dropout=0.)
        teacher=LeechGPT(dense_config).eval()
        student_config=dataclasses.replace(
            dense_config,moe_experts=4,moe_top_k=2,
            moe_expert_width=24,moe_even_width=24,
            moe_deployed_parity=True,moe_guarded_ffn=True,
            moe_route_geometry="c2_invariant")
        student=LeechGPT(student_config).eval()
        # Source pretrained attention/embedding/norm MUST match exactly.
        original=teacher.state_dict()
        state=student.state_dict()
        for key in set(state)&set(original):
            state[key]=original[key].detach().clone()
        student.load_state_dict(state)
        # Student FFN intentionally contaminated: its source is NOT canonical.
        with torch.no_grad():
            student.blocks[0].ffn[0].weight.add_(1.0)
        hybrid=build_guarded_hybrid(teacher,student).eval()
        for i in range(2):
            for key,reference in teacher.blocks[i].ffn.state_dict().items():
                self.assertTrue(torch.equal(reference,
                    hybrid.blocks[i].ffn.state_dict()[key]))
            self.assertEqual(float(hybrid.blocks[i].quality_gate),0.)
        sample=torch.tensor([[1,2,3,4],[5,6,7,8]],dtype=torch.long)
        with torch.no_grad():
            teacher_logits=teacher(sample)[0]
            zero_gate_logits=hybrid(sample)[0]
            student_only_logits=hybrid(sample,student_only=True)[0]
        self.assertTrue(torch.allclose(teacher_logits,zero_gate_logits,
                                       atol=1e-6,rtol=1e-6))
        self.assertEqual(student_only_logits.shape,teacher_logits.shape)
        self.assertTrue(torch.isfinite(student_only_logits).all())

if __name__=="__main__":
    unittest.main()
