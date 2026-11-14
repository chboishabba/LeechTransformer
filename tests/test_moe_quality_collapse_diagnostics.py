import unittest
import torch
from models.model import LeechGPT,LeechConfig
from scripts.diagnose_moe_quality_collapse import (
    substituted_ffns,token_cross_entropy,teacher_forced_layer_diagnostics,
    diagnose)


class QualityCollapseDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        kwargs=dict(vocab_size=32,d_model=48,n_layers=2,n_heads=2,
                    block_size=12,dropout=0.,lambda_geo=0.)
        self.dense=LeechGPT(LeechConfig(**kwargs)).eval()
        self.student=LeechGPT(LeechConfig(
            **kwargs,moe_experts=3,moe_top_k=2,moe_expert_width=24,
            moe_route_geometry="punctured9_wave",
            moe_transition_rank=24,moe_deployed_parity=True,
            moe_even_width=24)).eval()
        self.tokens=torch.randint(0,32,(2,5)).numpy()

    def test_empty_substitution_is_identical_and_restores_model(self):
        before=token_cross_entropy(self.dense,self.tokens,batch=2)
        with substituted_ffns(self.dense,self.student,[]):
            result=token_cross_entropy(self.dense,self.tokens,batch=2)
        self.assertEqual(before,result)
        try:
            with substituted_ffns(self.dense,self.student,[0]):
                self.assertIs(self.dense.blocks[0].moe,
                              self.student.blocks[0].moe)
                self.assertIsNone(self.dense.blocks[1].moe)
                raise RuntimeError("checking restoration")
        except RuntimeError:
            pass
        self.assertIsNone(self.dense.blocks[0].moe)
        self.assertIsNone(self.dense.blocks[0].even_student)
        self.assertEqual(before,token_cross_entropy(self.dense,self.tokens,batch=2))

    def test_every_layer_replacement_agrees_with_complete_student(self):
        with torch.no_grad():
            with substituted_ffns(self.dense,self.student,[0,1]):
                # Non-FFN model weights differ randomly in this test;
                # copy only shared structural weights from the teacher
                # to ensure the complete intervention equals student.
                pass
            teacher_state=self.dense.state_dict()
            student_state=self.student.state_dict()
            for key in teacher_state:
                if ".ffn." not in key and key in student_state and (
                        student_state[key].shape==teacher_state[key].shape):
                    student_state[key]=teacher_state[key]
            self.student.load_state_dict(student_state,strict=True)
            with substituted_ffns(self.dense,self.student,[0,1]):
                combined=self.dense(torch.as_tensor(self.tokens))[0]
            full=self.student(torch.as_tensor(self.tokens))[0]
            self.assertTrue(torch.allclose(combined,full,atol=1e-6,rtol=1e-6))

    def test_same_input_diagnostics_and_ce_curves(self):
        result=diagnose(self.dense,self.student,self.tokens,
                        batch=2,teacher_forced_rows=1,mode="both")
        self.assertEqual(len(result["isolated_layer_substitutions"]),2)
        self.assertEqual(len(result["prefix_layer_substitutions"]),2)
        self.assertEqual(len(result["same_teacher_input_layer_diagnostics"]),2)
        for entry in result["same_teacher_input_layer_diagnostics"]:
            self.assertEqual(len(entry["route_counts"]),3)
            self.assertGreaterEqual(entry["maximum_trace_history_rms"],0)
            self.assertGreaterEqual(entry["same_input_relative_mse"],0)
        self.assertIsNone(self.dense.blocks[0].moe)


if __name__=="__main__":
    unittest.main()
