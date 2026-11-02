"""D2 counterfactuals must preserve frozen source, avoid data leakage and
report teacher-input direction separately from same-sample scale oracle."""
import unittest

import numpy as np
import torch

from models.model import LeechConfig, LeechGPT
from scripts.ablate_early_layer_quality import (
    ZeroOddMoE, ResetHistoryMoE, selected_variant,
    route_orientation_ablation, same_input_metrics, ablate)


class EarlyLayerQualityAblationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(51)
        config=dict(vocab_size=37,d_model=48,n_layers=2,n_heads=2,
                    block_size=12,dropout=0.,lambda_geo=0.)
        self.teacher=LeechGPT(LeechConfig(**config)).eval()
        self.student=LeechGPT(LeechConfig(
            **config,moe_experts=3,moe_top_k=2,
            moe_expert_width=24,moe_transition_rank=24,
            moe_deployed_parity=True,moe_even_width=24,
            moe_route_geometry="c2_invariant")).eval()
        state=self.student.state_dict()
        for key,value in self.teacher.state_dict().items():
            if key in state and ".ffn." not in key:
                state[key]=value.clone()
        self.student.load_state_dict(state,strict=True)
        self.tokens=np.random.default_rng(8).integers(
            0,37,size=(4,5),dtype=np.int64)

    def test_full_d2_ablation_restores_dense_and_student_state(self):
        teacher_before={k:v.detach().clone()
                        for k,v in self.teacher.state_dict().items()}
        student_before={k:v.detach().clone()
                        for k,v in self.student.state_dict().items()}
        result=ablate(self.teacher,self.student,self.tokens,
                      layer=1,batch=2,teacher_forced_rows=2)
        self.assertEqual(set(result["variants"]),
            {"full","even_only","odd_only","reset_history",
             "balanced_router","zero_orientation","flip_orientation"})
        self.assertTrue(result["shared_backbone_identical"])
        self.assertEqual(result["source_split"],"train_only")
        self.assertEqual(result["variants"]["full"]["target_tokens"],
                         result["baseline_dense"]["target_tokens"])
        self.assertIsNone(self.teacher.blocks[1].moe)
        self.assertIsNone(self.teacher.blocks[1].even_student)
        for key,value in teacher_before.items():
            self.assertTrue(torch.equal(value,self.teacher.state_dict()[key]),
                            msg="teacher modified: "+key)
        for key,value in student_before.items():
            self.assertTrue(torch.equal(value,self.student.state_dict()[key]),
                            msg="student modified: "+key)
        for case in result["variants"].values():
            self.assertLessEqual(case["same_probe_oracle_scaled_rmse"],
                                 case["rmse"]+1e-10)
            self.assertTrue(np.isfinite(case["ce_nats"]))

    def test_early_interventions_do_not_leak_on_failure(self):
        moe=self.student.blocks[0].moe
        original=moe.orientation.weight.detach().clone()
        with self.assertRaisesRegex(RuntimeError,"deliberate"):
            with selected_variant(self.teacher,self.student,0,
                                  "flip_orientation"):
                raise RuntimeError("deliberate")
        self.assertTrue(torch.equal(original,moe.orientation.weight))
        self.assertIsNone(self.teacher.blocks[0].moe)
        self.assertIsNone(self.teacher.blocks[0].even_student)
        with self.assertRaises(RuntimeError):
            with selected_variant(self.teacher,self.student,0,
                                  "balanced_router"):
                raise RuntimeError("deliberate")
        self.assertNotIn("forward",moe.router.__dict__)
        self.assertIsNone(self.teacher.blocks[0].moe)

    def test_stateless_and_even_only_dimensions(self):
        moe=self.student.blocks[0].moe
        x=torch.randn(2,4,48)
        both=moe(x)
        reset=ResetHistoryMoE(moe)(x)
        zero=ZeroOddMoE(moe)(x)
        self.assertEqual(reset[0].shape,both[0].shape)
        self.assertEqual(reset[1].shape,both[1].shape)
        self.assertTrue(torch.equal(reset[1],torch.zeros_like(reset[1])))
        self.assertTrue(torch.equal(zero[0],torch.zeros_like(x)))
        self.assertEqual(zero[2].shape,both[2].shape)
        with route_orientation_ablation(moe,"balanced_router"):
            changed=moe(x)
        self.assertEqual(changed[2].shape,both[2].shape)

    def test_same_probe_fit_does_not_pretend_to_generalise(self):
        a=torch.tensor([[1.,0.],[0.,1.]])
        b=torch.tensor([[0.,2.],[2.,0.]])
        out=same_input_metrics(a,b)
        self.assertAlmostEqual(out["uncentered_cosine"],0.)
        self.assertAlmostEqual(out["same_probe_least_squares_scalar"],0.)
        self.assertIn("identical data",out["oracle_warning"])
        with self.assertRaises(ValueError):
            same_input_metrics(a,b[:1])
        with self.assertRaises(ValueError):
            ablate(self.teacher,self.student,self.tokens[:1],layer=1)


if __name__=="__main__":
    unittest.main()
