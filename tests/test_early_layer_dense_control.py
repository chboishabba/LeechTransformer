"""D2 capacity-matched nongeometric control; training-only, not promotion."""
import unittest
import numpy as np
import torch

from models.model import LeechConfig, LeechGPT
from scripts.train_early_layer_dense_control import (
    dense_width_for_budget, make_control, train_control)


class EarlyLayerDenseControlTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        config=dict(vocab_size=37,d_model=48,n_layers=2,n_heads=2,
                    block_size=12,dropout=0.,lambda_geo=0.)
        self.teacher=LeechGPT(LeechConfig(**config)).eval()
        self.student=LeechGPT(LeechConfig(**config,moe_experts=3,
            moe_top_k=2,moe_expert_width=24,moe_transition_rank=24,
            moe_deployed_parity=True,moe_even_width=24,
            moe_route_geometry="c2_invariant")).eval()
        state=self.student.state_dict()
        for name,value in self.teacher.state_dict().items():
            if name in state and ".ffn." not in name:
                state[name]=value.detach().clone()
        self.student.load_state_dict(state)
        rows=np.random.default_rng(11).integers(
            0,37,size=(10,5),dtype=np.int64)
        self.fit,self.development=rows[:8],rows[8:]

    def test_matched_budget_and_only_one_ffn_trainable(self):
        control,capacity=make_control(self.teacher,self.student,1)
        self.assertGreaterEqual(capacity["unmatched_parameter_slack"],0)
        self.assertLess(capacity["unmatched_parameter_slack"],97)
        self.assertTrue(all(p.requires_grad for p in
                            control.blocks[1].ffn.parameters()))
        self.assertTrue(all(not p.requires_grad for p in
                            control.blocks[0].ffn.parameters()))
        self.assertTrue(all(not p.requires_grad for p in
                            control.blocks[1].attn.parameters()))
        with self.assertRaises(ValueError):
            dense_width_for_budget(48,48)

    def test_one_epoch_is_diagnostic_not_teacher_free(self):
        before={k:v.clone() for k,v in self.teacher.state_dict().items()}
        trained,receipt=train_control(self.teacher,self.student,
            self.fit,self.development,layer=1,batch=2,epochs=1,seed=9)
        self.assertEqual(len(receipt["epochs"]),1)
        self.assertTrue(np.isfinite(
            receipt["final_control_development_ce_nats"]))
        self.assertFalse(receipt["full_teacher_free_promotion"])
        self.assertEqual(receipt["fit_rows"],8)
        self.assertEqual(receipt["development_rows"],2)
        for key,value in before.items():
            self.assertTrue(torch.equal(value,self.teacher.state_dict()[key]),
                            msg=key)
        for key,value in before.items():
            if not key.startswith("blocks.1.ffn."):
                self.assertTrue(torch.equal(value,trained.state_dict()[key]),
                                msg="nonfocus changed: "+key)

    def test_overlap_rejected(self):
        with self.assertRaisesRegex(ValueError,"overlap"):
            train_control(self.teacher,self.student,self.fit,
                          self.fit[:2],layer=1,epochs=1)


if __name__=="__main__":
    unittest.main()
