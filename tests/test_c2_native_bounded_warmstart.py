"""Source native C2 checkpoint repair semantics, no original GPU outcome claims."""
import dataclasses
import unittest
import torch
from models.model import LeechConfig,LeechGPT
from models.history_orbit_moe import HistoryOrbitMoE
from scripts.train_c2_bounded_quality_repair import (
    configure_original,trainable_parameters,recover_split)
from scripts.export_c2_teacher_free import strip_state,compare_bidi
import numpy as np

class NativeHistoryCapWarmstartContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73)
        self.cfg=LeechConfig(vocab_size=32,d_model=48,n_layers=12,n_heads=2,
            block_size=8,dropout=0.,moe_experts=4,moe_top_k=2,
            moe_expert_width=24,moe_even_width=24,moe_deployed_parity=True,
            moe_guarded_ffn=True,moe_route_geometry="c2_invariant")
        self.original=LeechGPT(self.cfg).eval()
        self.ids=torch.tensor([[1,2,3,4]],dtype=torch.long)

    def test_default_off_exact_source_identity(self):
        # Native config defaults remain original candidate semantics.
        equal=configure_original(self.original,cap=None,even0=1.,even1=1.)
        a=self.original(self.ids,student_only=True)[0]
        b=equal(self.ids,student_only=True)[0]
        self.assertTrue(torch.equal(a,b))
        self.assertIsNone(equal.blocks[10].moe.history_cap)

    def test_native_history_cap_c2_odd_and_selected_execution(self):
        moe=HistoryOrbitMoE(d_model=48,num_experts=4,top_k=2,
            expert_width=24,route_geometry="c2_invariant").eval()
        moe.history_cap=4.
        x=torch.randn(2,48)
        h=torch.zeros(2,4,48)
        for _ in range(12):
            y,h,ids,signs,w=moe.step(x,h)
            self.assertLessEqual(float(torch.linalg.vector_norm(
                h.double(),dim=-1).max()),4.0001)
        yp,hp,_,_,_=moe.step(x,h)
        yn,hn,_,_,_=moe.step(-x,-h)
        self.assertTrue(torch.allclose(yp,-yn,atol=1e-5))
        self.assertTrue(torch.allclose(hp,-hn,atol=1e-5))

    def test_repaired_guarded_and_pure_bidi_and_config(self):
        repaired=configure_original(self.original,cap=4.,
            even0=.75,even1=1.25)
        self.assertEqual(repaired.blocks[10].moe.history_cap,4.)
        self.assertEqual(repaired.blocks[11].moe.history_cap,4.)
        self.assertIsNone(repaired.blocks[9].moe.history_cap)
        pure,newcfg,removed=strip_state(repaired.state_dict(),repaired.cfg)
        self.assertEqual(newcfg.moe_late_history_cap,4.)
        self.assertEqual(newcfg.moe_even_layer0_scale,.75)
        self.assertTrue(all(b.ffn is None for b in pure.blocks))
        parity=compare_bidi(repaired,pure,self.ids,batch=1,device="cpu")
        self.assertLess(parity["max_logits_abs"],1e-5)

    def test_gradient_scope(self):
        repaired=configure_original(self.original,cap=64.,
            even0=.75,even1=1.)
        params=trainable_parameters(repaired,late=True)
        wanted={id(p) for p in params}
        self.assertTrue(wanted)
        for i,b in enumerate(repaired.blocks):
            for p in b.moe.transitions.parameters():
                self.assertEqual(p.requires_grad,i in (10,11))
            if i in (0,1):
                self.assertTrue(all(p.requires_grad for p in b.even_student.parameters()))
        self.assertEqual(wanted,{id(p) for p in repaired.parameters() if p.requires_grad})

    def test_original_split_rejected_if_overlap(self):
        rows=np.array([[1,2,3],[4,5,6],[7,8,9]],dtype=np.int64)
        with self.assertRaises(ValueError):
            recover_split({"kind":"native_gpu_cumulative_student_quality_curriculum",
                  "train_fit_indices":[0,1],"train_development_indices":[1,2]},rows)

if __name__=="__main__":unittest.main()
