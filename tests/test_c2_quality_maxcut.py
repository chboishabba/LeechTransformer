"""Source semantic tests for Q2 original C2 quality, Dust and stripping.

Small deterministic CPU fixtures test code semantics only; they are not native
RX 580 measurements and do not certify original 4.763 checkpoint quality.
"""
import dataclasses
import unittest
import torch
from models.model import LeechConfig,LeechGPT
from scripts.compare_c2_forward_only_optimizer import (
    target_readout,batch_loss,probe_source_future,dust_step,spsa_step,adam_step)
from scripts.export_c2_teacher_free import strip_state,compare_bidi

class C2QualityMaxCutTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        cfg=LeechConfig(vocab_size=32,d_model=48,n_layers=2,n_heads=2,
            block_size=8,dropout=0.,lambda_geo=0.,
            moe_experts=4,moe_top_k=2,moe_expert_width=24,
            moe_even_width=24,moe_deployed_parity=True,
            moe_guarded_ffn=True,moe_route_geometry="c2_invariant")
        self.model=LeechGPT(cfg).eval()
        self.tokens=torch.tensor([[1,2,3,4],[2,1,4,3]],dtype=torch.long)

    def test_source_bidi_removes_original_dense_ffns(self):
        student,cfg,state=strip_state(self.model.state_dict(),self.model.cfg)
        self.assertFalse(cfg.moe_guarded_ffn)
        self.assertFalse(any(".ffn." in k or ".quality_gate" in k for k in state))
        self.assertTrue(all(b.ffn is None for b in student.blocks))
        parity=compare_bidi(self.model,student,self.tokens,1,"cpu")
        self.assertLess(parity["max_logits_abs"],1e-5)

    def test_stripping_requires_complete_even_channel(self):
        bad=dataclasses.replace(self.model.cfg,moe_deployed_parity=False)
        with self.assertRaises(ValueError):
            strip_state(self.model.state_dict(),bad)

    def test_future_credit_identity_at_zero_discount(self):
        x=torch.tensor([[[1.,2.,3.],[2.,1.,4.]]])
        self.assertTrue(torch.equal(probe_source_future(x,0.),x))
        self.assertTrue(torch.equal(probe_source_future(x,1.),
                                    torch.tensor([[[6.,5.,3.],[7.,5.,4.]]])))

    def test_source_dust_updates_real_readout_without_gradients(self):
        model=self.model
        for p in model.parameters():p.requires_grad_(False)
        layer=target_readout(model,1)
        before=layer.weight.detach().clone()
        rec=dust_step(model,layer,self.tokens,lr=.001,sigma=.05,
            k=4,gamma=0.,antithetic=True,
            generator=torch.Generator().manual_seed(8))
        self.assertEqual(rec["forward_calls"],5)
        self.assertEqual(rec["backward_calls"],0)
        self.assertFalse(torch.equal(before,layer.weight))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_spsa_is_forward_only(self):
        for p in self.model.parameters():p.requires_grad_(False)
        layer=target_readout(self.model,1)
        record=spsa_step(self.model,layer,self.tokens,
            lr=1e-3,sigma=.05,k=4,
            generator=torch.Generator().manual_seed(7))
        self.assertEqual(record["forward_calls"],4)
        self.assertEqual(record["backward_calls"],0)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_adamw_is_explicit_control(self):
        for p in self.model.parameters():p.requires_grad_(False)
        layer=target_readout(self.model,1)
        for p in layer.parameters():p.requires_grad_(True)
        optimizer=torch.optim.AdamW(layer.parameters(),lr=1e-4)
        record=adam_step(self.model,layer,self.tokens,optimizer)
        self.assertEqual(record["backward_calls"],1)
        self.assertEqual(record["forward_calls"],1)

    def test_no_false_promotion_from_state_stripping(self):
        _,_,state=strip_state(self.model.state_dict(),self.model.cfg)
        self.assertLess(len(state),len(self.model.state_dict()))

if __name__=="__main__":unittest.main()
