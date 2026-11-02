"""Regression for quality-preserving teacher->MoE progressive continuation."""
import unittest
import numpy as np
import torch
from models.model import LeechConfig,LeechGPT
from scripts.train_guarded_moe_continuation import (
    build_guarded_hybrid,train_continuation,calibrate_gates)


class GuardedMoETrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        base=dict(vocab_size=37,d_model=48,n_layers=2,n_heads=2,
                  block_size=12,dropout=0.,lambda_geo=0.)
        self.teacher=LeechGPT(LeechConfig(**base)).eval()
        self.student=LeechGPT(LeechConfig(
            **base,moe_experts=3,moe_top_k=2,moe_expert_width=24,
            moe_route_geometry="e8_root_invariant",
            moe_transition_rank=24,moe_deployed_parity=True,
            moe_even_width=24)).eval()
        shared=self.teacher.state_dict()
        state=self.student.state_dict()
        for k,v in shared.items():
            if k in state and ".ffn." not in k:
                state[k]=v
        self.student.load_state_dict(state,strict=True)

    def test_zero_gate_matches_dense_and_one_matches_student(self):
        guarded=build_guarded_hybrid(self.teacher,self.student).eval()
        ids=torch.randint(0,37,(2,6))
        with torch.no_grad():
            logits_original=self.teacher(ids)[0]
            logits_unguarded=self.student(ids)[0]
            logits_zero=guarded(ids)[0]
        self.assertTrue(torch.equal(logits_original,logits_zero))
        with torch.no_grad():
            for block in guarded.blocks:block.quality_gate.fill_(1.)
            logits_one=guarded(ids)[0]
        self.assertTrue(torch.allclose(logits_one,logits_unguarded,
                                       atol=1e-6,rtol=1e-6))
        self.assertTrue(all(not p.requires_grad for block in guarded.blocks
                            for p in block.ffn.parameters()))
        self.assertTrue(any(p.requires_grad for block in guarded.blocks
                            for p in block.moe.parameters()))

    def test_local_student_loss_gets_gradient_at_zero_gate(self):
        guarded=build_guarded_hybrid(self.teacher,self.student)
        guarded.train()
        tokens=torch.randint(0,37,(2,4))
        guarded.zero_grad()
        guarded(tokens)
        local=sum(b.last_local_distillation_loss for b in guarded.blocks)
        local.backward()
        self.assertTrue(any(p.grad is not None for b in guarded.blocks
                            for p in b.moe.parameters()))
        self.assertTrue(all(p.grad is None for b in guarded.blocks
                            for p in b.ffn.parameters()))

    def test_student_rollout_reaches_every_layer_while_all_gates_are_zero(self):
        guarded=build_guarded_hybrid(self.teacher,self.student).train()
        for module in guarded.modules():
            if isinstance(module,torch.nn.Dropout):
                module.eval()
        ids=torch.randint(0,37,(2,6))
        self.assertEqual([float(b.quality_gate) for b in guarded.blocks],[0.,0.])
        guarded.zero_grad(set_to_none=True)
        logits=guarded(ids,student_only=True)[0]
        with torch.no_grad():
            self.student.eval()
            reference=self.student(ids)[0]
        self.assertTrue(torch.allclose(logits.detach(),reference,
                                       atol=1e-6,rtol=1e-6))
        loss=torch.nn.functional.cross_entropy(
            logits[:,:-1].reshape(-1,37),ids[:,1:].reshape(-1))
        loss.backward()
        for block in guarded.blocks:
            self.assertTrue(any(p.grad is not None and
                bool(torch.isfinite(p.grad).all()) and
                bool((p.grad!=0).any())
                for p in block.even_student.parameters()))
            self.assertTrue(all(p.grad is None for p in block.ffn.parameters()))
        self.assertEqual([float(b.quality_gate) for b in guarded.blocks],[0.,0.])

    def test_student_only_mode_rejects_unguarded_models(self):
        ids=torch.randint(0,37,(1,5))
        for model in (self.teacher,self.student):
            with self.assertRaisesRegex(ValueError,"quality-guarded"):
                model(ids,student_only=True)

    def test_calibration_does_not_admit_bad_student_for_zero_budget(self):
        guarded=build_guarded_hybrid(self.teacher,self.student).eval()
        tokens=np.random.default_rng(3).integers(0,37,(4,5),dtype=np.int64)
        from scripts.diagnose_moe_quality_collapse import token_cross_entropy
        baseline=token_cross_entropy(self.teacher,tokens)["ce_nats"]
        result=calibrate_gates(guarded,tokens,teacher_ce=baseline,
                               budget=0.,batch=2,device="cpu")
        self.assertLessEqual(result["actual_calibration_ce"],baseline+1e-6)
        self.assertTrue(all(0.<=float(b.quality_gate)<=1.
                            for b in guarded.blocks))

    def test_stale_full_gates_are_revoked_before_new_revision_admission(self):
        guarded=build_guarded_hybrid(self.teacher,self.student).eval()
        with torch.no_grad():
            for block in guarded.blocks:block.quality_gate.fill_(1.)
        tokens=np.random.default_rng(123).integers(0,37,(5,6),dtype=np.int64)
        from scripts.diagnose_moe_quality_collapse import token_cross_entropy
        teacher_ce=token_cross_entropy(self.teacher,tokens)["ce_nats"]
        result=calibrate_gates(guarded,tokens,teacher_ce=teacher_ce,
                               budget=0.,batch=2,device="cpu",
                               candidates=(0.,.05,.25,1.))
        self.assertEqual(result["previous_revision_gates_revoked"],[1.,1.])
        self.assertLessEqual(result["zero_gate_calibration_ce"],teacher_ce+1e-6)
        self.assertLessEqual(result["actual_calibration_ce"],teacher_ce+1e-6)

    def test_direct_caller_rejects_exact_train_heldout_window_overlap(self):
        rows=np.random.default_rng(77).integers(0,37,(10,6),dtype=np.int64)
        with self.assertRaisesRegex(ValueError,"overlap"):
            train_continuation(self.teacher,self.student,rows,rows[:2],
                               epochs=1,batch=2)

    def test_tiny_end_to_end_guarded_fit_and_provenance(self):
        rows=np.random.default_rng(7).integers(0,37,(10,5),dtype=np.int64)
        hold=np.random.default_rng(8).integers(0,37,(4,5),dtype=np.int64)
        model,receipt=train_continuation(self.teacher,self.student,
            rows,hold,epochs=1,batch=2,seed=7,
            calibration_budget_nats=.02)
        self.assertEqual(len(receipt["all_quality_gates"]),2)
        self.assertTrue(receipt["retained_dense_teacher_ffns"])
        self.assertLess(abs(receipt["zero_gate_heldout_ce"]-
                            receipt["teacher_heldout_ce"]),1e-5)
        self.assertIn("full_student_replacement_pass",receipt)
        self.assertIn("student_only_heldout_ce",receipt)
        self.assertIn("student_only_calibration_ce",receipt["epochs"][0])
        self.assertEqual(receipt["student_loss_weights"]["global_ce"],1.)
        self.assertTrue(np.isfinite(receipt["student_only_heldout_ce"]))

        self.assertTrue(all(0<=g<=1 for g in receipt["all_quality_gates"]))


if __name__=="__main__":
    unittest.main()
