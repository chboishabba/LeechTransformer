"""CPU-only contracts for independent original dense LILA training gate.

A CPU/CPU parity fixture is not evidence of gfx803 correctness;
the CLI explicitly requires an actual HIP GPU.
"""
from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import torch

from models.model import LeechConfig, LeechGPT
from scripts.lila_dense_training_parity import (
    validate_tokens, dense_train_loss, compared_gradients,
    compared_parameters, dense_trajectory, main,
)


class DenseTrainingParityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(121)
        cfg=LeechConfig(vocab_size=37,d_model=48,n_heads=2,n_layers=2,
                        block_size=8,dropout=0.,lambda_geo=0.,moe_experts=0)
        self.model=LeechGPT(cfg).eval()
        self.rows=np.random.default_rng(17).integers(
            0,37,(2,6),dtype=np.int64)

    def test_true_dense_next_token_loss_has_gradients(self):
        ids=torch.tensor(self.rows,dtype=torch.long)
        loss,logits=dense_train_loss(self.model,ids)
        self.assertEqual(list(logits.shape),[2,6,37])
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in self.model.parameters()))

    def test_real_token_validation_rejects_wrong_shape_or_vocab(self):
        validate_tokens(self.rows,self.model,2)
        for rows in (self.rows[:,0:1],self.rows.astype(np.float32),
                     np.full_like(self.rows,37)):
            with self.assertRaises(ValueError):
                validate_tokens(rows,self.model,2)
        with self.assertRaises(ValueError):
            validate_tokens(self.rows,self.model,3)

    def test_all_parameter_gradient_comparison_fails_on_change(self):
        import copy
        cpu=copy.deepcopy(self.model)
        other=copy.deepcopy(self.model)
        ids=torch.tensor(self.rows)
        loss1,_=dense_train_loss(cpu,ids)
        loss2,_=dense_train_loss(other,ids)
        loss1.backward()
        loss2.backward()
        self.assertTrue(compared_gradients(cpu,other)["passed"])
        with torch.no_grad():
            first=next(other.parameters())
            first.add_(0.25)
        self.assertFalse(compared_parameters(cpu,other)["passed"])
        next(other.parameters()).grad.add_(1.)
        self.assertFalse(compared_gradients(cpu,other)["passed"])

    def test_two_step_cpu_control_exercises_entire_gate(self):
        # CPU/CPU reference tests logic ONLY; it cannot be used by CLI.
        outcome=dense_trajectory(self.model,self.rows,device="cpu",
                                  steps=2,repeats=2)
        self.assertTrue(outcome["admitted"])
        self.assertEqual(outcome["steps_completed"],2)
        self.assertEqual(len(outcome["trajectory"]),2)
        for receipt in outcome["trajectory"]:
            for key in ("loss","forward","repeats","gradients",
                        "parameters","post_update_logits"):
                self.assertTrue(receipt[key]["passed"],key)

    def test_cli_denies_cpu_only_without_touching_hardware(self):
        with mock.patch("sys.argv",[
            "lila_dense_training_parity",
            "--teacher","/missing-checkpoint",
            "--train-ids","/missing-rows",
            "--receipt","/tmp/lila-no-hip-probe.json"]):
            # Before loading missing paths, CLI may report missing inputs;
            # either way never returns a successful numerical admission.
            self.assertEqual(main(),2)


if __name__=="__main__":
    unittest.main()
