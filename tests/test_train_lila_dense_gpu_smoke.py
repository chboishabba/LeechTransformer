"""Dense LILA GPU-native checkpoint/resume source contract.

Tests run a tiny real LeechGPT on CPU to validate exact restart and
optimizer/RNG restoration. CLI still requires HIP and refuses CPU fallback.
"""
from __future__ import annotations

import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from models.model import LeechConfig, LeechGPT
from scripts.train_lila_dense_gpu_smoke import (
    _capture_rng, _optimizer_state_compatible, _restore_rng,
    assert_gpu, deterministic_indices, same_tree, settings,
    to_cpu_tree, train, verify_saved_checkpoint,
)


def args_for(directory, stop=None):
    args=SimpleNamespace(
        output_dir=Path(directory), steps=4, stop_after=stop,
        save_every=2, batch=2, learning_rate=3e-4,
        seed=41, grad_clip=1.0)
    args.run_settings=settings(args, "a"*64, "b"*64, None)
    return args


class DenseGpuSmokeContractTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(212)
        self.model=LeechGPT(LeechConfig(
            vocab_size=37,d_model=48,n_heads=2,n_layers=1,
            block_size=8,dropout=0.10,lambda_geo=0.01,moe_experts=0))
        self.rows=np.random.default_rng(11).integers(
            0,37,size=(5,6),dtype=np.int64)

    def test_epoch_resampled_batches_deterministic_and_bounded(self):
        result=[deterministic_indices(5,2,i,41) for i in range(1,7)]
        self.assertEqual(result,
            [deterministic_indices(5,2,i,41) for i in range(1,7)])
        self.assertTrue(all(len(batch)==2 and all(0<=x<5 for x in batch)
                            for batch in result))
        self.assertNotEqual(result[:3],result[3:])
        with self.assertRaises(ValueError):
            deterministic_indices(1,2,1,41)

    def test_save_reload_then_exact_continuation_matches_uninterrupted(self):
        with TemporaryDirectory() as temp:
            full=copy.deepcopy(self.model)
            split=copy.deepcopy(self.model)
            torch.manual_seed(901)
            args_full=args_for(Path(temp)/"full")
            full_optimizer=torch.optim.AdamW(full.parameters(),lr=3e-4)
            whole=train(full,self.rows,args_full,optimizer=full_optimizer,
                        device="cpu")
            self.assertTrue(whole["admitted_for_scope"])
            self.assertEqual(whole["end_step"],4)
            self.assertEqual([x["step"] for x in whole["checkpoints_written"]],[2,4])
            torch.manual_seed(901)
            args_part=args_for(Path(temp)/"partial",stop=2)
            optimizer=torch.optim.AdamW(split.parameters(),lr=3e-4)
            partial=train(split,self.rows,args_part,optimizer=optimizer,
                          device="cpu")
            self.assertEqual(partial["end_step"],2)
            path=Path(temp)/"partial"/"dense_step_000002.pt"
            prior=torch.load(path,map_location="cpu",weights_only=True)
            verify_saved_checkpoint(path,prior)
            self.assertEqual(prior["step"],2)
            continued=copy.deepcopy(self.model)
            continued.load_state_dict(prior["model_state_dict"],strict=True)
            resume_optimizer=torch.optim.AdamW(continued.parameters(),lr=3e-4)
            _optimizer_state_compatible(
                resume_optimizer, prior["optimizer_state_dict"])
            _restore_rng(prior["rng_state"])
            args_resume=args_for(Path(temp)/"resumed")
            resumed=train(continued,self.rows,args_resume,
                          optimizer=resume_optimizer,previous=prior,
                          device="cpu")
            self.assertEqual(resumed["start_step"],2)
            self.assertEqual(resumed["end_step"],4)
            self.assertEqual([x["step"] for x in resumed["checkpoints_written"]],[4])
            self.assertTrue(resumed["resume_verified"])
            self.assertTrue(same_tree(
                to_cpu_tree(full.state_dict()),
                to_cpu_tree(continued.state_dict())))
            self.assertTrue(same_tree(
                to_cpu_tree(full_optimizer.state_dict()),
                to_cpu_tree(resume_optimizer.state_dict())))
            self.assertEqual(
                [x["train_loss_nats"] for x in whole["trace"]],
                [x["train_loss_nats"] for x in resumed["trace"]])
            self.assertEqual(
                [x["train_total_objective"] for x in whole["trace"]],
                [x["train_total_objective"] for x in resumed["trace"]])
            self.assertTrue(any(
                abs(x["train_total_objective"]-x["train_loss_nats"])>1e-5
                for x in whole["trace"]))
            self.assertTrue(all(
                item["reload_verified"] for item in
                whole["checkpoints_written"]+partial["checkpoints_written"]+
                resumed["checkpoints_written"]))

    def test_resume_source_contract_cannot_change_silently(self):
        a=args_for("unused")
        b=args_for("unused")
        self.assertEqual(a.run_settings,b.run_settings)
        b.batch=1
        altered=settings(b,"a"*64,"b"*64,None)
        self.assertNotEqual(altered,a.run_settings)
        self.assertFalse(same_tree(torch.tensor([1.]),torch.tensor([2.])))
        self.assertFalse(same_tree([1,2],[1,3]))

    def test_cpu_checkpoint_rng_bidi_without_cuda_initialization(self):
        # A host CPU-only PyTorch install must not access CUDA RNG
        # merely to test local source-free checkpoint determinism.
        from unittest.mock import patch
        with patch.object(torch.cuda,"is_available",return_value=False), \
             patch.object(torch.cuda,"get_rng_state_all",
                          side_effect=AssertionError("GPU touched")):
            state=_capture_rng()
        self.assertEqual(state["torch_gpu_all"],[])
        before=state["torch_cpu"].clone()
        torch.manual_seed(123)
        with patch.object(torch.cuda,"is_available",return_value=False), \
             patch.object(torch.cuda,"set_rng_state_all",
                          side_effect=AssertionError("GPU touched")):
            _restore_rng(state)
        self.assertTrue(torch.equal(torch.get_rng_state(),before))

    def test_saved_gpu_rng_never_silently_dropped_on_cpu_only_host(self):
        from unittest.mock import patch
        state=_capture_rng() if not torch.cuda.is_available() else None
        if state is None:
            state={"torch_cpu":torch.get_rng_state(),
                   "torch_gpu_all":[],"numpy":np.random.get_state(),
                   "python":__import__("random").getstate()}
        state["torch_gpu_all"]=[torch.tensor([1],dtype=torch.uint8)]
        with patch.object(torch.cuda,"is_available",return_value=False):
            with self.assertRaisesRegex(RuntimeError,
                                         "cannot restore saved GPU RNG"):
                _restore_rng(state)

    def test_cpu_rng_payload_cannot_be_promoted_into_gpu_resume(self):
        from unittest.mock import patch
        with patch.object(torch.cuda,"is_available",return_value=False):
            cpu_state=_capture_rng()
        self.assertEqual(cpu_state["torch_gpu_all"],[])
        with patch.object(torch.cuda,"is_available",return_value=True):
            with self.assertRaisesRegex(RuntimeError,
                                         "CPU-only RNG checkpoint cannot be restored"):
                _restore_rng(cpu_state)

    def test_gpu_admission_refuses_cpu_environment(self):
        from unittest.mock import patch
        with patch.object(torch.cuda,"is_available",return_value=False):
            with self.assertRaisesRegex(RuntimeError,"ROCm/HIP"):
                assert_gpu()


if __name__=="__main__":
    unittest.main()
