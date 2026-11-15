"""Original source full-replay candidate and low-cost per-step diagnostics.

These CPU tests prove an exact toy AdamW/scheduler/RNG continuation after a
source-bound checkpoint. Original native RX 580 optimizer/dataloader replay
requires a separate actual worktree/physical receipt.
"""
import random
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scripts.x369_long_grokking_receipts import (
    GrokkingStepLedger,is_full_mechanism_step,save_exact_replay,
    load_exact_replay,checked_identity)

IDENT={'model_sha256':'a'*64,'train_data_sha256':'b'*64,
       'tokenizer_sha256':'c'*64,'code_revision':'Leech-original-commit'}

def fixture_model():
    m=nn.Linear(3,2)
    o=torch.optim.AdamW(m.parameters(),lr=.03)
    s=torch.optim.lr_scheduler.StepLR(o,step_size=2)
    return m,o,s

def step(model,optimizer,scheduler):
    x=torch.tensor([[1.,2.,3.]])
    noise=random.random()+float(np.random.rand())+float(torch.rand(()))
    y=model(x)
    loss=(y-1.7*noise).square().mean()
    optimizer.zero_grad();loss.backward();optimizer.step();scheduler.step()
    return loss.item()

class OriginalGrokkingReplayContracts(unittest.TestCase):
    def test_full_adamw_scheduler_rng_and_cursor_resume(self):
        random.seed(4);np.random.seed(4);torch.manual_seed(4)
        m,o,s=fixture_model()
        for i in range(3):step(m,o,s)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'state.pt'
            receipt=save_exact_replay(path,step=3,model=m,optimizer=o,scheduler=s,
                                     data_cursor={'document_id':'story7','offset':45},
                                     source_identity=IDENT)
            expected_loss=step(m,o,s)
            expected=[p.detach().clone() for p in m.parameters()]
            expected_lr=o.param_groups[0]['lr']
            m2,o2,s2=fixture_model()
            stepnum,cursor=load_exact_replay(path,model=m2,optimizer=o2,scheduler=s2,
                                              expected_source_identity=IDENT,
                                              expected_sha256=receipt['sha256'])
            self.assertEqual(stepnum,3)
            self.assertEqual(cursor['offset'],45)
            got_loss=step(m2,o2,s2)
            self.assertEqual(expected_loss,got_loss)
            for a,b in zip(m2.parameters(),expected):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
            self.assertEqual(o2.param_groups[0]['lr'],expected_lr)
            with self.assertRaises(FileExistsError):
                save_exact_replay(path,step=4,model=m2,optimizer=o2,
                                  data_cursor=cursor,source_identity=IDENT)

    def test_source_identity_hash_and_cursor_fail_closed(self):
        with self.assertRaisesRegex(ValueError,'original'):
            checked_identity({'model_sha256':'bad'}, {'offset':1})
        with self.assertRaisesRegex(ValueError,'JSON'):
            checked_identity(IDENT,{'offset':float('nan')})
        m,o,s=fixture_model()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'state.pt'
            save_exact_replay(path,step=1,model=m,optimizer=o,scheduler=s,
                 data_cursor={'cursor':1},source_identity=IDENT)
            with self.assertRaisesRegex(ValueError,'identity'):
                load_exact_replay(path,model=m,optimizer=o,scheduler=s,
                    expected_source_identity=dict(IDENT,model_sha256='d'*64))
            with self.assertRaisesRegex(ValueError,'hash'):
                load_exact_replay(path,model=m,optimizer=o,scheduler=s,
                    expected_source_identity=IDENT,expected_sha256='e'*64)

    def test_each_step_log_and_geometric_expensive_gate(self):
        self.assertFalse(is_full_mechanism_step(127))
        self.assertTrue(is_full_mechanism_step(128))
        self.assertTrue(is_full_mechanism_step(256))
        self.assertFalse(is_full_mechanism_step(255))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'steps.jsonl'
            log=GrokkingStepLedger(path,IDENT)
            self.assertFalse(log.record(127,token_count=1000,train_ce=2.4,
                    learning_rate=.01,grad_norm=.8)['full_mechanism_probe_due'])
            self.assertTrue(log.record(128,token_count=1032,train_ce=2.3,
                    learning_rate=.01,grad_norm=.7)['full_mechanism_probe_due'])
            again=GrokkingStepLedger(path,IDENT)
            self.assertEqual(again.last_step,128)
            with self.assertRaisesRegex(ValueError,'increase'):
                again.record(128,token_count=1032,train_ce=2.3,
                    learning_rate=.01,grad_norm=.7)
            with self.assertRaisesRegex(ValueError,'nonfinite'):
                again.record(129,token_count=1064,train_ce=float('nan'),
                    learning_rate=.01,grad_norm=.7)
            self.assertEqual(again.last_step,128)

if __name__=='__main__':unittest.main()
