"""Negative/positive contracts for the independent 200-step receipt join.

Fixtures are SYNTHETIC and never stand in for a real gfx803 HIP run.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.gfx803_moe_gpu_gate import file_sha256
from scripts.verify_lila_dense_gpu_smoke import verify


class VerifyDenseSmokeTests(unittest.TestCase):
    def fixture(self, root):
        folder=Path(root)
        settings={"target_steps":200,"save_every":25,
                  "teacher_sha256":"a"*64,"train_tokens_sha256":"b"*64}
        def trace(count):
            return [{"step":i,"train_loss_nats":2.0/(i+1),
                     "grad_norm_preclip":0.5,"target_tokens":126,
                     "row_indices":[0,1]} for i in range(1,count+1)]
        def checkpoints(steps):
            result=[]
            for step in steps:
                path=folder/f"step_{step}.pt"
                path.write_bytes(f"synthetic {step}".encode())
                result.append({"step":step,"path":str(path),
                               "sha256":file_sha256(path),
                               "reload_verified":True})
            return result
        first={"status":"partial-restart-required","native_gpu_only":True,
               "no_c2":True,"settings":settings,
               "gpu":{"architecture":"gfx803","hip":"7.16.26385"},
               "start_step":0,"end_step":100,"steps_this_process":100,
               "checkpoint_resume_verified":False,"trace":trace(100),
               "checkpoints_written":checkpoints([25,50,75,100]),
               "mean_training_tokens_per_second":10.,
               "peak_allocated_bytes":1024}
        second=copy.deepcopy(first)
        second.update({"status":"complete","start_step":100,"end_step":200,
                       "steps_this_process":100,
                       "checkpoint_resume_verified":True,
                       "resume_verified":True,"resumed_step":100,
                       "resumed_checkpoint_sha256":first["checkpoints_written"][-1]["sha256"],
                       "trace":trace(200),
                       "checkpoints_written":checkpoints([125,150,175,200])})
        paths=(folder/"first.json",folder/"second.json")
        for path,data in zip(paths,(first,second)):
            path.write_text(json.dumps(data))
        return paths,first,second

    def test_valid_source_join_requires_exact_split_and_eight_files(self):
        with TemporaryDirectory() as folder:
            paths,_,_=self.fixture(folder)
            answer=verify(*paths)
            self.assertTrue(answer["verified"])
            self.assertEqual(answer["steps"],200)
            self.assertEqual(answer["checkpoints_verified"],8)
            self.assertEqual(answer["process_boundary_resume_step"],100)

    def test_wrong_parent_or_missing_checkpoint_fails(self):
        with TemporaryDirectory() as folder:
            paths,first,second=self.fixture(folder)
            altered=copy.deepcopy(second)
            altered["resumed_checkpoint_sha256"]="f"*64
            paths[1].write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError,"resume"):
                verify(*paths)
            paths[1].write_text(json.dumps(second))
            Path(first["checkpoints_written"][0]["path"]).unlink()
            with self.assertRaisesRegex(ValueError,"checkpoint"):
                verify(*paths)

    def test_mismatched_source_or_incomplete_trace_fails(self):
        with TemporaryDirectory() as folder:
            paths,_,second=self.fixture(folder)
            changed=copy.deepcopy(second)
            changed["settings"]["train_tokens_sha256"]="c"*64
            paths[1].write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,"source|hyperparameters"):
                verify(*paths)
            changed=copy.deepcopy(second)
            changed["trace"].pop()
            paths[1].write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,"incomplete"):
                verify(*paths)


if __name__=="__main__":
    unittest.main()
