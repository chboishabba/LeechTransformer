"""CPU-only regressions of the fail-closed gfx803 numerical admission.

These tests do NOT assert ROCm hardware correctness. Only an actual
gfx803/Polaris host running the named Nix stack and its own checkpoint
can produce an admitted GPU execution receipt.
"""
import subprocess
from pathlib import Path
from unittest import TestCase, mock

import numpy as np
import torch

from scripts.gfx803_moe_gpu_gate import comparison, run_gate


class Gfx803MoEGateUnitTests(TestCase):
    def test_tensor_parity_and_threshold(self):
        x=torch.tensor([[1.,2.],[-3.,4.]])
        self.assertTrue(comparison(x,x.clone())["passed"])
        near=x+1e-5
        self.assertTrue(comparison(x,near)["passed"])
        far=x+torch.tensor([[0.,0.],[0.,1.]])
        result=comparison(x,far)
        self.assertFalse(result["passed"])
        self.assertGreater(result["fail_count"],0)
        self.assertEqual(result["shape"],[2,2])

    def test_nan_and_shape_fail_closed(self):
        ref=torch.tensor([1.,2.])
        self.assertFalse(comparison(ref,torch.tensor([1.,float("nan")]))["passed"])
        self.assertFalse(comparison(ref,torch.tensor([[1.,2.]]))["passed"])

    def test_hip_invisibility_cannot_fall_back_to_cpu(self):
        with mock.patch.object(torch.cuda,"is_available",return_value=False):
            with self.assertRaisesRegex(RuntimeError,"ROCm/HIP"):
                run_gate(None,None,None)

    def test_host_launcher_is_valid_bash(self):
        root=Path(__file__).resolve().parents[1]
        shell=root/"scripts"/"run_gfx803_moe_nix.sh"
        result=subprocess.run(["bash","-n",str(shell)],
                              text=True,capture_output=True,check=False)
        self.assertEqual(result.returncode,0,msg=result.stderr)

    def test_launcher_refuses_missing_provenance(self):
        root=Path(__file__).resolve().parents[1]
        shell=root/"scripts"/"run_gfx803_moe_nix.sh"
        result=subprocess.run(["bash",str(shell),"--receipt","/tmp/no-gate.json"],
                              cwd=root,text=True,capture_output=True,check=False)
        self.assertNotEqual(result.returncode,0)
        self.assertIn("missing --teacher",result.stderr)


if __name__=="__main__":
    import unittest
    unittest.main()
