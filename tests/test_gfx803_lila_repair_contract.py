"""CPU-only contract checks for gfx803 LILA repair paths, no hardware admission."""
import os
from pathlib import Path
import subprocess
import unittest

from scripts.gfx803_linear_repair import repair_receipt
from scripts.gfx803_repair_runtime import mapped_paths, verify_loaded_sgemm_shim
from scripts.gfx803_wgm1_qkv_repair import validate_join


class RepairContractTests(unittest.TestCase):
    def test_uninstalled_hybrid_does_not_admit(self):
        result = repair_receipt()
        self.assertEqual(result["training_admission"], "denied")
        self.assertFalse(result["full_gpu_execution"])
        self.assertFalse(result["gpu_gemm_forced_wgm1"])

    def test_proc_maps_exact_lib_path_extraction(self):
        paths = mapped_paths(
            "7f00-7f01 r-xp 000 00:00 0 /nix/store/a/lib/librocblas.so\n"
            "7f01-7f02 r-xp 000 00:00 0 /nix/store/b/lib/libgfx803_sgemm_shim.so (deleted)\n"
        )
        self.assertEqual(paths, {
            "/nix/store/a/lib/librocblas.so",
            "/nix/store/b/lib/libgfx803_sgemm_shim.so",
        })

    def test_shim_without_root_fails_closed(self):
        with self.assertRaises((RuntimeError, FileNotFoundError)):
            verify_loaded_sgemm_shim(
                "/nonexistent/libgfx803_sgemm_shim.so",
                proc_maps="", env={"LD_PRELOAD": "/nonexistent/libgfx803_sgemm_shim.so"})

    def test_wgm1_join_requires_original_complete_correct_trace(self):
        good = {
            "schema_version": 2, "solution_id": 7, "attribution_complete": True,
            "no_training_admission": True,
            "profile_observer_effect_absent_for_first_output": True,
            "unprofiled_repeats_completed": 100,
            "unprofiled_numerics": "correct-at-tested-shape",
            "profiler_numerics": "correct-at-tested-shape",
            "teacher_sha256": "a"*64, "gpu_arch": "gfx803",
            "torch_hip": "6.4",
            "rocblas_sha256": "b"*64, "weight_sha256": "c"*64,
            "input_sha256": "d"*64,
            "gemm": {"transA_transB": ["T","N"], "M": 1152,"N":128,"K":384},
            "trace": {"status":"identified","wgm":1,
                      "unique_kernel_names":["Cijk_verified_WGM1"]},
        }
        validate_join(good,teacher_sha="a"*64,arch="gfx803",hip="6.4")
        for name in ("unprofiled_repeats_completed","unprofiled_numerics",
                     "teacher_sha256","gpu_arch","trace"):
            import copy
            bad=copy.deepcopy(good)
            if name == "trace":
                bad["trace"]["wgm"]=8
            elif name == "unprofiled_repeats_completed":
                bad[name]=2
            else:
                bad[name]="wrong"
            with self.assertRaises(RuntimeError):
                validate_join(bad,teacher_sha="a"*64,arch="gfx803",hip="6.4")

    def test_runner_has_valid_bash_syntax_and_help(self):
        script = Path(__file__).resolve().parents[1] / "scripts/run_gfx803_lila_repair_gate.sh"
        subprocess.run(["bash", "-n", str(script)], check=True)
        out = subprocess.run(["bash", str(script), "--help"],
                             capture_output=True, text=True)
        self.assertEqual(out.returncode, 0)
        self.assertIn("10.1-schaka", out.stderr)
        self.assertIn("6.4-cpu-linear", out.stderr)

    def test_runner_refuses_fake_lanes_without_gpu(self):
        script = Path(__file__).resolve().parents[1] / "scripts/run_gfx803_lila_repair_gate.sh"
        out = subprocess.run(["bash", str(script), "--lane", "unknown",
                              "--receipt", "should-not-create.json", "--",
                              "--teacher", "x", "--student", "y", "--train-ids", "z"],
                             capture_output=True, text=True)
        self.assertNotEqual(out.returncode, 0)


if __name__ == "__main__":
    unittest.main()
