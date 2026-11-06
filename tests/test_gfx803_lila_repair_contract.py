"""CPU-only contract checks for gfx803 LILA repair paths, no hardware admission."""
import os
from pathlib import Path
import subprocess
import unittest

from scripts.gfx803_linear_repair import repair_receipt
from scripts.gfx803_repair_runtime import mapped_paths, verify_loaded_sgemm_shim


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
