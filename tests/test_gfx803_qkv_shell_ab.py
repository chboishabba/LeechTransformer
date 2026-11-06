"""CPU-only enforcement of WGM/SGEMM A/B operator scripts.

A source-level Bash contract is not a GPU result, but it prevents future
accidental bypasses of the original-model fixture, same-runtime policy and
mandatory trace observer-effect checks.
"""
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]


class OperatorShellABContracts(unittest.TestCase):
    def test_both_shell_scripts_parse_without_running_nix_or_gpu(self):
        for rel in ("scripts/run_gfx803_qkv_solution_ab_nix.sh",
                    "scripts/run_gfx803_qkv_shim_ab.sh"):
            with self.subTest(file=rel):
                result = subprocess.run(["bash", "-n", str(ROOT / rel)],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_solution_scan_uses_real_fixture_and_100_plus_repeats(self):
        s = (ROOT / "scripts/run_gfx803_qkv_solution_ab_nix.sh").read_text()
        for snippet in (
            "scripts.export_gfx803_qkv_fixture",
            "scripts.gfx803_qkv_solution_ab",
            "--expected-input-sha",
            "--repeats",
            "rocprofv3",
            "scripts.parse_gfx803_qkv_solution_ab",
            "artifact-index.json",
        ):
            with self.subTest(symbol=snippet):
                self.assertIn(snippet, s)
        self.assertNotIn("python -m train", s)
        self.assertNotIn("torch.optim", s)
        self.assertNotIn('> "$dest/join.log" 2>&1 || true', s)

    def test_shim_preload_only_applies_to_child_process(self):
        s = (ROOT / "scripts/run_gfx803_qkv_shim_ab.sh").read_text()
        self.assertIn('timeout --kill-after=15 "$TIMEOUT_SECS"', s)
        self.assertIn('env LD_PRELOAD="$SHIM"', s)
        self.assertLess(s.index('timeout --kill-after=15 "$TIMEOUT_SECS"'),
                        s.index('env LD_PRELOAD="$SHIM"'))
        self.assertIn('GFX803_SGEMM_SHIM_DISABLE="$disabled"', s)
        self.assertIn('scripts.parse_gfx803_qkv_shim_ab', s)
        self.assertNotIn('scripts.parse_gfx803_qkv_shim_ab --out-dir "$OUT" || true', s)

    def test_both_scanners_refuse_stale_output_and_weak_repetition(self):
        for rel in ("scripts/run_gfx803_qkv_solution_ab_nix.sh",
                    "scripts/run_gfx803_qkv_shim_ab.sh"):
            with self.subTest(path=rel):
                source = (ROOT / rel).read_text()
                self.assertIn("REPEATS < 100", source)
                self.assertIn('find "$OUT" -mindepth 1 -print -quit', source)
                self.assertIn("Refuse pre-existing", source)
                self.assertIn("exit 2", source)

    def test_original_6_4_scan_rejects_foreign_shim(self):
        s = (ROOT / "scripts/run_gfx803_qkv_solution_ab_nix.sh").read_text()
        self.assertIn("libgfx803_sgemm_shim", s)
        self.assertIn("gfx803-pytorch-stack", s)
        self.assertNotIn("gfx803-pytorch-stack-upgrade", s)


if __name__ == "__main__":
    unittest.main()
