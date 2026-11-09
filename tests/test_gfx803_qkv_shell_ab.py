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
        self.assertIn('CHILD_PRELOAD="$SHIM"', s)
        self.assertIn('CHILD_PRELOAD="$ROCBLAS_PRELOAD:$SHIM"', s)
        self.assertLess(s.index('timeout --kill-after=15 "$TIMEOUT_SECS"'),
                        s.index('env LD_PRELOAD="$CHILD_PRELOAD"'))
        self.assertIn('GFX803_SGEMM_SHIM_DISABLE="$disabled"', s)
        self.assertIn('scripts.parse_gfx803_qkv_shim_ab', s)
        self.assertNotIn('scripts.parse_gfx803_qkv_shim_ab --out-dir "$OUT" || true', s)
        self.assertIn("do not resolve the selected", s)
        self.assertNotIn('SHIM="$(realpath "$SHIM")" PYTHON_BIN=', s)

    def test_scanners_refuse_stale_output_and_enforce_their_repeat_policy(self):
        source = (ROOT / "scripts/run_gfx803_qkv_solution_ab_nix.sh").read_text()
        self.assertIn("REPEATS < 3", source)
        self.assertIn('find "$OUT" -mindepth 1 -print -quit', source)
        self.assertIn("Refuse pre-existing", source)
        shim = (ROOT / "scripts/run_gfx803_qkv_shim_ab.sh").read_text()
        self.assertIn("TRIAL == 1 && REPEATS != 3", shim)
        self.assertIn("TRIAL == 0 && REPEATS < 100", shim)
        self.assertIn('find "$OUT" -mindepth 1 -print -quit', shim)
        self.assertIn("Refuse pre-existing", shim)

    def test_schaka_trial_is_explicit_three_repeat_and_nonpromotional(self):
        s = (ROOT / "scripts/run_gfx803_qkv_shim_ab.sh").read_text()
        for required in (
            '--trial) TRIAL=1; PROFILE=0; shift',
            'TRIAL == 1 && REPEATS != 3',
            'TRIAL == 0 && REPEATS < 100',
            '"trial_complete":complete',
            '"interposer_effect_attributed":False',
            '"gpu_training_admitted":False',
            'shim_ab_summary.json',
        ):
            self.assertIn(required, s)
        self.assertIn('if ((PROFILE)); then', s)

    def test_10_1_profiling_never_uses_legacy_host_rocm(self):
        s = (ROOT / "scripts/run_gfx803_qkv_shim_ab.sh").read_text()
        self.assertIn('"$ROOT/core-$RUNTIME_LAYOUT_VERSION/bin/rocprofv3"', s)
        self.assertIn('"$PROFILE_BIN" --runtime-trace', s)
        self.assertIn("Refuse inherited LD_PRELOAD", s)
        self.assertNotIn('LD_PRELOAD=/usr/lib/libstdc++.so.6', s)
        self.assertNotIn('PATH="/usr/bin:/bin:/opt/rocm/bin:$PATH"', s)

    def test_original_6_4_scan_rejects_foreign_shim(self):
        s = (ROOT / "scripts/run_gfx803_qkv_solution_ab_nix.sh").read_text()
        self.assertIn("libgfx803_sgemm_shim", s)
        self.assertIn("gfx803-pytorch-stack", s)
        self.assertNotIn("gfx803-pytorch-stack-upgrade", s)


if __name__ == "__main__":
    unittest.main()
