import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_native_schaka101_lila.sh"


class NativeSchakaLauncherTests(unittest.TestCase):
    def test_dense_control_is_an_admitted_cuda_module(self):
        with tempfile.TemporaryDirectory() as directory:
            wrapper = Path(directory) / "wrapper"
            args = Path(directory) / "args.txt"
            wrapper.write_text(
                "#!/usr/bin/env bash\n"
                "if [[ $1 == -c ]]; then exit 0; fi\n"
                "printf '%s\\n' \"$@\" > \"$NATIVE_LAUNCHER_ARGS\"\n"
            )
            wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
            result = subprocess.run(
                ["bash", str(LAUNCHER), "--module", "scripts.train_early_layer_dense_control", "--",
                 "--teacher", "teacher.pt"],
                cwd=ROOT,
                text=True,
                capture_output=True,
                env={**os.environ, "GFX803_SCHAKA101_PYTHON": str(wrapper),
                     "NATIVE_LAUNCHER_ARGS": str(args)},
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("scripts.train_early_layer_dense_control", args.read_text())
            self.assertIn("--device", args.read_text())
            self.assertIn("cuda", args.read_text())

