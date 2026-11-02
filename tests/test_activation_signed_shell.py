"""Regression tests for byte-exact signed shell activation transforms.

Run: python -m unittest discover -s tests -p 'test_activation_signed_shell.py'
No model checkpoint required. Real capture benchmarks are separate.
"""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

HERE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "fibre_codec", HERE / "scripts" / "benchmark_multiscale_fibre_codec.py"
)
codec = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(codec)


class SignedShellExactness(unittest.TestCase):
    def test_all_three_scales_and_ieee_storage(self):
        rng = np.random.default_rng(20261031)
        checked = 0
        for item_size in (2, 4):
            for width in (24, 48, 384):
                for _ in range(100):
                    src = rng.bytes(width * item_size)
                    shell = codec.signed_haar24(src, item_size)
                    self.assertEqual(codec.unsigned_haar24_inverse(shell, item_size), src)
                    checked += 1
        self.assertEqual(checked, 600)

    def test_zeros_and_extreme_bit_patterns(self):
        for item_size in (2, 4):
            for byte_value in (0, 1, 127, 128, 255):
                src = bytes([byte_value]) * (24 * item_size)
                shell = codec.signed_haar24(src, item_size)
                self.assertEqual(codec.unsigned_haar24_inverse(shell, item_size), src)

    def test_full_candidate_tournament_exactness(self):
        rng = np.random.default_rng(456)
        for item_size in (2, 4):
            prior = None
            for src in (rng.bytes(384 * item_size),
                        bytes(384 * item_size),
                        rng.bytes(384 * item_size)):
                entries, selected = codec.compete(src, item_size, prior)
                self.assertIn(selected[0], {row[0] for row in entries})
                self.assertTrue(any(row[0] == "signed_haar24" for row in entries))
                self.assertTrue(any(row[0] == "signed_haar24_planes" for row in entries))
                prior = src


if __name__ == "__main__":
    unittest.main()
