"""Fail-closed protocol tests for frozen X369-F5 matched lockbox."""
import unittest
import numpy as np

from scripts.score_frozen_c2c3_lockbox import (
    validate_lockbox, select_on_development, arm_checkpoints)


class FrozenLockboxContractTests(unittest.TestCase):
    def setUp(self):
        self.original = np.arange(24, dtype=np.int64).reshape(1, 24)
        self.viewed = np.arange(100, 124, dtype=np.int64).reshape(1, 24)
        self.selection = np.arange(200, 224, dtype=np.int64).reshape(1, 24)
        self.locked = np.arange(300, 324, dtype=np.int64).reshape(1, 24)

    def test_all_seen_sources_and_selection_rejected(self):
        self.assertGreater(validate_lockbox(
            self.locked, self.original, self.viewed, self.selection), 0)
        for reused in (self.original, self.viewed, self.selection):
            with self.subTest(reused=reused[0, 0]), self.assertRaises(ValueError):
                validate_lockbox(
                    reused, self.original, self.viewed, self.selection)
        near = np.arange(208, 232, dtype=np.int64).reshape(1, 24)
        with self.assertRaises(ValueError):
            validate_lockbox(
                near, self.original, self.viewed, self.selection)

    def test_duplicate_and_width_mismatch_fail_closed(self):
        with self.assertRaises(ValueError):
            validate_lockbox(np.repeat(self.locked, 2, axis=0),
                             self.original, self.viewed, self.selection)
        with self.assertRaises(ValueError):
            validate_lockbox(np.arange(25).reshape(1,25),
                             self.original, self.viewed, self.selection)

    def test_selection_uses_dev_mean_not_test_or_best_single_seed(self):
        receipt = {"paired": [
            {"seed": 41, "c2_best_ce": 4.0, "c2c3_best_ce": 3.0},
            {"seed": 42, "c2_best_ce": 3.0, "c2c3_best_ce": 3.5},
        ]}
        winner, means, seeds = select_on_development(receipt)
        self.assertEqual(winner, "c2c3_wave")
        self.assertEqual(seeds, {41, 42})
        self.assertAlmostEqual(means["c2_invariant"], 3.5)
        self.assertAlmostEqual(means["c2c3_wave"], 3.25)
        receipt["paired"][1]["c2c3_best_ce"] = 4.0
        self.assertEqual(select_on_development(receipt)[0], "c2_invariant")
        receipt["paired"][1]["c2c3_best_ce"] = 4.0
        receipt["paired"][0]["c2c3_best_ce"] = 3.0
        receipt["paired"][0]["c2_best_ce"] = 2.0
        receipt["paired"][1]["c2_best_ce"] = 5.0
        self.assertEqual(select_on_development(receipt)[0], "c2_invariant")

    def test_duplicate_seed_and_missing_candidates_fail_closed(self):
        paired = [{"seed": 41, "c2_best_ce": 4., "c2c3_best_ce": 3.},
                  {"seed": 41, "c2_best_ce": 4., "c2c3_best_ce": 3.}]
        with self.assertRaises(ValueError):
            select_on_development({"paired": paired})
        arms = [{"seed": 41, "geometry": "c2_invariant",
                 "candidate_checkpoint": "path.pt",
                 "candidate_checkpoint_sha256": "sha"}]
        with self.assertRaises(ValueError):
            arm_checkpoints({"arms": arms}, {41})
        both = arms + [dict(arms[0], geometry="c2c3_wave")]
        self.assertEqual(len(arm_checkpoints({"arms": both}, {41})), 2)


if __name__ == "__main__":
    unittest.main()
