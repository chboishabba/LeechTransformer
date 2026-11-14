"""Exact fixed-coordinate F1 order-three and PSD counterexample contracts."""
import unittest
import numpy as np
from scripts.audit_x369_f1_rank_and_orientation_null import (
    c3_full_rank_gauge_diagnostic)

class X369RankAndOrientationTests(unittest.TestCase):
    def test_psd_ceiling_saturated_for_one_noninvariant_axis(self):
        x=np.zeros((9,12,12))
        x[:,:,0]=np.arange(108).reshape(9,12)
        r=c3_full_rank_gauge_diagnostic(x,orientation_null_draws=29)
        self.assertAlmostEqual(r["observed_original_c3_full_commutator_defect"],
                               np.sqrt(2),places=12)
        self.assertLess(abs(r["coherent_c3_frobenius_overlap"]),1e-12)
        self.assertAlmostEqual(
            r["coherent_c3_orthogonal_complement_fraction"],np.sqrt(2/3),
            places=12)
        self.assertAlmostEqual(
            r["full_covariance_effective_rank_participation"],1.,places=12)

    def test_global_c3_can_be_better_than_heterogeneous_orientation(self):
        x=np.zeros((9,9,12))
        for row in range(9):
            for tok in range(9):
                phase=(row*9+tok)%3
                x[row,tok,np.arange(4)*3+phase]=1.
        r=c3_full_rank_gauge_diagnostic(x,orientation_null_draws=39)
        self.assertLess(
            r["observed_original_c3_full_commutator_defect"],1e-12)
        self.assertAlmostEqual(
            r["coherent_c3_reynolds_projection_fraction"],1.)
        self.assertGreater(r["orientation_null_mean"],.1)

    def test_random_noninvariant_null_deterministic(self):
        x=np.random.default_rng(91).normal(size=(9,12,24))
        a=c3_full_rank_gauge_diagnostic(
            x,seed=4,orientation_null_draws=39)
        b=c3_full_rank_gauge_diagnostic(
            x,seed=4,orientation_null_draws=39)
        self.assertEqual(a,b)
        self.assertGreater(a["coherent_c3_frobenius_overlap"],0.)
        self.assertTrue(a["orientation_null_available"])

    def test_one_triad_not_misclassified(self):
        x=np.random.default_rng(1).normal(size=(9,12,3))
        r=c3_full_rank_gauge_diagnostic(x)
        self.assertFalse(r["orientation_null_available"])

    def test_fail_closed(self):
        with self.assertRaises(ValueError):
            c3_full_rank_gauge_diagnostic(np.zeros((2,8,12)))
        with self.assertRaises(ValueError):
            c3_full_rank_gauge_diagnostic(np.ones((9,8,11)))
        with self.assertRaises(ValueError):
            c3_full_rank_gauge_diagnostic(
                np.random.default_rng(1).normal(size=(9,8,12)),
                orientation_null_draws=0)

if __name__=="__main__":unittest.main()
