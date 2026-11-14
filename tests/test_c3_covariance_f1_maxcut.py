"""F1 full covariance and row cluster regressions."""
import unittest
import numpy as np
from scripts.audit_c3_covariance_f1_maxcut import audit_layers,pooled_metrics,full_metrics
class C3CovarianceF1Tests(unittest.TestCase):
    def test_full_detects_cross_triad_structure_pooled_cannot(self):
        rng=np.random.default_rng(4)
        x=rng.normal(size=(20,32,12));x[...,3:6]=x[...,:3]
        d=full_metrics(x)
        self.assertTrue(d["full_hidden_covariance_tested"])
        self.assertGreater(d["off_triad_covariance_frobenius_fraction"],.2)
        self.assertEqual(d["per_triad_c3_defect_summary"]["num_triads"],4)
    def test_clustered_simultaneous_adjustment(self):
        rng=np.random.default_rng(7);x=rng.normal(size=(9,16,12))
        d=audit_layers([x,2*x],seed=8,bootstraps=39)
        self.assertAlmostEqual(d["bonferroni_tail_each_side"],.0125)
        np.testing.assert_allclose(d["layers"][0]["cluster_bootstrap_commutator_interval"],
                                   d["layers"][1]["cluster_bootstrap_commutator_interval"])
        self.assertTrue(d["no_population_isotropy_or_c3_symmetry_proof"])
    def test_c3_symmetric_but_not_isotropic(self):
        rng=np.random.default_rng(9)
        x=np.repeat(rng.normal(size=(40,20,4,1)),3,axis=-1).reshape(40,20,12)
        d=pooled_metrics(x)
        self.assertLess(d["c3_commutator_defect"],1e-12)
        self.assertGreater(d["isotropy_defect"],.7)
    def test_fail_closed_shape_cluster_count_and_tolerance(self):
        with self.assertRaises(ValueError):
            audit_layers([np.ones((2,10,12))],bootstraps=19)
        with self.assertRaises(ValueError):
            audit_layers([np.ones((3,10,12)),np.ones((4,10,12))],bootstraps=19)
        with self.assertRaises(ValueError):
            audit_layers([np.ones((3,10,12))],bootstraps=19,tolerance=-1)
if __name__=="__main__":unittest.main()
