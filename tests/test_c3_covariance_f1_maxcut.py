"""F1 full covariance and row cluster regressions."""
import unittest
import numpy as np
from scripts.audit_c3_covariance_f1_maxcut import audit_layers,pooled_metrics,full_metrics,full_hidden_cluster_defects
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
    def test_full_bootstrap_equals_explicit_recomputed_covariance(self):
        rng=np.random.default_rng(71)
        x=rng.normal(size=(9,11,12))
        draws=rng.integers(9,size=(25,9))
        fast=full_hidden_cluster_defects(x,draws)
        brute=np.asarray([full_metrics(x[idx])[
            "full_c3_commutator_defect"] for idx in draws])
        np.testing.assert_allclose(fast,brute,rtol=1e-12,atol=1e-12)
        scaled=full_hidden_cluster_defects(3*x,draws)
        np.testing.assert_allclose(fast,scaled,rtol=1e-12,atol=1e-12)

    def test_pooled_interval_not_mislabeled_full_hidden(self):
        rng=np.random.default_rng(88)
        x=rng.normal(size=(9,32,12))
        result=audit_layers([x],seed=41,bootstraps=39)
        layer=result["layers"][0]
        self.assertIn("cluster_bootstrap_full_hidden_commutator_interval",layer)
        self.assertIn("full_hidden_against_predeclared_tolerance",layer)
        self.assertTrue(layer["full_hidden_bootstrap_includes_off_triad_blocks"])
        self.assertTrue(result["full_hidden_inference_not_pooled_inference"])
        # Statistic definitions and sampling are different even with shared rows.
        self.assertNotEqual(layer["cluster_bootstrap_commutator_interval"],
                            layer["cluster_bootstrap_full_hidden_commutator_interval"])

    def test_reject_fake_full_cluster_draws(self):
        x=np.random.default_rng(12).normal(size=(9,12,12))
        with self.assertRaises(ValueError):
            full_hidden_cluster_defects(x,np.zeros((5,8),dtype=int))
        with self.assertRaises(ValueError):
            full_hidden_cluster_defects(x,np.full((5,9),9,dtype=int))

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
