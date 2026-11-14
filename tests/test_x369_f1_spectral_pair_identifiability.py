"""One-pair spectral C3 source-only identifiability regressions."""
import unittest
import numpy as np
from scripts.audit_x369_f1_spectral_pair_identifiability import spectral_pair_screen

class X369SpectralPairTests(unittest.TestCase):
    def test_rank_one_nullspace_symmetry_is_vacuous(self):
        x=np.zeros((9,6,12))
        x[:,:,0]=np.arange(54,dtype=float).reshape(9,6)
        r=spectral_pair_screen(x)
        self.assertAlmostEqual(r["original_covariance_participation_rank"],1.)
        unrestricted=r["single_pair_order_three_screens"][0]
        self.assertAlmostEqual(unrestricted[
            "achievable_single_pair_c3_covariance_defect"],0.,places=10)
        self.assertLess(unrestricted["actual_rotating_trace_share"],1e-10)
        forced=r["single_pair_order_three_screens"][-1]
        self.assertGreater(forced[
            "achievable_single_pair_c3_covariance_defect"],1.)
        self.assertGreater(forced["actual_rotating_trace_share"],.25)
        self.assertFalse(r["covariance_only_c3_representation_identifiable"])

    def test_equal_active_eigenvalues_support_exact_rotating_pair(self):
        x=np.zeros((9,6,6))
        for row in range(9):
            x[row,:,0]=[1.,-1.,0.,0.,0.,0.]
            x[row,:,1]=[0.,0.,1.,-1.,0.,0.]
        r=spectral_pair_screen(x)
        forced=r["single_pair_order_three_screens"][-1]
        self.assertTrue(forced["pair_exists"])
        self.assertAlmostEqual(forced["actual_rotating_trace_share"],1.)
        self.assertLess(forced[
            "achievable_single_pair_c3_covariance_defect"],1e-10)
        self.assertGreater(r["observed_original_coherent_triads_c3_defect"],.1)
        self.assertFalse(r["learned_c3_phase_or_semantic_equivariance_proved"])

    def test_source_document_top_pc_concentration(self):
        x=np.zeros((9,6,12))
        x[0,:,0]=np.array([-20,20,-20,20,-20,20])
        for row in range(1,9):
            x[row,:,0]=[-1,1,-1,1,-1,1]
        r=spectral_pair_screen(x)
        self.assertGreater(r["largest_document_share_of_top_pc_energy"],.9)
        self.assertLess(r["effective_document_count_for_top_pc_energy"],2.)

    def test_duplicate_thresholds_and_bad_values_fail_closed(self):
        x=np.random.default_rng(4).normal(size=(9,6,12))
        for thresholds in ((0.,0.),(-.1,.1),(.5,.2),(0.,float("nan"))):
            with self.assertRaises(ValueError):
                spectral_pair_screen(x,min_rotating_shares=thresholds)
        with self.assertRaises(ValueError):
            spectral_pair_screen(x[:2])
        with self.assertRaises(ValueError):
            spectral_pair_screen(np.zeros((9,6,12)))

    def test_rank_invariance_under_global_rescaling(self):
        x=np.random.default_rng(12).normal(size=(9,6,12))
        a=spectral_pair_screen(x)
        b=spectral_pair_screen(3*x)
        self.assertAlmostEqual(a["original_covariance_participation_rank"],
                               b["original_covariance_participation_rank"])
        self.assertAlmostEqual(a["observed_original_coherent_triads_c3_defect"],
                               b["observed_original_coherent_triads_c3_defect"])
        for first,second in zip(a["single_pair_order_three_screens"],
                                b["single_pair_order_three_screens"]):
            self.assertEqual(first["pair_exists"],second["pair_exists"])
            if first["pair_exists"]:
                self.assertAlmostEqual(
                    first["achievable_single_pair_c3_covariance_defect"],
                    second["achievable_single_pair_c3_covariance_defect"])

if __name__=="__main__":
    unittest.main()
