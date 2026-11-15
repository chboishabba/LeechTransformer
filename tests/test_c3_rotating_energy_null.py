"""Statistical C3 2/3 rotating-plane null regression and negative controls."""
import unittest
import numpy as np
from scripts.audit_c3_rotating_energy_null import rotating_energy,result,covariance_diagnostics

class C3RotatingEnergyNullTests(unittest.TestCase):
    def test_isotropic_like_data_has_near_two_thirds(self):
        rng=np.random.default_rng(117)
        x=rng.normal(size=(9,64,48))
        r=result(x,seed=12,permutations=39,bootstraps=39)
        self.assertLess(abs(r["observed_rotating_fraction"]-2./3.),.06)
        self.assertFalse(r["trained_C3_circuit_proved"])
        self.assertFalse(r["full_residual_isotropy_proved"])

    def test_pure_dc_is_zero_rotating(self):
        x=np.ones((9,64,16,3),dtype=np.float64).reshape(9,64,48)
        np.testing.assert_allclose(rotating_energy(x),np.zeros(9),atol=1e-15)

    def test_pure_rotating_is_full(self):
        x=np.tile(np.array([1.,-1.,0.]),16).reshape(1,1,48)
        x=np.broadcast_to(x,(9,64,48)).copy()
        np.testing.assert_allclose(rotating_energy(x),np.ones(9),atol=1e-15)

    def test_c3_invariant_but_not_isotropic_covariance(self):
        # Equal-phase common mode plus independent signed phase components:
        # permutation symmetry does NOT force the isotropic 2/3 energy.
        rng=np.random.default_rng(23)
        common=rng.normal(size=(20,64,16,1))*3.
        transverse=rng.normal(size=(20,64,16,3))*.05
        x=(common+transverse).reshape(20,64,48)
        c=covariance_diagnostics(x)
        self.assertGreater(c["centered_isotropy_relative_frobenius_defect"],.5)
        self.assertLess(c["centered_c3_rotation_covariance_commutator_defect"],.08)
        self.assertLess(float(rotating_energy(x).mean()),.05)

    def test_second_moment_vs_centered_covariance_mean_shift(self):
        # The raw 2/3 energy criterion is not implied by isotropic
        # *centred* noise when the source mean is strongly DC.
        rng=np.random.default_rng(18)
        x=(rng.normal(size=(20,64,48))+.8)
        c=covariance_diagnostics(x)
        self.assertLess(c["centered_isotropy_relative_frobenius_defect"],.10)
        self.assertLess(float(rotating_energy(x).mean()),.5)

    def test_reject_bad_shapes_and_zero_rows(self):
        with self.assertRaises(ValueError):
            rotating_energy(np.zeros((9,64,48)))
        with self.assertRaises(ValueError):
            rotating_energy(np.ones((9,64,47)))
        with self.assertRaises(ValueError):
            result(np.ones((1,64,48)),permutations=39,bootstraps=39)

if __name__=="__main__":unittest.main()
