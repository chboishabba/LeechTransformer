"""Statistical C3 2/3 rotating-plane null regression and negative controls."""
import unittest
import numpy as np
from scripts.audit_c3_rotating_energy_null import rotating_energy,result

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

    def test_reject_bad_shapes_and_zero_rows(self):
        with self.assertRaises(ValueError):
            rotating_energy(np.zeros((9,64,48)))
        with self.assertRaises(ValueError):
            rotating_energy(np.ones((9,64,47)))
        with self.assertRaises(ValueError):
            result(np.ones((1,64,48)),permutations=39,bootstraps=39)

if __name__=="__main__":unittest.main()
