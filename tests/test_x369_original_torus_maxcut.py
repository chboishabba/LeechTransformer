"""Exact nine-site universal coefficient regression; not neural evidence."""
import unittest
from scripts.certify_x369_original_torus_maxcut import (
    certify,e_mul,e_conj,omega,translate_offsets,PUNCTURE_SITES)

class OriginalTorusMaxCutTests(unittest.TestCase):
    def test_full_universal_coefficient_certificate(self):
        r=certify()
        self.assertEqual(r['inverse_fourier_matrix_coefficients_checked'],81)
        self.assertEqual(r['convolution_bilinear_coefficients_checked'],729)
        self.assertEqual(r['pointwise_to_frequency_convolution_coefficients_checked'],729)
        self.assertEqual(r['parseval_sesquilinear_coefficients_checked'],81)
        self.assertEqual(r['real_conjugate_basis_coefficients_checked'],81)
        self.assertEqual(r['induced_translation_group_basis_coefficients_checked'],5832)
        self.assertEqual(r['centred_gram_bilinear_coefficients_checked'],729)
        self.assertEqual(r['punctured_eight_naive_norm_counterexample'],
                         {'before':8,'after_translation_1_0':1})
        self.assertFalse(r['agda_kernel_verified'])
    def test_eisenstein_ring_conjugation(self):
        w=omega(1)
        self.assertEqual(e_mul(e_mul(w,w),w),(1,0))
        self.assertEqual(e_conj(w),omega(-1))
        self.assertEqual(e_conj(e_conj(w)),w)
    def test_induced_translation_inverse(self):
        offsets={u:int(u==PUNCTURE_SITES[0]) for u in PUNCTURE_SITES}
        self.assertEqual(translate_offsets(translate_offsets(offsets,(1,0)),(-1,0)),offsets)

if __name__=='__main__':unittest.main()
