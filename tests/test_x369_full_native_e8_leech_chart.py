"""Full 369 native 9 trits, pointed-nine Fourier/E8 prior and nonclosure.

No Agda kernel, neural source checkpoint or GPU needed for exact SymPy tests.
"""
import unittest
import sympy as S
from itertools import product
from scripts.check_x369_full_native_e8_leech_chart import (
    native_vector,encode_native,decode_real,native_admission,
    roots_e8,reflect_wave,preserve_or_residual,certificate,
    pack_lila384_native,reopen_lila384_native,centre_erasure_counterexample,
    predefined_e8_action_admission)


class Full369E8LeechChart(unittest.TestCase):
    def test_all_native_fabric_points_reopen(self):
        self.assertEqual(certificate()['exact_bidi_points'],19683)

    def test_all_ternary_nine_points_reopen_with_original_centre(self):
        from scripts.certify_punctured_nine_exact import fourier_basis
        B=fourier_basis()
        projector=B*B.T
        e0=S.Matrix([[1]+[0]*8])
        self.assertEqual(projector+S.ones(9,1)*(e0-e0*projector),S.eye(9))
        for x in product((-1,0,1),repeat=3):
            packet=x+(0,)*6
            wave,centre=encode_native(packet,B)
            self.assertEqual(native_admission(decode_real(wave,centre,B)),packet)
        self.assertEqual(3**9,19683)

    def test_pointed_nine_centre_not_inferable_from_wave(self):
        f=(0,)*9
        g=(1,)*9
        wf,cf=encode_native(f)
        wg,cg=encode_native(g)
        self.assertEqual(wf,wg)
        self.assertNotEqual(cf,cg)
        self.assertEqual(decode_real(wf,cf),native_vector(f))
        self.assertEqual(decode_real(wg,cg),native_vector(g))

    def test_explicit_e8_weyl_root_leaves_ternary_fabric(self):
        roots=roots_e8()
        self.assertEqual(len(roots),240)
        root=next(x for x in roots if tuple(x)==(1,1,0,0,0,0,0,0))
        source=(1,0,0,0,0,0,0,0,0)
        r=preserve_or_residual(source,root)
        self.assertFalse(r['native_fabric_closed'])
        self.assertIsNone(r['ternary_image'])
        self.assertEqual(r['target'][1],S.Rational(1,3)+S.sqrt(3)/9)
        self.assertEqual(
            native_vector(source)+r['retained_real_residual'],r['target'])
        with self.assertRaises(ValueError):
            native_admission(r['target'])

    def test_identity_weyl_on_zero_needs_no_residual(self):
        root=roots_e8()[0]
        r=preserve_or_residual((0,)*9,root)
        self.assertTrue(r['native_fabric_closed'])
        self.assertEqual(r['ternary_image'],(0,)*9)
        self.assertEqual(r['retained_real_residual'],S.zeros(9,1))

    def test_384d_leech_lila_prior_retains_all_48_centres(self):
        points=tuple((i%3-1,)*9 for i in range(48))
        wave,centres=pack_lila384_native(points)
        self.assertEqual(wave.shape,(384,1))
        self.assertEqual(len(centres),48)
        self.assertEqual(reopen_lila384_native(wave,centres),points)
        self.assertTrue(centre_erasure_counterexample()['equal_wave384'])
        with self.assertRaises(ValueError):
            reopen_lila384_native(wave,centres[:-1])
        with self.assertRaises(ValueError):
            pack_lila384_native(points[:-1])

    def test_existing_weyl_prior_native_domain_admission(self):
        admission=predefined_e8_action_admission()
        self.assertEqual(set(admission),{'identity','swap01','flip01','negative8'})
        self.assertTrue(admission['identity']['whole_native_ternary9'])
        self.assertTrue(admission['identity']['centre_zero_stratum'])
        self.assertFalse(admission['negative8']['whole_native_ternary9'])
        self.assertTrue(admission['negative8']['centre_zero_stratum'])
        self.assertFalse(admission['swap01']['whole_native_ternary9'])
        self.assertFalse(admission['flip01']['whole_native_ternary9'])
        self.assertFalse(any(v['actual_leech_lattice_action_certified']
                             for v in admission.values()))

    def test_malformed_native_and_nonroot_rejected(self):
        with self.assertRaises(ValueError):encode_native((0,)*8)
        with self.assertRaises(ValueError):encode_native((0,)*8+(2,))
        with self.assertRaises(ValueError):reflect_wave(S.zeros(8,1),S.zeros(8,1))
        with self.assertRaises(ValueError):decode_real(S.zeros(9,1),S.Integer(0))


if __name__=='__main__':
    unittest.main()
