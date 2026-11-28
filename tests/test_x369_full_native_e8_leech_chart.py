"""Full 369 native 9 trits, pointed-nine Fourier/E8 prior and nonclosure.

No Agda kernel, neural source checkpoint or GPU needed for exact SymPy tests.
"""
import unittest
import sympy as S
from itertools import product
from scripts.check_x369_full_native_e8_leech_chart import (
    native_vector,encode_native,decode_real,native_admission,
    roots_e8,reflect_wave,preserve_or_residual,certificate)


class Full369E8LeechChart(unittest.TestCase):
    def test_all_native_fabric_points_reopen(self):
        self.assertEqual(certificate()['exact_bidi_points'],19683)

    def test_all_ternary_nine_points_reopen_with_original_centre(self):
        from scripts.certify_punctured_nine_exact import fourier_basis
        B=fourier_basis()
        count=0
        for x in product((-1,0,1),repeat=9):
            wave,centre=encode_native(x,B)
            self.assertEqual(native_admission(decode_real(wave,centre,B)),x)
            count+=1
        self.assertEqual(count,19683)

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

    def test_malformed_native_and_nonroot_rejected(self):
        with self.assertRaises(ValueError):encode_native((0,)*8)
        with self.assertRaises(ValueError):encode_native((0,)*8+(2,))
        with self.assertRaises(ValueError):reflect_wave(S.zeros(8,1),S.zeros(8,1))
        with self.assertRaises(ValueError):decode_real(S.zeros(9,1),S.Integer(0))


if __name__=='__main__':
    unittest.main()
