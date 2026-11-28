"""Exact full-hyperfabric and marked-source regression suite.
No GPU/training/Agda acceptance inferred from these finite tests.
Original Leech-LILA Anatolii Kornienko; DASHI extension.
"""
import unittest
from itertools import permutations
from fractions import Fraction

from scripts.check_x369_native_hyperfabric_nine_maxcut import (
    audit as full_audit, encode, decode, rotate, split, join, route)
from scripts.check_x369_marked_source_seed_hyperfabric import (
    audit as marked_audit, packet, reopen, paired_initial, unpair_initial,
    MEANS, TRIADS, ZERO)


class NativeFullNineHyperformal(unittest.TestCase):
    def test_exhaustive_source_equivalent_six_faces(self):
        v=full_audit()
        self.assertEqual(v['full_nine_points'],19683)
        self.assertEqual(v['original_H07_face_checks'],118098)
        self.assertEqual(v['original_H07_terminal_path_checks'],118098)
        self.assertEqual(v['nonwrapping_P3_9_edges'],118098)
        self.assertEqual(v['periodic_C3_9_edges'],177147)
        self.assertEqual(v['zero_sum_nine_fibre'],3139)

    def test_prefix_vs_appraisal_collision(self):
        p=(0,)*9
        q=(0,)*6+(1,0,0)
        self.assertEqual(split(p)[0],split(q)[0])
        self.assertNotEqual(p,q)
        self.assertEqual(join(split(p)),p)
        self.assertEqual(join(split(q)),q)
        self.assertNotEqual(encode(q),encode(p))
        for w in permutations('XYZ'):
            self.assertEqual(decode(route(q,w)),rotate(q))

    def test_zero_and_binary_forgets_sign(self):
        a=(-1,0,1)*3
        b=(1,0,-1)*3
        self.assertEqual(tuple(mask for mask,sign in encode(a)),
                         tuple(mask for mask,sign in encode(b)))
        self.assertNotEqual(encode(a),encode(b))
        self.assertEqual(decode(encode(a)),a)
        self.assertEqual(decode(encode(b)),b)
        with self.assertRaises(ValueError):
            decode(((0,1),)*9)


class OriginalMarkedSourceSeeds(unittest.TestCase):
    def test_complete_declared_seeds(self):
        receipt=marked_audit()
        self.assertEqual(receipt['exact_source_seed_packets'],189)
        self.assertTrue(receipt['third_appraisal_forced_to_zero'])
        self.assertFalse(receipt['neural_transition_factors_through_seed'])

    def test_all_mean_residual_pair_reopening(self):
        for m in MEANS:
            for d in TRIADS:
                p=packet(m,d)
                self.assertEqual(p[6:],ZERO)
                self.assertEqual(reopen(p),(m,d))
                branches=paired_initial(m,d)
                self.assertEqual(unpair_initial(*branches),(m,d))
                self.assertTrue(all(isinstance(x,Fraction) for row in branches for x in row))

    def test_nonmarked_source_refused(self):
        with self.assertRaises(ValueError):
            reopen((0,)*6+(1,0,0))
        with self.assertRaises(ValueError):
            packet((1,1,0),(0,0,0))
        with self.assertRaises(ValueError):
            reopen((1,1,0)+(0,)*6)

if __name__=='__main__':
    unittest.main()
