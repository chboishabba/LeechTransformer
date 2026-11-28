"""Finite E8^3 24D root and Leech modular-candidate separation."""
import unittest
from scripts.check_x369_e8_three_leech_glue_obstruction import (
    certificate,triple_e8_norm_two_roots)


class E8ThreeLeechGlueObstruction(unittest.TestCase):
    def test_all_720_direct_sum_roots_are_distinct_and_norm_two(self):
        roots=triple_e8_norm_two_roots()
        self.assertEqual(len(roots),720)
        self.assertEqual(len(set(tuple(r) for r in roots)),720)
        self.assertTrue(all((r.T*r)[0]==2 for r in roots))

    def test_modular_and_direct_sum_rootless_boundary(self):
        r=certificate()
        self.assertEqual(r['ambient_dimension'],24)
        self.assertEqual(r['three_e8_direct_sum_q1'],720)
        self.assertEqual(r['leech_modular_candidate_q1'],0)
        self.assertEqual(r['leech_modular_candidate_q2'],196560)
        self.assertFalse(r['leech_lattice_gluing_cosets_supplied_here'])
        self.assertFalse(r['existing_leech_actual_minimal_vector_enumeration'])
        self.assertFalse(r['original_lila_e8_prior_equivariant_to_leech'])


if __name__=='__main__':
    unittest.main()
