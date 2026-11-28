"""Original signed-source coordinate S3 versus trit palette S3 graph boundary."""
import unittest
import sympy as S
from models.geometric_reflection_fibre import GROUP,mul,IDENTITY
from scripts.check_x369_original_marked_fabric_signed_graph import (
    certificate,monomial9,full_signed_action,strict_p3_nine_adjacent)

class OriginalSignedFabricGrid(unittest.TestCase):
    def test_exact_full_source_coordinate_action(self):
        r=certificate()
        self.assertEqual(r['full_original_native_trit_states'],19683)
        self.assertEqual(r['full_nonwrapping_P3_nine_edges'],118098)
        self.assertEqual(r['periodic_C3_nine_edges'],177147)
        self.assertEqual(r['source_9x9_signed_monomial_witnesses'],12)
        self.assertEqual(r['group_composition_matrices_checked'],144)
        self.assertEqual(r['representative_grid_edges_checked'],216)
        self.assertTrue(r['every_full_native_P3_edge_preserved_by_monomial_proof'])
        self.assertFalse(r['original_neural_recurrent_equivariance_proved'])

    def test_original_s3_group_and_inverse_on_full_nine_coordinates(self):
        p=(1,0,-1,-1,0,1,1,-1,0)
        for g in GROUP:
            self.assertEqual(tuple(monomial9(g)*S.Matrix(p)),
                             full_signed_action(g,p))
            self.assertEqual(monomial9(g).T*monomial9(g),S.eye(9))
            for h in GROUP:
                self.assertEqual(full_signed_action(g,full_signed_action(h,p)),
                                 full_signed_action(mul(g,h),p))

    def test_palette_rotation_is_not_grid_automorphism(self):
        low=(0,)*9
        high=(1,)+(0,)*8
        self.assertTrue(strict_p3_nine_adjacent(low,high))
        def palette(v):
            return tuple({-1:0,0:1,1:-1}[x] for x in v)
        self.assertFalse(strict_p3_nine_adjacent(palette(low),palette(high)))
        self.assertFalse(strict_p3_nine_adjacent((1,)+(0,)*8,(-1,)+(0,)*8))

    def test_original_seed_zero_tail_closed_but_not_full_model_claim(self):
        p=(0,0,0,1,-1,0,0,0,0)
        for g in GROUP:
            q=full_signed_action(g,p)
            self.assertEqual(q[6:],(0,0,0))
        with self.assertRaises(ValueError):
            full_signed_action(IDENTITY,(0,)*8)

if __name__=='__main__':
    unittest.main()
