import unittest
import sympy as S
from scripts.x369_transported_seed_action import SeedAction,row,replay_conflict
from scripts.x369_exact_operator_transport import CONTROLS

class SeedTransportTest(unittest.TestCase):
    def test_bidi(self):
        B=S.eye(6).row_join(S.Matrix(6,3,lambda i,j:S.Rational(i+j+1,7)))
        A=S.eye(6).row_join(S.zeros(6,3))
        c=SeedAction(B,A)
        self.assertEqual(c.certificate()['relations'],6)
        h=row(tuple(S.Rational(i-3,5) for i in range(9)))
        x,r=c.split(h);y,s=c.target(h)
        self.assertEqual(c.reverse(y,s),h)
        for u in CONTROLS:
            a,b=c.split(c.act(h,u))
            self.assertEqual(a,x*c.actions[u])
            self.assertEqual(b,r)
        with self.assertRaises(ValueError):c.reverse(y,r+B[0,:])

    def test_rank_rejection(self):
        with self.assertRaises(ValueError):SeedAction(S.zeros(6,9),S.eye(6))

    def test_wrong_conflict(self):
        with self.assertRaises(ValueError):replay_conflict({'audit':{'decoder_fit':{'kind':'linear_conflict','consistent':True}}})

if __name__=='__main__':unittest.main()
