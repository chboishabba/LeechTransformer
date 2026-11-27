"""Exact algebra fixtures; native checkpoint acceptance is separate."""
import unittest
from fractions import Fraction as Q
from itertools import product
from types import SimpleNamespace
from scripts.x369_exact_operator_transport import Historical, CONTROLS
from scripts.x369_simultaneous_weld import literal_matrix, apply_matrix
from scripts.x369_indexed_residual_closure import join, split, add, sub

class TransportTest(unittest.TestCase):
    def setUp(self):
        self.owner=Historical.__new__(Historical)
        self.owner.old=SimpleNamespace(width=12,experts=4)
        self.owner.matrices={u:literal_matrix(u) for u in CONTROLS}
        self.owner.values=lambda h:h
        # Non-equivariant deterministic source fixture, not a fitted model.
        self.owner.step=lambda h,u:((),tuple(x+Q((i+u[0]+u[1])%3-1,4) for i,x in enumerate(h)))

    def test_all_signed_seeds(self):
        o=self.owner
        h=tuple(Q(i%7-3,8) for i in range(96))
        for z in product((-1,0,1),repeat=6):
            p=o.packet(h,z)
            self.assertEqual(join((add(p[1],p[2]),p[3]),o.section),h)
            self.assertEqual(o.decode(o.section(z)),z)
            self.assertEqual(o.decode(p[3]),(0,)*6)

    def test_ordered_cocycle(self):
        o=self.owner
        z=(1,0,-1,0,1,-1)
        h=tuple(Q(i%5-2,4) for i in range(96))
        p=o.packet(h,z)
        for u,v in product(CONTROLS,repeat=2):
            _,a=o.advance(p,u)
            _,b=o.advance(a,v)
            self.assertEqual(b[1],o.action(v,o.action(u,z)))
            self.assertEqual(join((add(b[1],b[2]),b[3]),o.section),b[0])

    def test_invalid_packet(self):
        o=self.owner
        h=(Q(0),)*96
        p=o.packet(h,(0,)*6)
        with self.assertRaises(ValueError):
            o.advance((p[0],p[1],(1,)*6,p[3]),CONTROLS[0])

if __name__=='__main__':
    unittest.main()
