"""Exact constructive X369 contract regressions; no GPU or training.
Original Leech-LILA: Anatolii Kornienko; DASHI extension.
"""
import unittest
from itertools import product
from fractions import Fraction as Q


class ConstructiveMaxCut(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from scripts.audit_x369_joint_action import joint_records
        from scripts.x369_learned_generator_weld import fit_generators,certify
        cls.matrices=fit_generators(joint_records())
        cls.certificate=certify(cls.matrices)

    def test_identified_generators_not_fitted_composition(self):
        self.assertEqual(self.certificate['group_products'],144)
        self.assertEqual(self.certificate['literals'],10)
        from scripts.x369_learned_generator_weld import certify
        altered=dict(self.matrices);v=[list(r) for r in altered['a',1]]
        v[0][0]=9;altered['a',1]=tuple(map(tuple,v))
        with self.assertRaises(ValueError):certify(altered)

    def test_chronological_inverse_zero_extent(self):
        from scripts.x369_ordered_residual_codec import encode,ALPHABET
        for n in range(6):
            for word in product(ALPHABET,repeat=n):
                h=encode(word)
                self.assertEqual(h.decode(),word)
                self.assertEqual(h.extent,n)
                self.assertLessEqual(h.value.bit_length(),2*n)
        self.assertNotEqual(encode(()),encode(('a',)))

    def test_indexed_consumer_and_latch_separation(self):
        from scripts.x369_constructive_consumer import Consumer
        from scripts.audit_x369_ordered_signed_task import endpoint
        seed=(1,0,-1,0,-1,1)
        sem=Consumer(self.matrices,'semantic')
        mem=Consumer(self.matrices,'first')
        hist=Consumer(self.matrices,'history')
        for suffix in ('','a','b','j','n','jn','abjn'):
            a=tuple((c,1 if c in 'ajn' else 2) for c in 'ab'+suffix)
            b=tuple((c,1 if c in 'ajn' else 2) for c in 'ba'+suffix)
            x,y=sem.run(seed,a),sem.run(seed,b)
            self.assertEqual(x,y)
            self.assertEqual(mem.run(seed,a).residual,'a')
            self.assertEqual(mem.run(seed,b).residual,'b')
            self.assertEqual(hist.reopen(hist.run(seed,a)),a)
            self.assertEqual(hist.reopen(hist.run(seed,b)),b)

    def test_blocks_inverse_composition_and_discard_refusal(self):
        from scripts.x369_costed_constructive import literal_blocks,apply,inverse,power,reopen,Block
        blocks=literal_blocks(self.matrices)
        keys=tuple(sorted(blocks))
        z=(2,1,0,-1,0,1)
        for a,b in product(blocks.values(),repeat=2):
            c=a.then(b)
            self.assertEqual(apply(c.action,z),apply(b.action,apply(a.action,z)))
            self.assertEqual(apply(inverse(c.action),apply(c.action,z)),z)
            self.assertEqual(reopen(c,keys),(a.first,b.first))
        x=blocks['a',1].then(blocks['j',1])
        for n in range(80):
            v,ledger=power(x,n,history=True)
            self.assertEqual(reopen(v,keys),(('a',1),('j',1))*n)
            self.assertEqual(v.extent,2*n)
            t=z
            for i in range(n):t=apply(x.action,t)
            self.assertEqual(apply(v.action,z),t)
            self.assertGreaterEqual(ledger['action_compositions'],0)
        lost,_=power(x,13,history=False)
        with self.assertRaises(ValueError):reopen(lost,keys)
        with self.assertRaises(ValueError):lost.then(x)
        with self.assertRaises(ValueError):Block(extent=1,first=('a',1),value=1,history_retained=False)

    def test_minimal_future_consumer_and_negative_budget(self):
        from scripts.x369_consumer_automata import compile_observers,execute
        controls=('a','b','j','n')
        def step(r,u):
            return r,(r if r is not None else u if u in 'ab' else None)
        receipt,metrics=compile_observers((None,),controls,step,lambda r:r)
        self.assertEqual(metrics['classes'],3)
        self.assertEqual(metrics['state_id_bits'],2)
        for suffix in product(controls,repeat=3):
            a,_=execute(receipt,0,('a','b')+suffix)
            b,_=execute(receipt,0,('b','a')+suffix)
            self.assertNotEqual(a,b)
        with self.assertRaises(ValueError):
            compile_observers((0,),('a',),lambda z,u:(0,z+1),lambda z:0,budget=3)

    def test_fixed_projection_and_typed_margin(self):
        from scripts.x369_controlled_interval import (
            fixed_orbit_projection,typed_decode_box,typed_logits_box,unique_margin,point)
        from scripts.audit_x369_ordered_signed_task import TRIADS,gram
        h=(((point(2),point(0),point(0)),(point(0),point(0),point(0))),
           ((point(0),point(0),point(0)),(point(0),point(0),point(0))))
        ident=fixed_orbit_projection(h,((0,0,0),))
        self.assertEqual(ident,h)
        zero=fixed_orbit_projection(h,((0,0,0),(0,0,1)))
        self.assertTrue(all(x==point(0) for e in zero for b in e for x in b))
        m=(1,0,0);d=(1,-1,0)
        a=tuple(map(point,tuple(x+y for x,y in zip(m,d))+(0,)*6))
        b=tuple(map(point,tuple(x-y for x,y in zip(m,d))+(0,)*6))
        z=((a,b),(tuple(map(point,(0,)*9)),tuple(map(point,(0,)*9))))
        mm,dd=typed_decode_box(z)
        self.assertEqual(tuple(v.lo for v in mm),m)
        self.assertEqual(tuple(v.lo for v in dd),d)
        L=typed_logits_box(mm,dd)
        self.assertEqual(unique_margin(L[0])[0],TRIADS.index(m))
        self.assertEqual(unique_margin(L[2])[0],TRIADS.index(d))


if __name__=='__main__':
    unittest.main()
