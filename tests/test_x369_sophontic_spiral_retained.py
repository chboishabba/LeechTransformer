"""Retained source reasoning plus real task checkpoint diagnostics."""
import struct
import unittest
import torch
from scripts.x369_sophontic_spiral_retained import (
    Spiral,Trialectic,geometric_move,pack_reasoning,reopen_reasoning,
    reason_step,spiral_certificate)
from scripts.x369_whole_source_indexed_transport import pack_joint
from tests.test_x369_whole_source_indexed_transport import context_fixture
from scripts.x369_grokking_geometry_trajectory import measure,spectral
from scripts.train_x369_mod3_codec_gpu import Learner
from scripts.audit_x369_typed_mod3_prediction_consumer import TASK_PAIRS
from scripts.run_x369_mod3_maxcut_gpu import exact_fourier_law_logits


class SophonticSpiral(unittest.TestCase):
    def test_complete_spiral_geometry(self):
        self.assertEqual(spiral_certificate()['signed_sheet_sense_cases'],39366)

    def test_independent_reasoning_and_original_neural_projection(self):
        h=struct.pack('<96f',*([0.,-0.,1.,2.,-3.,4.]*16))
        cfg=dict(experts=4,width=12,dtype='float32')
        c=context_fixture()
        indices=dict(history=0,physical_time='t',refinement=1,frobenius='F',version='v')
        j=pack_joint(h,cfg,c,indices)
        p=pack_reasoning(j,Spiral((-1,0,1)*3,0,1),
            Trialectic((1,1,1),'reciprocal'),'source:evidence:1')
        def step(b,u):
            self.assertEqual(b,h)
            return ('original consumer',),b
        def advance(i,c,n,u):return dict(i,history=i['history']+1)
        def semantic(old,new,u):
            return Trialectic(old.trialectic.edges,'coercive'),'source:evidence:2'
        obs,q=reason_step(p,(8,0),'reflection',step,lambda c,u:c,advance,semantic)
        self.assertEqual(obs,('original consumer',))
        reopened,s,t,e=reopen_reasoning(q)
        self.assertEqual(reopened[0],h)
        self.assertEqual((s.height,s.sense),(1,-1))
        self.assertEqual(t.edges,p.trialectic.edges)
        self.assertNotEqual(t.face,p.trialectic.face)
        self.assertEqual(e,'source:evidence:2')
        with self.assertRaises(ValueError):
            reason_step(p,(8,0),'hold',step,lambda c,u:c,advance,None)

    def test_real_task_character_and_source_measurement(self):
        pairs=torch.tensor(TASK_PAIRS)
        law=exact_fourier_law_logits(pairs,torch.float64)
        r=spectral(law)
        self.assertAlmostEqual(r['character_pair_fraction'][2],1.0,places=12)
        self.assertLess(r['parseval_absolute_residual'],1e-9)
        torch.manual_seed(41)
        m=Learner(12,4).double().eval()
        with torch.no_grad():
            raw=m(pairs)
            got,receipt=measure(m,pairs,None,True)
        self.assertTrue(torch.equal(raw,got))
        self.assertEqual(len(receipt['actual_router_steps']),2)
        self.assertFalse(receipt['grokking_mechanism_certified'])
        for row in receipt['actual_router_steps']:
            self.assertEqual(sum(row['expert_occupancy']),18)
            self.assertLess(row['history']['parseval_absolute_residual'],1e-7)


if __name__=='__main__':unittest.main()
