"""Exact source port BIDI and independent-index refusal (source-free fixtures)."""
import struct
import unittest
from scripts.x369_stage13_pants_orbit_source_ports import (
    attach,reopen_ports,source_edge,status,model_ports)

CFG=dict(experts=4,width=12,dtype='float32')
A=struct.pack('<96f',*([0.,-0.,1.,2.,-3.,4.]*16))
B=struct.pack('<96f',*([1.,-0.,0.,-2.,3.,4.]*16))
SLOTS=(1,2,3,4,5,6,7,8,9)
NINE=(1,0,-1)*3


def stage_reader(h):
    from scripts.x369_hypervoxel_stage13_pants_same_object import stitch_stage13_from_native
    return dict(native9=NINE,source13=stitch_stage13_from_native(NINE,(1,0,-1,1),SLOTS),
        native_slots=SLOTS,indices=dict(history=0 if h==A else 1,physical_time='t',
        refinement=2,frobenius='f',version='v'))


def pants_reader(h):
    return dict(proposition='p',capacity=3,phase='phase',wave=('wave',),open=h==A)


def transport(p,u):
    return dict(history=1,physical_time='t',refinement=2,frobenius='f',version='v')


class Stage13OrbitPorts(unittest.TestCase):
    def test_existing_geometry_and_exact_original_step(self):
        a=attach(A,CFG,stage_reader,pants_reader)
        p,boundary=reopen_ports(a)
        self.assertEqual(p.source_bytes,A)
        self.assertEqual(p.interaction+p.appraisal_a+p.appraisal_b,NINE)
        self.assertTrue(boundary.open)
        def step(h,u):
            self.assertEqual((h,u),(A,(8,0)))
            return ('raw consumer',),B
        obs,b,r=source_edge(a,(8,0),step,stage_reader,pants_reader,transport)
        self.assertEqual(obs,('raw consumer',))
        self.assertEqual(reopen_ports(b)[0].source_bytes,B)
        self.assertTrue(r['source_index_transport_matches'])
        self.assertFalse(r['boundary_fields_equal']['open'])
        self.assertFalse(r['all_future_reduced_chart_sufficiency_certified'])

    def test_missing_or_conflicting_ports_are_not_synthesized(self):
        missing=status(*model_ports(object()))
        self.assertEqual(len(missing['missing']),3)
        self.assertFalse(missing['actual_source_semantics_certified'])
        with self.assertRaises(ValueError):attach(A,CFG,None,pants_reader)
        def wrong(h):
            r=stage_reader(h);r['native9']=(0,)*9;return r
        with self.assertRaises(ValueError):attach(A,CFG,wrong,pants_reader)
        a=attach(A,CFG,stage_reader,pants_reader)
        with self.assertRaises(ValueError):
            source_edge(a,(8,0),lambda h,u:((),B),stage_reader,pants_reader,
                        lambda p,u:dict(history=1))


if __name__=='__main__':
    unittest.main()
