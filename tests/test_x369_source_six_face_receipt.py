"""Exact synthetic control-face negatives for original X369 source contract."""
import unittest
from scripts.x369_source_six_face_receipt import FACE_PATHS,cube_faces,certificate

class SixSourceFaces(unittest.TestCase):
    def test_six_actual_faces_and_commuting_additive_source(self):
        self.assertEqual(len(FACE_PATHS),6)
        def step(s,u):return (('value',s),),s+u
        r=cube_faces(0,{'x':1,'y':2,'z':3},step,probe=1)
        self.assertTrue(r['all_six_endpoints_commute'])
        self.assertEqual(tuple(f['face'] for f in r['faces']),
                         ('xy0','xy1','xz0','xz1','yz0','yz1'))
        self.assertTrue(all(all(f['consumer_equal'].values()) for f in r['faces']))
        self.assertFalse(r['hyperfabric_9d_physical_dynamics_certified'])

    def test_noncommuting_source_reports_actual_endpoint_and_consumer(self):
        def step(s,u):
            obs=(('state',s),)
            return obs,(s+1 if u=='x' else 2*s if u=='y' else s-3)
        r=cube_faces(0,{'x':'x','y':'y','z':'z'},step,probe='x')
        self.assertFalse(r['all_six_endpoints_commute'])
        bad=[f for f in r['faces'] if not f['endpoint_bytes_equal']]
        self.assertTrue(bad)
        self.assertTrue(any(not f['consumer_equal']['state'] for f in bad))
        self.assertTrue(all(len(f['left_endpoint_sha256'])==64 for f in bad))

    def test_refuses_repeated_or_missing_source_axes(self):
        with self.assertRaises(ValueError):
            cube_faces(0,{'x':1,'y':1,'z':2},lambda s,u:((('s',s),),s))
        with self.assertRaises(ValueError):
            cube_faces(0,{'x':1,'y':2},lambda s,u:((('s',s),),s))

    def test_certificate_has_no_native_claim(self):
        r=certificate()
        self.assertEqual(r['faces'],6)
        self.assertTrue(r['negative_affine'])
        self.assertFalse(r['native_executed'])

if __name__=='__main__':
    unittest.main()
