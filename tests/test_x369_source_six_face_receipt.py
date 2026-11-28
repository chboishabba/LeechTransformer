"""Exact synthetic control-face negatives for original X369 source contract."""
import unittest
from scripts.x369_source_six_face_receipt import FACE_PATHS,cube_faces,certificate,retain_byte_face,reopen_byte_face

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

    def test_source_byte_face_xor_residual_exact_and_signed_zero(self):
        a=bytes.fromhex('00000000 00000080 01000000 0000803f')
        b=bytes.fromhex('00000080 00000000 01000000 0000803f')
        packet=retain_byte_face(a,b)
        self.assertEqual(reopen_byte_face(packet),(a,b))
        self.assertEqual(packet['source_state_bytes'],len(a))
        self.assertEqual(packet['residual_nonzero_byte_count'],2)
        self.assertNotEqual(packet['left_hex'],b.hex())
        with self.assertRaises(ValueError):
            retain_byte_face(a,b[:3])
        with self.assertRaises(ValueError):
            reopen_byte_face(dict(packet,source_state_bytes=1))

    def test_original_byte_source_has_six_reopenable_faces(self):
        def byte_step(h,u):
            raw=bytes((v+u)%256 for v in h)
            return (('readout',h),),raw
        r=cube_faces(bytes((0,128,255)),{'x':1,'y':2,'z':3},byte_step,probe=1)
        self.assertTrue(r['all_six_endpoints_commute'])
        for face in r['faces']:
            a,b=reopen_byte_face(face['retained_byte_face'])
            self.assertEqual(a,b)
            self.assertTrue(face['source_byte_bidi_exact'])

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
