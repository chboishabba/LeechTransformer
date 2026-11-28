"""Six original source faces versus independent task group commutators."""
import unittest
from models.geometric_reflection_fibre import GROUP
from scripts.x369_original_marked_seed_fabric import marked_seed_records
from scripts.x369_original_source_face_defect_attribution import (
    original_source_face_decomposition,
    source_word_endpoint,certificate)

def seed(m,d):return bytes(v+1 for v in m+d)

MEANS=((0,0,0),(1,0,0))
TRIADS=((0,0,0),(0,1,0))

def setup():
    marks=marked_seed_records(MEANS,TRIADS,seed)
    h=seed(MEANS[1],TRIADS[1])
    axes={
      'x':(GROUP.index((1,0,0)),0),
      'y':(GROUP.index((0,1,0)),1),
      'z':(GROUP.index((0,0,1)),3)}
    return h,marks['records'][h],axes

class OriginalFaceDefectAttribution(unittest.TestCase):
    def test_source_identity_separates_known_task_noncommutation(self):
        h,mark,axes=setup()
        def original(h,u):return (('actual',h),),h
        r=original_source_face_decomposition(h,mark,axes,original,seed)
        self.assertEqual(r['face_count'],6)
        self.assertEqual(r['source_noncommuting_faces'],0)
        self.assertTrue(r['all_six_face_xor_decompositions_exact'])
        self.assertTrue(all(row['original_endpoints_agree'] for row in r['faces']))
        self.assertTrue(any(not row['independent_task_endpoints_agree']
                            for row in r['faces']))
        self.assertTrue(all(row['exact_face_defect_attribution_reopens']
                            for row in r['faces']))

    def test_source_known_task_step_only_has_task_commutator(self):
        h,mark,axes=setup()
        def step(original,control):
            m=tuple(x-1 for x in original[:3])
            d=tuple(x-1 for x in original[3:])
            row=source_word_endpoint({'mean':m,'signed_residual':d},
                seed,('x',),{'x':control})
            return (('actual',original),),row
        r=original_source_face_decomposition(h,mark,axes,step,seed)
        self.assertTrue(all(row['original_endpoints_agree']==
                            row['independent_task_endpoints_agree']
                            for row in r['faces']))
        self.assertTrue(all(row['exact_source_defect_difference_xor_hex']==
                            bytes(6).hex() for row in r['faces']))
        self.assertTrue(r['all_six_face_xor_decompositions_exact'])

    def test_learned_source_mismatch_reopens_both_paths(self):
        h,mark,axes=setup()
        def step(s,u):
            return (('actual',s),), bytes(((s[0]+1)%3,))+s[1:]
        r=original_source_face_decomposition(h,mark,axes,step,seed)
        self.assertEqual(len(r['faces']),6)
        for row in r['faces']:
            self.assertTrue(row['original_source_history_preserved'])
            self.assertEqual(len(bytes.fromhex(
                row['exact_source_face_xor_residual_hex'])),6)
            a=bytes.fromhex(row['original_left_defect']['actual_source_hex'])
            b=bytes.fromhex(row['original_right_defect']['actual_source_hex'])
            self.assertEqual(bytes(x^y for x,y in zip(a,b)).hex(),
                             row['exact_source_face_xor_residual_hex'])

    def test_exact_negative_and_control_validation(self):
        self.assertTrue(certificate()['exact_source_vs_task_face_residuals'])
        h,mark,axes=setup()
        with self.assertRaises(ValueError):
            original_source_face_decomposition(h,mark,
               dict(axes,x=axes['y']),
               lambda a,u:((),a),seed)

if __name__=='__main__':
    unittest.main()
