"""Regression: existing DASHI original five-field pants interface on source bytes."""
import unittest
from scripts.x369_full_fabric_pants_source_weld import (
    make_seam,reopen_seam,weld_original_source_faces,certificate,FIELDS)


def signature(**changes):
    ans=dict(proposition='selected source',capacity=2,phase='p0',
             wave=('real','imag'),open=True)
    ans.update(changes)
    return ans


class SourcePantsMaxCut(unittest.TestCase):
    def test_exact_bidi_with_distinct_histories_and_equal_seam(self):
        a=bytes.fromhex('00000080 00000000')  # -0 and +0
        b=bytes.fromhex('00000000 00000080')
        r=make_seam(a,b,signature(),signature(),provenance=('a','b'))
        self.assertEqual(r['status'],'admitted_selected_five_field_seam')
        self.assertEqual(len(r['before_incidence']),2)
        self.assertEqual(len(r['after_incidence']),2)
        self.assertEqual(r['after_edges'],('merged_edge',))
        self.assertEqual(r['merged_provenance'],('a','b'))
        self.assertEqual(reopen_seam(r),(a,b))
        self.assertTrue(all(r['five_field_match'].values()))
        self.assertFalse(r['source_reader_future_descent_proved'])

    def test_each_original_interface_coordinate_is_necessary(self):
        base=signature()
        variants=dict(proposition='other',capacity=3,phase='p1',
                      wave=('other','wave'),open=False)
        for field,changed in variants.items():
            with self.subTest(field=field):
                out=make_seam(b'a',b'b',base,
                              signature(**{field:changed}),
                              provenance=('left','right'))
                self.assertEqual(out['status'],'refused_unmatched_original_boundary')
                self.assertFalse(out['five_field_match'][field])
                self.assertIsNone(out['after_edges'])
                self.assertEqual(reopen_seam(out),(b'a',b'b'))

    def test_matches_pants_boundary_but_not_original_consumer(self):
        def step(state,control):
            return (('readout',state),),bytes((x+control)%256 for x in state)
        # All pairs of added byte steps commute; six faces match, even
        # though intermediate path observations are order-sensitive.
        out=weld_original_source_faces(b'\x00\x80',
           {'x':1,'y':2,'z':3},step,
           lambda source,original:signature(),probe=1)
        self.assertEqual(out['face_count'],6)
        self.assertEqual(out['admitted'],6)
        self.assertEqual(out['refused'],0)
        for row in out['rows']:
            self.assertTrue(all(row['original_consumers_equal'].values()))
            self.assertTrue(row['bidirectional_reopen_exact'])
        self.assertFalse(out['universal_original_model_equivariance'])

    def test_neural_observation_not_inferred_from_matching_pants_fields(self):
        a,b=b'\x00',b'\x01'
        result=make_seam(a,b,signature(),signature(),
                         provenance=('source/path1','source/path2'))
        self.assertEqual(result['status'],'admitted_selected_five_field_seam')
        self.assertNotEqual(a,b)
        self.assertFalse(result['source_reader_future_descent_proved'])

    def test_original_noncommuting_source_refused_where_phase_differs(self):
        def step(state,control):
            v=state[0]
            if control=='x':v=(v+1)%256
            elif control=='y':v=(2*v)%256
            elif control=='z':v=(v-3)%256
            return (('actual',v),),bytes((v,))
        out=weld_original_source_faces(bytes((1,)),
            dict(x='x',y='y',z='z'),step,
            lambda source,original:signature(phase=source[0]),probe='x')
        self.assertEqual(out['face_count'],6)
        self.assertGreater(out['refused'],0)
        for row in out['rows']:
            self.assertEqual(len(reopen_seam(row)),2)
            if row['status']=='refused_unmatched_original_boundary':
                self.assertFalse(row['five_field_match']['phase'])

    def test_rejects_guessed_or_missing_original_observables(self):
        for missing in FIELDS:
            malformed=signature()
            del malformed[missing]
            with self.assertRaises(ValueError):
                make_seam(b'a',b'b',malformed,signature(),
                          provenance=('left','right'))
        with self.assertRaises(ValueError):
            make_seam(b'a',b'b',signature(capacity=-1),signature(),
                      provenance=('left','right'))
        with self.assertRaises(ValueError):
            make_seam(b'a',b'b',signature(open=1),signature(),
                      provenance=('left','right'))
        with self.assertRaises(ValueError):
            make_seam(b'a',b'b',signature(),signature(),
                      provenance=('only one',))

    def test_static_certificate_does_not_claim_native(self):
        receipt=certificate()
        self.assertTrue(receipt['selected_seam_admitted'])
        self.assertTrue(receipt['phase_mismatch_refused'])
        self.assertFalse(receipt['source_native_executed'])


if __name__=='__main__':
    unittest.main()
