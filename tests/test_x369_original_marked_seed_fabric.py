"""Original marked 189-source-seed admission, alias and successor discipline."""
import unittest
from scripts.x369_original_marked_seed_fabric import (
    marked_seed_records,mark_original_source_history,
    seed_marker_future_audit,decode_canonical_original_seed_bytes,certificate,
    signed_original_task_action,marked_source_group_certificate)

MEANS=((0,0,0),(-1,0,0),(1,0,0))
DELTAS=((0,0,0),(-1,0,0),(1,0,0))

def owner(m,d):
    return bytes((m[0]+1,d[0]+1))

class OriginalMarkedSeedGeometry(unittest.TestCase):
    def test_source_owned_exact_seed_bidi(self):
        mark=marked_seed_records(MEANS,DELTAS,owner)
        self.assertTrue(mark['original_seed_marking_injective'])
        self.assertEqual(mark['distinct_original_byte_seeds'],9)
        self.assertEqual(mark['attempted_seed_count'],9)
        for m in MEANS:
            for d in DELTAS:
                state=owner(m,d)
                receipt=mark_original_source_history(state,mark)
                self.assertEqual(receipt['status'],'marked_original_source_seed')
                self.assertEqual(receipt['native_input9'],m+d+(0,0,0))
                self.assertEqual(receipt['original_source_length'],2)
                self.assertFalse(receipt['stage13_source_semantics_inferred'])
        unmarked=mark_original_source_history(bytes((4,1)),mark)
        self.assertEqual(unmarked['status'],'unmarked_source_history')
        self.assertFalse(unmarked['original_native_nine_recovered'])

    def test_original_signed_seed_float32_and_float64_byte_bidi(self):
        import struct
        for dtype,scalar in (('float32','f'),('float64','d')):
            for E,W in ((4,12),(2,3),(8,9)):
                for m in MEANS:
                    for d in DELTAS:
                        values=[0.0]*(E*2*W)
                        for i in range(3):
                            values[i]=float(m[i]+d[i])
                            values[W+i]=float(m[i]-d[i])
                        h=struct.pack('='+str(len(values))+scalar,*values)
                        decoded=decode_canonical_original_seed_bytes(
                            h,experts=E,width=W,dtype=dtype)
                        self.assertEqual(decoded['native_input9'],m+d+(0,0,0))
                        self.assertEqual(decoded['mean'],m)
                        self.assertEqual(decoded['signed_residual'],d)
                        altered=bytearray(h)
                        altered[-1]=altered[-1]^128
                        self.assertIsNone(decode_canonical_original_seed_bytes(
                            bytes(altered),experts=E,width=W,dtype=dtype))

    def test_source_signed_seed_noncanonical_neural_history_refused(self):
        import struct
        h=struct.pack('=24f',*([0.0]*24))
        self.assertEqual(
            decode_canonical_original_seed_bytes(h,experts=4,width=3,dtype='float32')['mean'],
            (0,0,0))
        with self.assertRaises(ValueError):
            decode_canonical_original_seed_bytes(h,experts=4,width=12,dtype='float32')
        with self.assertRaises(ValueError):
            decode_canonical_original_seed_bytes(h,experts=1,width=3,dtype='float32')

    def test_complete_original_189_by_12_group_action_and_inverse(self):
        from itertools import product
        means=((0,0,0),)+tuple(tuple(s*int(i==j) for i in range(3))
            for j in range(3) for s in (-1,1))
        triads=tuple(product((-1,0,1),repeat=3))
        def original_fixture(m,d):
            return bytes(v+1 for v in m+d)
        marked=marked_seed_records(means,triads,original_fixture)
        self.assertEqual(marked['distinct_original_byte_seeds'],189)
        receipt=marked_source_group_certificate(marked)
        self.assertEqual(receipt['marked_states'],189)
        self.assertEqual(receipt['group_elements'],12)
        self.assertEqual(receipt['closed_action_edges'],2268)
        self.assertEqual(receipt['group_compositions_checked'],27216)
        self.assertEqual(sum(receipt['orbit_size_distribution']),189)
        self.assertFalse(receipt['original_model_equivariance_proved'])
        self.assertFalse(receipt['independent_task_action_not_neural_transition'] is False)
        with self.assertRaises(ValueError):
            signed_original_task_action((3,0,0),(0,0,0),(0,0,0))

    def test_source_alias_is_explicit_counterexample(self):
        a=marked_seed_records(MEANS,DELTAS,lambda m,d:b'unchanged')
        self.assertFalse(a['original_seed_marking_injective'])
        self.assertEqual(a['distinct_original_byte_seeds'],1)
        self.assertEqual(len(a['original_source_byte_aliases']),8)
        with self.assertRaises(ValueError):
            seed_marker_future_audit(a,('u',),
                lambda h,u:((('obs',0),),h))

    def test_unmarked_successor_keeps_original_bytes_without_fake_label(self):
        m=marked_seed_records(MEANS,DELTAS,owner)
        def step(h,u):
            return (('actual_bytes',h),),bytes((3,h[1]))
        r=seed_marker_future_audit(m,('u','v'),step,max_seeds=9)
        self.assertEqual(r['source_steps'],18)
        self.assertEqual(r['successors_unmarked'],18)
        self.assertEqual(r['successors_also_marked'],0)
        self.assertTrue(r['first_unmarked_successor']['original_successor_bytes_retained'])
        self.assertFalse(r['original_all_futures_factorized'])

    def test_exact_fixture_certificate_and_malformed_input(self):
        self.assertTrue(certificate()['source_byte_initial_label_bidi'])
        with self.assertRaises(ValueError):
            marked_seed_records(((2,0,0),),DELTAS,owner)
        with self.assertRaises(ValueError):
            marked_seed_records(MEANS,DELTAS,lambda m,d:(0,1))
        m=marked_seed_records(MEANS,DELTAS,owner)
        with self.assertRaises(ValueError):
            mark_original_source_history((1,2),m)
        with self.assertRaises(ValueError):
            seed_marker_future_audit(m,('u',),
                lambda h,u:((('obs',0),),h),max_seeds=10)

if __name__=='__main__':
    unittest.main()
