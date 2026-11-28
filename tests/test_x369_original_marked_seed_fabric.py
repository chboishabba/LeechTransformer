"""Original marked 189-source-seed admission, alias and successor discipline."""
import unittest
from scripts.x369_original_marked_seed_fabric import (
    marked_seed_records,mark_original_source_history,
    seed_marker_future_audit,certificate)

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
