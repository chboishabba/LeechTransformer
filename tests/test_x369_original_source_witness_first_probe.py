"""Original-source witness-first proof: source-owned bytes and independently replayed negative."""
import unittest
from scripts.x369_original_source_witness_first_probe import (
    witness_first_fibre_probe,independently_replay_witness)

def source(h,u):
    i=h[0]
    if i in (0,1):
        return (('logits_0',i),('ids',0)),bytes((i+2,))
    return (('logits_0',i),('ids',i%2)),h

def chart(h):
    return ('original_mark',h[0]) if h[0] in (0,1) else ('unmarked',)

class SourceWitnessFirstProbe(unittest.TestCase):
    def test_genuine_original_source_consumer_separator_before_closure(self):
        seeds=(bytes((0,)),bytes((1,)))
        r=witness_first_fibre_probe(seeds,('tick',),source,chart,
                                    ('logits_0','ids'),seed_budget=2)
        self.assertEqual(r['status'],'refuted')
        self.assertEqual(r['witness']['kind'],'observation')
        self.assertEqual(r['witness']['consumer'],'logits_0')
        self.assertEqual(r['witness']['left']['word'],['tick'])
        self.assertTrue(r['independent_replay']['original_source_separator_replayed'])
        self.assertFalse(r['finite_closure_claim'])
        self.assertLess(r['original_source_edges'],10)
        self.assertTrue(independently_replay_witness(
            r['witness'],seeds,('tick',),source,chart,('logits_0','ids'))[
            'source_bytes_provenance_checked'])

    def test_same_observations_but_successor_chart_separates(self):
        seeds=(bytes((0,)),bytes((1,)))
        def step(h,u):
            if h[0] in (0,1):return (('obs',0),),bytes((h[0]+2,))
            return (('obs',0),),bytes((h[0]-2,))
        r=witness_first_fibre_probe(seeds,('tick',),step,chart,('obs',),seed_budget=2)
        self.assertEqual(r['status'],'refuted')
        self.assertEqual(r['witness']['kind'],'successor_chart')
        self.assertEqual(r['witness']['consumer'],'__transition__')

    def test_no_source_differences_remains_inconclusive(self):
        seeds=(bytes((0,)),bytes((1,)))
        def step(h,u):
            if h[0] in (0,1):return (('obs',0),),bytes((h[0]+2,))
            return (('obs',0),),bytes((2,))
        r=witness_first_fibre_probe(seeds,('tick',),step,chart,('obs',),seed_budget=2)
        self.assertEqual(r['status'],'sample_inconclusive')
        self.assertIsNone(r['witness'])
        self.assertFalse(r['all_original_float_histories_covered'])

    def test_tampered_witness_and_malformed_source_fail_closed(self):
        seeds=(bytes((0,)),bytes((1,)))
        r=witness_first_fibre_probe(seeds,('tick',),source,chart,
                                    ('logits_0','ids'),seed_budget=2)
        witness=dict(r['witness'])
        witness['left_value_sha256']='0'*64
        with self.assertRaises(ValueError):
            independently_replay_witness(
                witness,seeds,('tick',),source,chart,('logits_0','ids'))
        with self.assertRaises(ValueError):
            witness_first_fibre_probe(seeds,('tick','tick'),source,chart,('ids',))
        with self.assertRaises(ValueError):
            witness_first_fibre_probe(seeds,('tick',),source,chart,
                                      ('missing',),seed_budget=2)

if __name__=='__main__':
    unittest.main()
