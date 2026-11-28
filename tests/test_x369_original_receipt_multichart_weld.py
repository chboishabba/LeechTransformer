"""Regression for the real stored native source, and partial-graph negatives."""
import json
import unittest
from pathlib import Path
from scripts.x369_original_receipt_multichart_weld import (
    certify, analyse_chart, extract_edges, chart_values, FIELDS)

PATH = Path(__file__).resolve().parents[1] / 'docs/receipts/x369_original_native_minimal_consumer_20261128.json'


class RealReceipt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = PATH.read_bytes()
        cls.cert = certify(cls.raw)

    def test_whole_receipt_graph_and_bound(self):
        c = self.cert
        self.assertEqual((c['observed_states'], c['observed_edges']), (48,44))
        self.assertEqual((c['exact_payload_edges'],c['digest_only_edges']),(16,28))
        mark = c['charts']['source_owned_original_189_initial_mark_only']
        self.assertGreaterEqual(mark['at_least_residual_labels_in_one_fibre'],4)
        self.assertGreaterEqual(mark['at_least_fixed_width_residual_bits'],2)
        self.assertEqual(mark['status'],'refuted_on_observed_source')
        self.assertEqual(c['charts']['full_original_history_bytes']['witnessed_incompatible_pairs'],0)
        for row in c['charts'].values():
            self.assertFalse(row['full_source_sufficiency_certified'])
            n = len(row['residual_clique'])
            self.assertEqual(len(row['clique_witnesses']),n*(n-1)//2)
        self.assertFalse(c['original_gpu_model_newly_replayed'])
        self.assertFalse(c['source_graph_finite_closure_certified'])
        json.dumps(c,sort_keys=True)

    def test_pinned_receipt_changed_refused(self):
        with self.assertRaises(ValueError):
            certify(self.raw+b' ')

    def test_tampered_later_word_payload_refused_even_after_rehash(self):
        receipt = json.loads(self.raw)
        rows = receipt['stages']['indexed_fibre']['original_arbitrary_word_source_residual']['exact_source_word_transport_receipts']
        rows[3]['receipt']['original_step_receipts'][3]['original_source_observations']['logits_0'] = 'forged'
        with self.assertRaises(ValueError):
            extract_edges(receipt)

    def test_tampered_signed_defect_endpoint_refused(self):
        receipt = json.loads(self.raw)
        rows = receipt['stages']['indexed_fibre']['original_signed_source_defect']['full_original_byte_defect_receipts']
        rows[-1]['second_defect']['actual_source_hex'] = '00'*384
        with self.assertRaises(ValueError):
            extract_edges(receipt)

    def test_exact_chart_owner_and_size_guard(self):
        cfg = dict(experts=4,width=12,dtype='float32')
        v = chart_values(bytes(384),cfg)
        self.assertEqual(v['constructed_semantic_six'],(0,)*6)
        self.assertEqual(v['semantic_plus_cap_profile'][0],(0,)*6)
        with self.assertRaises(ValueError):
            chart_values(bytes(383),cfg)


def edge(n, obs=0, *, full=True):
    return dict(successor=n,
                payload={f:obs for f in FIELDS} if full else None,
                hashes={f:str(obs) for f in FIELDS})


class PartialGraph(unittest.TestCase):
    def test_delayed_consumer_separator(self):
        codes = {'a':0,'b':0,'c':1,'d':1,'e':2}
        edges = {('a',(0,0)):edge('c'),('b',(0,0)):edge('d'),
                 ('c',(1,0)):edge('e',1),('d',(1,0)):edge('e',2)}
        r = analyse_chart(codes,edges)
        w = next(x for x in r['all_pair_witnesses'] if x['left']=='a')
        self.assertEqual(w['word'],[(0,0),(1,0)])
        self.assertEqual(w['kind'],'consumer')

    def test_successor_chart_separator_and_digest_only(self):
        c = {'a':0,'b':0,'c':1,'d':2}
        e = {('a',(0,0)):edge('c'),('b',(0,0)):edge('d')}
        self.assertEqual(analyse_chart(c,e)['all_pair_witnesses'][0]['kind'],'retained_chart')
        e[('a',(0,0))] = edge('c',1,full=False)
        e[('b',(0,0))] = edge('d',2,full=False)
        self.assertEqual(analyse_chart(c,e)['all_pair_witnesses'][0]['evidence'],'distinct_stored_digest')

    def test_missing_controls_are_not_equal_outputs(self):
        c = {'a':0,'b':0}
        e = {('a',(0,0)):edge('a'),('b',(1,0)):edge('b')}
        r = analyse_chart(c,e)
        self.assertEqual(r['status'],'sample_inconclusive')
        self.assertEqual(r['pairs_without_common_recorded_control'],1)
        self.assertFalse(r['full_source_sufficiency_certified'])


if __name__ == '__main__':
    unittest.main()
