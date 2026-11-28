"""Original source marked-only versus residual-aware full-history consumer max-cut."""
import unittest
from scripts.x369_original_source_consumer_coarse_fine_maxcut import (
    source_consumer_maxcut,marked_source_chart,certificate)

def marks():
    return dict(original_seed_marking_injective=True,
      records={
        bytes((0,)):dict(mean=(0,0,0),signed_residual=(0,0,0)),
        bytes((1,)):dict(mean=(1,0,0),signed_residual=(0,0,0))})

def actual(h,u):
    x=h[0]
    return (('original_observation',x),('original_route',x%2)),bytes((
        x+2 if x<2 else x,))

class SourceCoarseFineGeometry(unittest.TestCase):
    def test_closed_four_state_coarse_refuted_fine_admitted(self):
        seeds=(bytes((0,)),bytes((1,)))
        r=source_consumer_maxcut(
            seeds,('tick',),actual,marks(),
            ('original_observation','original_route'),budget=4,
            export_realization=True)
        g=r['source_graph']
        self.assertEqual(g['graph_status'],'closed')
        self.assertTrue(g['finite_closed_graph_for_realization'])
        self.assertEqual(g['charts']['original_189_seed_mark_only']['status'],'refuted')
        self.assertEqual(g['charts']['entire_original_recurrent_bytes']['status'],'finite_closed')
        self.assertTrue(r['independently_replayed_negative_evidence'][
            'original_189_seed_mark_only']['original_observation']['replayed'])
        self.assertEqual(g['finite_realizations']['original_189_seed_mark_only'][
            'original_source_replay']['edges'],4)
        self.assertEqual(g['finite_realizations']['entire_original_recurrent_bytes'][
            'original_source_replay']['classes'],4)
        self.assertFalse(r['original_all_float_states_covered'])

    def test_budget_refuses_quotient_but_still_retains_original(self):
        r=source_consumer_maxcut(
            (bytes((0,)),bytes((1,))),('tick',),actual,marks(),
            ('original_observation',),budget=2,export_realization=True)
        self.assertEqual(r['source_graph']['graph_status'],'state_budget')
        self.assertEqual(r['source_graph']['finite_realizations'],{})
        self.assertFalse(r['source_graph']['finite_closed_graph_for_realization'])

    def test_actual_source_owned_views_only_when_supplied(self):
        seeds=(bytes((0,)),bytes((1,)))
        r=source_consumer_maxcut(
            seeds,('tick',),actual,marks(),('original_observation',),
            budget=4,stage13_chart=lambda h:('stage13',h[0]%2),
            pants_chart=lambda h:('pants',h[0]%2))
        self.assertEqual(r['native_typed_optional_source_charts_present'],
                         {'source_owned_stage13':True,
                          'source_owned_five_field_pants':True})
        self.assertIn('source_owned_stage13',r['source_graph']['charts'])
        self.assertIn('source_owned_five_field_pants',r['source_graph']['charts'])
        with self.assertRaises(ValueError):
            source_consumer_maxcut(
                seeds,('tick',),actual,marks(),('original_observation',),
                budget=4,stage13_chart='not-original-source-callback')

    def test_no_guessed_stage_or_pants_semantics(self):
        r=source_consumer_maxcut(
            (bytes((0,)),bytes((1,))),('tick',),actual,marks(),
            ('original_observation',),budget=4)
        self.assertEqual(r['native_typed_optional_source_charts_present'],
                         {'source_owned_stage13':False,
                          'source_owned_five_field_pants':False})
        self.assertNotIn('source_owned_stage13',r['source_graph']['charts'])
        self.assertNotIn('source_owned_five_field_pants',r['source_graph']['charts'])

    def test_static_fixture(self):
        self.assertTrue(certificate()['source_negative_independently_replayed'])
        self.assertFalse(certificate()['original_neural_checkpoint_executed'])

if __name__=='__main__':
    unittest.main()
