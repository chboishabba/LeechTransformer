"""Regression gates for original X369 byte-consumer fibre factorization.

No GPU access or Agda kernel is needed to establish these negative witnesses.
"""
import unittest
from scripts.x369_indexed_source_fibre_audit import fibre_audit,certificate


class IndexedSourceFibreMaxCut(unittest.TestCase):
    def test_synthetic_reference_certificate(self):
        c=certificate()
        self.assertEqual(c['synthetic_closed'],4)
        self.assertTrue(c['negative_consumer'])
        self.assertTrue(c['negative_transition'])
        self.assertTrue(c['budget_refusal'])
        self.assertFalse(c['original_native_tested'])

    def test_same_spatial_chart_different_independent_consumer(self):
        # Interaction/semantic slot identical; appraisal- or residual-owned
        # observation different. A six-coordinate chart cannot be universal.
        states=((0,0),(0,1))
        def step(h,u):
            return (('semantic',h[0]),('appraisal',h[1])),h
        audit=fibre_audit(states,('stay',),step,
            {'semantic':lambda h:h[0], 'full':lambda h:h},
            ('semantic','appraisal'),budget=2)
        self.assertEqual(audit['graph_status'],'closed')
        self.assertEqual(audit['charts']['semantic']['status'],'refuted')
        self.assertIn('appraisal',audit['charts']['semantic']['witnessed_failures'])
        self.assertEqual(audit['charts']['full']['status'],'finite_closed')
        self.assertFalse(audit['charts']['full']['unrestricted_float_source_certified'])

    def test_zero_fibre_and_version_distinctions(self):
        # Distinct visible-zero interpretations and independently varying
        # consumer versions cannot be identified merely by the value 0.
        states=((0,'forward',0),(0,'reverse',0),(0,'forward',1))
        def step(h,u):
            return (('orientation',h[1]),('version',h[2])),h
        result=fibre_audit(states,('stay',),step,
            {'visible_zero':lambda h:h[0],
             'zero_and_orientation':lambda h:h[:2],
             'full':lambda h:h},
            ('orientation','version'),budget=3)
        self.assertEqual(result['charts']['visible_zero']['status'],'refuted')
        self.assertEqual(result['charts']['zero_and_orientation']['status'],'refuted')
        self.assertIn('version',
          result['charts']['zero_and_orientation']['witnessed_failures'])
        self.assertEqual(result['charts']['full']['status'],'finite_closed')

    def test_incomplete_exploration_is_not_universal(self):
        def step(h,u):
            return (('output',0),),h+1
        result=fibre_audit((0,),('tick',),step,
            {'constant':lambda h:0},('output',),budget=2)
        self.assertEqual(result['graph_status'],'state_budget')
        self.assertEqual(result['charts']['constant']['status'],'incomplete')
        self.assertFalse(result['charts']['constant']['unrestricted_float_source_certified'])

    def test_successor_chart_defect_without_observation_defect(self):
        def step(h,u):
            return (('output',0),), (1 if h==0 else 2 if h==2 else h)
        out=fibre_audit((0,2),('tick',),step,
            {'parity':lambda h:h%2},('output',),budget=3)
        self.assertEqual(out['charts']['parity']['status'],'refuted')
        self.assertIn('__transition__',
                      out['charts']['parity']['witnessed_failures'])

    def test_malformed_observation_fail_closed(self):
        with self.assertRaises(ValueError):
            fibre_audit((0,),('tick',),
                lambda h,u:((('dup',0),('dup',1)),h),
                {'code':lambda h:h},('dup',),budget=1)
        with self.assertRaises(ValueError):
            fibre_audit((0,),('tick',),
                lambda h,u:((('different',0),),h),
                {'code':lambda h:h},('required',),budget=1)


if __name__=='__main__':
    unittest.main()
