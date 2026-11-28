import unittest
from tests.test_x369_native_reasoning_evidence import fixture
from scripts.x369_native_stage13_evidence_manifest import (
    plan,compile_axes,AXES,expected_address)


class NativeStage13(unittest.TestCase):
    def test_no_fake_axes_or_centre(self):
        c=fixture(2);m=plan(c,{'path':'fixture'})
        self.assertEqual(len(m['actual_source_candidates']),4)
        r=compile_axes(m,{'source':c})
        self.assertEqual(r['held_axes'],list(AXES))
        self.assertEqual(r['bound_axis_count'],0)
        self.assertTrue(all(x['trit'] is None for x in r['axes']))
        self.assertFalse(r['centre_independently_addressed'])
        self.assertEqual(expected_address(11),dict(revolution=2,phase='overflow'))

    def test_duplicate_reciprocal_answer_is_not_independent(self):
        c=fixture(2);m=plan(c,{'path':'fixture'})
        a,b=[x for x in m['actual_source_candidates'] if x['selector_ref']=='s0']
        def binding(r):
            return dict(context='source',coordinate=r['coordinate'],selector_ref=r['selector_ref'],
                comparison_ref=r['comparison_ref'],observation_ref=r['observation_ref'],
                interpretation_source_ref='fixture-declaration')
        m['axes'][0]['binding']=binding(a)
        r=compile_axes(m,{'source':c})
        self.assertEqual(r['bound_axis_count'],1)
        self.assertEqual(r['axes'][0]['residual'],'missing')
        self.assertIn('atlas-0',r['held_axes'])
        m['axes'][1]['binding']=binding(b)
        with self.assertRaises(ValueError):compile_axes(m,{'source':c})

if __name__=='__main__':unittest.main()
