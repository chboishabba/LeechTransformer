import json
import unittest
from types import SimpleNamespace
from tests.test_x369_native_reasoning_evidence import fixture
from scripts.audit_x369_native_reasoning_evidence import native_digest
from scripts.x369_gwb_three_view_boundary import bridge,EDITIONS


def cohort(query):
    c=fixture(3);j=json.loads(c.raw);cap=j['capsule_envelope']['capsule']
    c.query_ref=query;cap['query_ref']=query
    f=j['family'];co=j['cohort'];rows=[]
    for i,(e,o,s) in enumerate(zip(EDITIONS,f['projection']['sources'],cap['sources'])):
        h=str(i)*64;parser='parser:'+e;span=[i,i+1]
        ob=o['observation'];ob.update(parser_or_producer_ref='db-parser:'+parser,
            provenance_refs=['sha256:'+h],native_metadata_refs=['syntax:predicate-span:'+':'.join(map(str,span))])
        s['observation_digest']=native_digest('slr.context.observation.v1',ob)
        rows.append(dict(edition=e,selector_ref=o['selector_ref'],observation=ob,
            source_revision_ref=ob['source_revision_ref'],source_sha256=h,parser_run_ref=parser,
            predicate_span=span,year_support=dict(positive=True,negative=False),
            day_support=dict(positive=True,negative=False) if e=='es' else None,
            candidate_only=True,formal_premise_paid=False,world_truth_paid=False))
    cap['family_digest']=native_digest('slr.context.family.v1',f)
    cap['full_readback_digest']=native_digest('slr.context.full.v1',dict(family=f,cohort=co))
    c.raw=json.dumps(j);c.capsule_digest=native_digest('slr.context.capsule.v1',cap)
    return c,rows


class GwbBoundary(unittest.TestCase):
    def test_independent_queries_and_held_faces(self):
        y,v=cohort('year');d,_=cohort('day')
        h=dict(schema='slr.gwb-office-referee-handoff.v1',family_ref='f',consumer_ref='c',
            views=v,candidate_only=True,formal_premise_paid=False,world_truth_paid=False)
        r=bridge(h,dict(year_support=y,day_support=d))
        self.assertEqual(r['independent_trialectic_faces'],2)
        for lane in r['lanes'].values():
            self.assertFalse(lane['all_direct_pairs_bound'])
            self.assertFalse(lane['all_direct_candidate_answers_match'])
            self.assertTrue(all(x['native_support'] is None for x in lane['edges']))
            self.assertIsNone(lane['boundary']['face_value'])
        with self.assertRaises(ValueError):bridge(h,dict(year_support=y,day_support=y))
        h['views'][0]['predicate_span']=[8,9]
        with self.assertRaises(ValueError):bridge(h,dict(year_support=y,day_support=d))

if __name__=='__main__':unittest.main()
