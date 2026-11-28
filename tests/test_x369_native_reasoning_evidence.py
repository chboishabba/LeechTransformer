import json
import unittest
from itertools import combinations
from types import SimpleNamespace
from scripts.audit_x369_native_reasoning_evidence import native_digest,delta,audit,triangle


def fixture(n=3):
    ids=[f's{i}' for i in range(n)]
    obs=[dict(selector_ref=k,observation=dict(observation_ref='o'+k,source_revision_ref='r'+k)) for k in ids]
    sources=[dict(selector_ref=r['selector_ref'],observation_ref=r['observation']['observation_ref'],
        source_revision_ref=r['observation']['source_revision_ref'],
        observation_digest=native_digest('slr.context.observation.v1',r['observation'])) for r in obs]
    family=dict(projection=dict(sources=obs))
    pairs=[]
    for i,(a,b) in enumerate(combinations(ids,2)):
        pairs.append(dict(pair_ref=f'p{i}',left_selector_ref=a,right_selector_ref=b,
            forward_comparison_ref=f'f{i}',backward_comparison_ref=f'b{i}',
            forward=delta(None,None,missing=True),backward=delta(None,None,missing=True),authenticated_pair=None))
    co=dict(shared_transaction_snapshot=True,derives_transitive_licences=False,
        formal_consumer_equivalence_paid=False,all_live_semantic_readings_covered=False,
        source_independence_paid=False,world_truth_paid=False,selector_refs=ids,
        authenticated_pair_count=0,pair_count=len(pairs),pairs=pairs,inconsistent_selector_refs=[])
    capsule=dict(query_ref='q',family_ref='f',world_truth_paid=False,creates_semantic_authority=False,
        sources=sources,pairs=[dict(p,proof_binding=None,proof_readback_digest=None,formal_premise=None) for p in pairs],
        family_digest=native_digest('slr.context.family.v1',family),
        full_readback_digest=native_digest('slr.context.full.v1',dict(family=family,cohort=co)))
    raw=dict(family=family,cohort=co,capsule_envelope=dict(capsule=capsule))
    return SimpleNamespace(raw=json.dumps(raw),capsule_digest=native_digest('slr.context.capsule.v1',capsule),
        query_ref='q',family_ref='f',consumer_ref='c',admission='archival')


class NativeEvidence(unittest.TestCase):
    def test_native_pair_geometry(self):
        c=fixture();r=audit(c)
        self.assertEqual(r['explicit_pair_records'],3)
        self.assertEqual(r['triangle_candidates'],1)
        self.assertEqual(r['available_authenticated_support_coordinates'],0)
        t=triangle(c,['s0','s1','s2'])
        self.assertEqual(len(t['edges']),3)
        self.assertIsNone(t['face_value'])
        self.assertFalse(t['transitive_licence_created'])
        self.assertFalse(t['all_three_pair_receipts_available'])
        with self.assertRaises(ValueError):triangle(fixture(2),['s0','s1','s2'])

    def test_exact_source_digest_and_deltas(self):
        c=fixture(2);j=json.loads(c.raw);j['family']['projection']['sources'][0]['observation']['observation_ref']='tamper'
        c.raw=json.dumps(j)
        with self.assertRaises(ValueError):audit(c)
        p=dict(positive=True,negative=False);n=dict(positive=False,negative=True)
        self.assertEqual(delta(p,n)['lost'],p)
        self.assertEqual(delta(p,n)['gained'],n)
        self.assertEqual(delta(p,n)['opposed_signs'],True)
        self.assertEqual(delta(p,p)['status'],'preserved')
        self.assertEqual(delta(None,p)['status'],'unavailable_answer')

if __name__=='__main__':unittest.main()
