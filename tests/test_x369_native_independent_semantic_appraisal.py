import copy
import unittest
from scripts.x369_native_independent_semantic_appraisal import verify,digest,assemble,SCHEMA
from scripts.audit_x369_native_reasoning_evidence import native_digest


def fixture(kind='axis',value=None):
    geometry=dict(axis='centre',address=dict(revolution=None,phase='centre')) if kind=='axis' else dict(
        selectors=['a','b','c'],boundary_digest='fixture',boundary_evidence_refs=['fixture-dyad'],all_three_pair_receipts_available=False)
    scope=dict(kind=kind,geometry=geometry,source_revision_ref='rev',observation_ref='obs',
        consumer_ref='consumer',interpretation_source_ref='fixture',interpretation_sha256='fixture')
    query='query:x369:'+native_digest('x369.independent.semantic.demand.v1',scope)
    row=dict(original_predicate_transport=dict(checked_observation=dict(observation_ref='obs')),outside_model_possible=True)
    c=dict(schema='slr.source-semantic-candidate-fragments.v1',
        original_transport=dict(source_coordinate_meet=dict(source_revision_ref='rev')),rows=[row])
    a=dict(schema=SCHEMA,query_ref=query,kind=kind,geometry=geometry,value=value)
    b=dict(schema='slr-source-semantic-query-binding-v1',proof_scope='bounded-source-predicate-query',
        query_ref=query,source_revision_ref='rev',observation_ref='obs',consumer_ref='consumer',
        statement_digest='fixture-statement',evidence_ref='fixture-'+kind,candidate_receipt_sha256=digest(c),
        predicate_row_sha256=digest(row),answer_sha256=digest(a))
    r=dict(schema='slr.authenticated-bounded-source-semantic-query.v1',formal_premise_paid=True,
        common_current_authority_snapshot=False,creates_semantic_authority=False,
        all_live_semantic_readings_covered=False,governed_identity_paid=False,world_truth_paid=False,
        writes_database=False,reopen_refs=['fixture'],binding=b,candidate_receipt=c,answer=a)
    return r,scope,query


class IndependentAppraisal(unittest.TestCase):
    def test_axis_missing_and_both_and_neither(self):
        for value,residual,avail in [(None,'missing',False),
            (dict(positive=True,negative=True),'both',False),
            (dict(positive=False,negative=False),'neither',True)]:
            a=verify(*fixture(value=value))
            self.assertEqual(a['result']['residual'],residual)
            self.assertEqual(a['result']['available'],avail)
            self.assertFalse(a['common_current_authority_snapshot'])

    def test_face_not_inferred_and_bound_to_geometry(self):
        a=verify(*fixture('face','underdeterminedFace'))
        self.assertTrue(a['result']['available'])
        self.assertFalse(a['result']['complete_boundary_and_face'])
        self.assertFalse(verify(*fixture('face'))['result']['available'])
        r,s,q=fixture('face','reciprocalFace')
        r['answer']['value']='coerciveFace'
        with self.assertRaises(ValueError):verify(r,s,q)
        r,s,q=fixture('face','reciprocalFace')
        r['binding']['evidence_ref']='fixture-dyad'
        with self.assertRaises(ValueError):verify(r,s,q)
        r,s,q=fixture('face','reciprocalFace')
        other=copy.deepcopy(s);other['geometry']['boundary_digest']='foreign'
        with self.assertRaises(ValueError):verify(r,other,q)

    def test_candidate_receipt_and_evidence_reuse(self):
        r,s,q=fixture();r['candidate_receipt']['rows'][0]['outside_model_possible']=False
        with self.assertRaises(ValueError):verify(r,s,q)
        a=verify(*fixture(value=dict(positive=True,negative=False)))
        f=verify(*fixture('face','reciprocalFace'))
        out=assemble([a,f])
        self.assertEqual(len(out['held_axes']),12)
        self.assertTrue(out['independent_face_available'])
        with self.assertRaises(ValueError):assemble([a,a])

if __name__=='__main__':unittest.main()
