import json
import unittest
from types import SimpleNamespace
from scripts.x369_slr_signed_support_stage import (
    encode_support,reopen_support,source_support_views,audit)


class SignedSupportStage(unittest.TestCase):
    def test_five_distinct_states(self):
        values=[None]+[dict(positive=p,negative=n) for p in (False,True) for n in (False,True)]
        encoded=[encode_support(v) for v in values]
        self.assertEqual(len(set(encoded)),5)
        self.assertEqual(sum(t==0 for t,_ in encoded),3)
        self.assertEqual([reopen_support(v) for v in encoded],values)
        with self.assertRaises(ValueError):
            encode_support(dict(positive=1,negative=False))

    def fixture(self,missing=False,inconsistent=False):
        p=None if missing else dict(positive=True,negative=False)
        q=None if missing else dict(positive=True,negative=True)
        row=dict(pair_ref='pair',left_selector_ref='a',right_selector_ref='b',
                 forward_comparison_ref='ab',backward_comparison_ref='ba',
                 forward=dict(left_support=p,right_support=q,status='different'),
                 backward=dict(left_support=q,right_support=p,status='different'),
                 proof_binding=None if missing else 'fixture')
        body=dict(query_ref='q',family_ref='f',world_truth_paid=False,
                  creates_semantic_authority=False,pairs=[row])
        cohort=dict(pairs=[dict(row,authenticated_pair=None if missing else 'fixture')],
                    inconsistent_selector_refs=['a'] if inconsistent else [])
        return SimpleNamespace(raw=json.dumps(dict(capsule_envelope=dict(capsule=body),cohort=cohort)),
                    query_ref='q',family_ref='f',capsule_digest='fixture',admission='fixture')

    def test_direction_and_hold(self):
        views=source_support_views(self.fixture())
        self.assertEqual(views['pair','backward','left'].selector_ref,'b')
        self.assertEqual(views['pair','backward','left'].residual,'both')
        self.assertEqual(audit(self.fixture())['authenticated_coordinates'],4)
        self.assertEqual(audit(self.fixture(inconsistent=True))['authenticated_coordinates'],2)
        r=audit(self.fixture(missing=True))
        self.assertEqual(r['support_residual_histogram'],{'missing':4})
        self.assertEqual(r['authenticated_coordinates'],0)
        self.assertFalse(r['original_stage13_source_binding_certified'])
        self.assertFalse(r['world_truth_paid'])

if __name__=='__main__':unittest.main()
