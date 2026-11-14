"""F4/F6 fresh real-document corpus admission and no-reuse tests."""
import unittest
import numpy as np
from scripts.prepare_x369_document_disjoint_fresh import (
    construct_fresh_splits,validate_prior_inventory,sha,signatures)

class FakeOriginalTokenizer:
    def encode(self,text):
        return [int(x) for x in text.split()]

def documents(n=12):
    for split,offset in (("development",100),("reserved",400)):
        for i in range(n):
            # Each row has independent integer token vocabulary; these
            # tests are artificial CONTRACT inputs, never scientific data.
            start=offset+i*40
            yield {"document_id":f"{split}-{i}",
                   "text":" ".join(str(v) for v in range(start,start+30)),
                   "split":split}

class FreshX369DocumentSplitTests(unittest.TestCase):
    def setUp(self):
        self.old=np.arange(20).reshape(1,20)
        self.tok=FakeOriginalTokenizer()
    def build(self,docs=None,**kwargs):
        return construct_fresh_splits(
            documents() if docs is None else docs,self.tok,[self.old],
            length=20,window=16,development_rows=4,
            reserved_rows=4,seed=41,expected_vocab=2048,**kwargs)
    def test_split_disjoint_from_prior_and_reserved_never_scored(self):
        arrays,receipt=self.build()
        self.assertEqual(arrays["development"].shape,(4,20))
        self.assertEqual(arrays["reserved"].shape,(4,20))
        p=signatures(self.old,16)
        a=signatures(arrays["development"],16)
        b=signatures(arrays["reserved"],16)
        self.assertFalse(p&a)
        self.assertFalse(p&b)
        self.assertFalse(a&b)
        self.assertTrue(receipt["reserved_set_was_not_scored_or_fit"])
        self.assertFalse(receipt["release_document_disjointness_admitted"])
        self.assertEqual(len(set(receipt["source_document_ids"]["development"])),4)
    def test_missing_historical_document_inventory_fails_closed(self):
        with self.assertRaisesRegex(ValueError,"inventory"):
            validate_prior_inventory(None,["abc"],provisional=False)
        h,i,c=validate_prior_inventory(None,["abc"],provisional=True)
        self.assertFalse(c)
        self.assertEqual(len(h),0)
    def test_hash_bound_complete_manifest(self):
        prior_hash=sha(b"actual original train ids")
        m={"kind":"x369_historical_document_inventory_v1",
           "complete":True,"source_text_sha256":[sha(b"old text")],
           "source_document_ids":["old-document-id"],
           "prior_token_array_sha256":[prior_hash]}
        hashes,ids,complete=validate_prior_inventory(
            m,[prior_hash],provisional=False)
        self.assertTrue(complete)
        self.assertEqual(ids,{"old-document-id"})
        with self.assertRaisesRegex(ValueError,"bind"):
            validate_prior_inventory(m,["unlisted"],provisional=False)
    def test_prior_document_identity_and_text_hash_rejected(self):
        docs=list(documents())
        docs[0]["document_id"]="already-used"
        docs[1]["text"]=" ".join(map(str,range(700,730)))
        same_text_sha=sha(docs[1]["text"].encode("utf8"))
        arrays,receipt=self.build(docs,historical_ids={"already-used"},
                                  historical_hashes={same_text_sha})
        self.assertEqual(arrays["development"].shape,(4,20))
        self.assertGreaterEqual(receipt["rejected"]["duplicate_document"],2)
        self.assertNotIn("already-used",receipt["source_document_ids"]["development"])
    def test_token_overlap_and_source_exhaustion(self):
        docs=list(documents())
        docs[0]["text"]=" ".join(map(str,range(0,21)))
        _,receipt=self.build(docs)
        self.assertGreaterEqual(receipt["rejected"][
            "prior_or_internal_token_overlap"],1)
        with self.assertRaisesRegex(ValueError,"exhausted"):
            self.build(docs[:2])
    def test_reject_source_width_and_duplicate_source_text(self):
        with self.assertRaisesRegex(ValueError,"context width"):
            construct_fresh_splits(documents(),self.tok,
                [np.arange(21).reshape(1,21)],length=20,window=16,
                development_rows=3,reserved_rows=3,seed=4)
        docs=list(documents())
        docs[2]["text"]=docs[1]["text"]
        _,receipt=self.build(docs)
        self.assertGreaterEqual(receipt["rejected"]["duplicate_document"],1)

if __name__=="__main__":unittest.main()
