import hashlib
import tempfile
import unittest
from pathlib import Path
from scripts.prepare_real_leech_tokens import tokenize_distinct_documents,source_iterators


class DeterministicTokenizer:
    def encode(self,text):
        # Test-only fake tokenizer; production uses checked-in SentencePiece.
        return [2]+[4+(ord(c)%67) for c in text]+[3]


class RealCorpusPreparationTests(unittest.TestCase):
    def test_real_text_windows_split_disjoint_and_deterministic(self):
        train=[{"text":"Original training story number "+str(i)+
                " with enough actual letters and tokens."} for i in range(7)]
        heldout=[{"text":"Genuine validation passage "+str(i)+
                  " with different content and enough tokens."} for i in range(6)]
        kwargs=dict(train_rows=4,heldout_rows=3,length=16,seed=41)
        a,proof=tokenize_distinct_documents(
            iter(train),iter(heldout),DeterministicTokenizer(),**kwargs)
        b,proof2=tokenize_distinct_documents(
            iter(train),iter(heldout),DeterministicTokenizer(),**kwargs)
        self.assertEqual(a["train"].shape,(4,16))
        self.assertEqual(a["heldout"].shape,(3,16))
        self.assertEqual(a["train"].tobytes(),b["train"].tobytes())
        self.assertEqual(a["heldout"].tobytes(),b["heldout"].tobytes())
        self.assertEqual(proof,proof2)
        self.assertNotEqual(proof["train"]["document_hash_set_sha256"],
                            proof["heldout"]["document_hash_set_sha256"])

    def test_duplicate_leakage_rejected_across_splits(self):
        common="Exact same document with enough tokens to create a window."
        train=iter([{"text":common}])
        heldout=iter([{"text":common},{"text":common}])
        with self.assertRaisesRegex(ValueError,"heldout source exhausted"):
            tokenize_distinct_documents(train,heldout,
                DeterministicTokenizer(),train_rows=1,heldout_rows=1,
                length=8,seed=3)
    
    def test_short_or_empty_examples_do_not_become_padding(self):
        docs=iter([{"text":""},{"text":"x"},{"text":"This line genuinely has adequate number of tokens."}])
        result,proof=tokenize_distinct_documents(
            docs,iter([{"text":"Entirely separate genuine validation line with many tokens."}]),
            DeterministicTokenizer(),train_rows=1,heldout_rows=1,length=12,seed=7)
        self.assertEqual(result["train"].shape,(1,12))
        self.assertEqual(proof["train"]["skipped_short"],2)

    def test_local_text_requires_two_distinct_real_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/"training.txt"
            b=Path(tmp)/"validation.txt"
            a.write_text("one real training line\n")
            b.write_text("one disjoint heldout line\n")
            train,heldout,meta=source_iterators(
                "local-text",train_text=a,heldout_text=b)
            self.assertEqual(next(train)["text"],"one real training line")
            self.assertEqual(next(heldout)["text"],"one disjoint heldout line")
            self.assertTrue(meta["disjoint"])
            with self.assertRaises(ValueError):
                source_iterators("local-text",train_text=a,heldout_text=a)


if __name__=="__main__":
    unittest.main()
