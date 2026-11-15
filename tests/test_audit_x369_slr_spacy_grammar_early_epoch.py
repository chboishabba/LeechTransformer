"""SLR source grammar producer acceptance: numeric IDs, split/visibility and metrics.

A tiny tokenizer stand-in checks source-contract code paths only. It is not
a physical TinyStories/SLR/Leech checkpoint benchmark or substitute tokenizer.
"""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import torch

from scripts.audit_x369_slr_spacy_grammar_early_epoch import (
    parse_original_doc,load_docs,word_targets,grammar_score,
    MajorityGrammarBaseline,shuffled_word_hidden)
from models.slr_spacy_parallel_surface import (
    CompactSymbolVocabulary,SpacySyntacticProbe)

class SourceTokenizer:
    def encode(self,text,out_type=None):
        # Actual SentencePiece immutable_protobuf piece begin/end are
        # Unicode source character spans (see real library integration test).
        p=[SimpleNamespace(id=i+2,begin=i,end=i+1,surface=char)
           for i,char in enumerate(text)]
        if out_type=="immutable_proto":
            return SimpleNamespace(pieces=p)
        return [x.id for x in p]

def fixture(docid='story-1',split='train',text='Lily saw Max.'):
    spans=[(0,4),(5,8),(9,12),(12,13)]
    return dict(schema_version='slr.numeric_spacy.leech_grammar.v1',
        document_id=docid,split=split,source_text=text,
        source_sha256=hashlib.sha256(text.encode()).hexdigest(),
        parser_contract_sha256='a'*64,parser_receipt_sha256='b'*64,
        world_resolution_deferred=True,
        parser_observation_is_semantic_authority=False,
        visibility='full_context',slr_tokens=[
            dict(token_id=100+i,start_char=l,end_char=h,
                 pos_symbol_id=200+i,dependency_id=300+i,
                 morph_set_id=400+i,head_token_id=101,
                 pos_symbol_digest=hashlib.sha256(f'pos:{i}'.encode()).hexdigest(),
                 dependency_digest=hashlib.sha256(f'dep:{i}'.encode()).hexdigest(),
                 morph_set_digest=hashlib.sha256(f'morph:{i}'.encode()).hexdigest())
            for i,(l,h) in enumerate(spans)])

class OriginalSourceSLRGrammarProducerTest(unittest.TestCase):
    def test_original_spm_offsets_and_sourced_numeric_heads(self):
        f=fixture();r=parse_original_doc(f,SourceTokenizer(),'c'*64)
        self.assertEqual(r['alignment'].weights.shape,(4,13))
        self.assertEqual(r['alignment'].head_positions,(1,1,1,1))
        self.assertEqual(r['token_ids'],list(range(2,15)))
        f['expected_token_ids']=[0]
        with self.assertRaisesRegex(ValueError,'token IDs'):
            parse_original_doc(f,SourceTokenizer(),'c'*64)

    def test_unicode_sentencepiece_source_character_spans(self):
        text='Éli saw Max.'
        f=fixture(text=text)
        spans=((0,3),(4,7),(8,11),(11,12))
        f['slr_tokens']=[
            dict(token_id=100+i,start_char=lo,end_char=hi,
                 pos_symbol_id=200+i,dependency_id=300+i,
                 morph_set_id=400+i,head_token_id=101,
                 pos_symbol_digest=hashlib.sha256(f'pos:{i}'.encode()).hexdigest(),
                 dependency_digest=hashlib.sha256(f'dep:{i}'.encode()).hexdigest(),
                 morph_set_digest=hashlib.sha256(f'morph:{i}'.encode()).hexdigest())
            for i,(lo,hi) in enumerate(spans)]
        doc=parse_original_doc(f,SourceTokenizer(),'c'*64)
        self.assertEqual(doc['alignment'].weights.shape,(4,len(text)))
        self.assertEqual(doc['alignment'].weights[0,:3].tolist(),
                         [1/3,1/3,1/3])
        self.assertEqual(doc['alignment'].head_positions,(1,1,1,1))
        # A span not reproducing the original Unicode source character
        # is forbidden even if token IDs would otherwise look correct.
        class BrokenProtoTokenizer(SourceTokenizer):
            def encode(self,text,out_type=None):
                result=super().encode(text,out_type)
                if out_type=='immutable_proto':
                    first=result.pieces[0]
                    result.pieces[0]=SimpleNamespace(id=first.id,
                                                      begin=1,end=first.end,surface=first.surface)
                return result
        with self.assertRaisesRegex(ValueError,'source codepoint span/surface'):
            parse_original_doc(f,BrokenProtoTokenizer(),'c'*64)

    def test_train_development_source_and_reserved_barriers(self):
        with TemporaryDirectory() as td:
            p=Path(td)/'source.jsonl'
            a=fixture()
            b=fixture('story-2','development',text='Lily saw Max.')
            p.write_text('\n'.join(json.dumps(x) for x in (a,b)))
            with self.assertRaisesRegex(ValueError,'same source text'):
                load_docs(p,SourceTokenizer(),'c'*64)
            b['source_text']='Max saw Lily.'
            b['source_sha256']=hashlib.sha256(b['source_text'].encode()).hexdigest()
            b['slr_tokens']=[dict(t,token_id=t['token_id']+200,
                                  head_token_id=t['head_token_id']+200)
                             for t in b['slr_tokens']]
            p.write_text('\n'.join(json.dumps(x) for x in (a,b)))
            self.assertEqual(len(load_docs(p,SourceTokenizer(),'c'*64)),2)
            b['split']='reserved'
            p.write_text('\n'.join(json.dumps(x) for x in (a,b)))
            with self.assertRaisesRegex(ValueError,'reserved'):
                load_docs(p,SourceTokenizer(),'c'*64)

    def test_word_targets_macro_f1_and_head_attachment(self):
        r=parse_original_doc(fixture(),SourceTokenizer(),'c'*64)
        v={name:CompactSymbolVocabulary([
            getattr(t,name) for t in r['tokens']])
            for name in ('pos_symbol_id','dependency_id','morph_set_id')}
        y=word_targets(r,v,'cpu')
        self.assertEqual(y['head'].tolist(),[1]*4)
        p=SpacySyntacticProbe(8,5,5,5)
        s=grammar_score(p,[(torch.randn(4,8),y)])
        self.assertEqual(s['metrics']['pos']['count'],4)
        self.assertEqual(s['dependency_LAS_count'],4)
        self.assertIsNotNone(s['metrics']['pos']['macro_f1'])


    def test_serialized_proto_and_legacy_source_equal(self):
        try:
            from sentencepiece import sentencepiece_pb2
        except ImportError:
            self.skipTest("SentencePiece protobuf required for physical proto-API check")
        class ModernProtoTokenizer(SourceTokenizer):
            def encode(self,text,out_type=None):
                if out_type=="immutable_proto":
                    raise ValueError("immutable_proto unavailable")
                if out_type=="proto":
                    document=sentencepiece_pb2.SentencePieceText()
                    document.text=text
                    for piece in super().encode(text,out_type="immutable_proto").pieces:
                        item=document.pieces.add()
                        item.id=piece.id
                        item.begin=piece.begin
                        item.end=piece.end
                        item.surface=piece.surface
                    return document.SerializeToString()
                return super().encode(text,out_type)
        f=fixture(text="Lily saw Max.")
        legacy=parse_original_doc(f,SourceTokenizer(),'c'*64)
        modern=parse_original_doc(f,ModernProtoTokenizer(),'c'*64)
        self.assertEqual(legacy['token_ids'],modern['token_ids'])
        torch.testing.assert_close(
            legacy['alignment'].weights,modern['alignment'].weights,rtol=0,atol=0)
        self.assertEqual(legacy['alignment'].head_positions,
                         modern['alignment'].head_positions)

    def test_real_sentencepiece_unicode_codepoint_offsets(self):
        # An actual trained SentencePiece package, not fabricated proto spans.
        import contextlib
        import io
        try:
            import sentencepiece as spm
        except ImportError:
            self.skipTest("physical SentencePiece model unavailable in portable Python")
        with TemporaryDirectory() as td:
            base=Path(td)
            corpus=base/'corpus.txt'
            corpus.write_text(('Élie saw Max.\nLily saw Max.\n')*50)
            with contextlib.redirect_stderr(io.StringIO()):
                spm.SentencePieceTrainer.train(
                    input=str(corpus),model_prefix=str(base/'sp'),
                    model_type='char',vocab_size=64,hard_vocab_limit=False)
            tokenizer=spm.SentencePieceProcessor(model_file=str(base/'sp.model'))
            f=fixture(text='Élie saw Max.')
            f['slr_tokens']=[
                dict(token_id=100+i,start_char=a,end_char=b,
                     pos_symbol_id=200+i,dependency_id=300+i,
                     morph_set_id=400+i,head_token_id=101,
                     pos_symbol_digest=hashlib.sha256(f'pos:{i}'.encode()).hexdigest(),
                     dependency_digest=hashlib.sha256(f'dep:{i}'.encode()).hexdigest(),
                     morph_set_digest=hashlib.sha256(f'morph:{i}'.encode()).hexdigest())
                for i,(a,b) in enumerate(((0,4),(5,8),(9,12),(12,13)))]
            result=parse_original_doc(f,tokenizer,'c'*64)
            self.assertGreater(len(result['token_ids']),4)
            self.assertEqual(result['alignment'].weights.shape[0],4)
            self.assertEqual(result['alignment'].head_positions,(1,1,1,1))
            for a in tokenizer.encode(f['source_text'],
                                      out_type='immutable_proto').pieces:
                self.assertTrue(0<=a.begin<=a.end<=len(f['source_text']))
                if a.surface:
                    self.assertEqual(a.surface,
                                     f['source_text'][a.begin:a.end])

    def test_stable_grammar_digests_reject_local_symbol_aliases(self):
        a=fixture()
        b=fixture('story-2','development')
        # Same numeric database-local ID can denote a different symbol in
        # another DB. Stable SHA label identity disambiguates on import.
        b['slr_tokens'][0]['pos_symbol_digest']='f'*64
        r1=parse_original_doc(a,SourceTokenizer(),'c'*64)
        r2=parse_original_doc(b,SourceTokenizer(),'c'*64)
        self.assertNotEqual(r1['tokens'][0].pos_symbol_id,
                            r2['tokens'][0].pos_symbol_id)
        # Inside one SLR document, reusing one ID for two conflicting
        # stable digests is evidence of a broken exporter/DB join.
        a['slr_tokens'][1]['pos_symbol_id']=a['slr_tokens'][0]['pos_symbol_id']
        with self.assertRaisesRegex(ValueError,'conflicting stable digest'):
            parse_original_doc(a,SourceTokenizer(),'c'*64)
        a=fixture()
        del a['slr_tokens'][0]['pos_symbol_digest']
        with self.assertRaisesRegex(ValueError,'stable symbol digest'):
            parse_original_doc(a,SourceTokenizer(),'c'*64)

    def test_original_train_majority_and_self_head_negative_baseline(self):
        r=parse_original_doc(fixture(),SourceTokenizer(),'c'*64)
        v={name:CompactSymbolVocabulary([
            getattr(t,name) for t in r['tokens']])
            for name in ('pos_symbol_id','dependency_id','morph_set_id')}
        targets=word_targets(r,v,'cpu')
        # No development labels are consulted to select majority class.
        reference=torch.randn(4,8)
        prior=MajorityGrammarBaseline([(reference,targets)],v)
        self.assertEqual(prior.winners["pos"],1)
        self.assertEqual(prior(reference)["head"].argmax(-1).tolist(),
                         list(range(4)))
        stats=grammar_score(prior,[(reference,targets)])
        self.assertEqual(stats['metrics']['pos']['correct'],1)
        self.assertEqual(stats['dependency_LAS_count'],4)
        self.assertIsNone(stats["mean_per_doc_objective"])

    def test_frozen_hidden_word_shuffle_deterministic_nonidentity(self):
        h=torch.arange(30.,dtype=torch.float32).reshape(5,6)
        first=shuffled_word_hidden(h,seed=41,epoch=0,doc_index=2)
        second=shuffled_word_hidden(h,seed=41,epoch=0,doc_index=2)
        torch.testing.assert_close(first,second,rtol=0,atol=0)
        self.assertFalse(torch.equal(first,h))
        self.assertEqual(set(map(tuple,first.tolist())),
                         set(map(tuple,h.tolist())))
        self.assertTrue(torch.equal(shuffled_word_hidden(
            h[:1],seed=41,epoch=0,doc_index=2),h[:1]))

    def test_prefix_parser_and_future_original_source_exclusion(self):
        f=fixture();f['visibility']='prefix';f['prefix_visible_end_char']=8
        f['parser_input_sha256']=hashlib.sha256(
            f['source_text'][:8].encode()).hexdigest()
        f['slr_tokens']=f['slr_tokens'][:2]
        r=parse_original_doc(f,SourceTokenizer(),'c'*64)
        self.assertEqual(r['alignment'].weights.shape,(2,8))
        f['slr_tokens']=fixture()['slr_tokens']
        with self.assertRaisesRegex(ValueError,'future-derived'):
            parse_original_doc(f,SourceTokenizer(),'c'*64)

if __name__=='__main__':unittest.main()
