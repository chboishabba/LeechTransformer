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
    parse_original_doc,load_docs,word_targets,grammar_score)
from models.slr_spacy_parallel_surface import (
    CompactSymbolVocabulary,SpacySyntacticProbe)

class SourceTokenizer:
    def encode(self,text,out_type=None):
        p=[SimpleNamespace(id=2+i,begin=i,end=i+1) for i in range(len(text))]
        if out_type=="immutable_proto":
            return SimpleNamespace(pieces=p)
        return [x.id for x in p]

def fixture(docid='story-1',split='train',text='Lily saw Max.'):
    spans=[(0,4),(5,8),(9,12),(12,13)]
    return dict(document_id=docid,split=split,source_text=text,
        source_sha256=hashlib.sha256(text.encode()).hexdigest(),
        parser_contract_sha256='a'*64,parser_receipt_sha256='b'*64,
        visibility='full_context',slr_tokens=[
            dict(token_id=100+i,start_char=l,end_char=h,
                 pos_symbol_id=200+i,dependency_id=300+i,
                 morph_set_id=400+i,head_token_id=101)
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
            200+i if name=='pos_symbol_id' else
            300+i if name=='dependency_id' else 400+i
            for i in range(4)]) for name in
           ('pos_symbol_id','dependency_id','morph_set_id')}
        y=word_targets(r,v,'cpu')
        self.assertEqual(y['head'].tolist(),[1]*4)
        p=SpacySyntacticProbe(8,5,5,5)
        s=grammar_score(p,[(torch.randn(4,8),y)])
        self.assertEqual(s['metrics']['pos']['count'],4)
        self.assertEqual(s['dependency_LAS_count'],4)
        self.assertIsNotNone(s['metrics']['pos']['macro_f1'])

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
