"""SLR/spaCy parallel source interface: 6 original-source CPU contracts.

These intentionally run on controlled fixture data, not a fabricated
SLR or TinyStories physical parser receipt. Actual original trained-model
and document-level identity remain separate acceptance gates.
"""
import hashlib
import unittest
import torch
from torch import nn
from models.slr_spacy_parallel_surface import (
    SourceSpan, NumericSpacyToken, CompactSymbolVocabulary,
    align_slr_spacy_to_subwords, SpacySyntacticProbe,
    ZeroGatedParserEmbedding)

HEX="a"*64

def fixtures():
    text="Lily saw Max."
    return dict(source_text=text,source_sha256=hashlib.sha256(text.encode()).hexdigest(),
        parser_contract_sha256=HEX,tokenizer_sha256=HEX,visibility="full_context",
        offset_unit="unicode_codepoints",
        subword_spans=[SourceSpan(0,2),SourceSpan(2,4),SourceSpan(5,8),SourceSpan(9,12),SourceSpan(12,13)],
        slr_tokens=[NumericSpacyToken(101,SourceSpan(0,4),1001,2001,3001,102),
                    NumericSpacyToken(102,SourceSpan(5,8),1002,2002,3002,102),
                    NumericSpacyToken(103,SourceSpan(9,12),1001,2003,3001,102),
                    NumericSpacyToken(104,SourceSpan(12,13),1003,2004,3003,102)])

class OriginalSLRParallelContracts(unittest.TestCase):
    def test_original_subword_to_spacy_exact_reopen_and_weights(self):
        a=align_slr_spacy_to_subwords(**fixtures())
        self.assertEqual(a.weights.shape,(4,5))
        self.assertEqual(a.weights[0].tolist(),[.5,.5,0,0,0])
        self.assertEqual(a.head_positions,(1,1,1,1))
        self.assertTrue(torch.allclose(a.weights.sum(-1),torch.ones(4,dtype=torch.float64)))
        h=torch.tensor([[2.,4.],[6.,8.],[9.,12.],[11.,13.],[5.,6.]])
        torch.testing.assert_close(a.word_hidden(h)[0],torch.tensor([4.,6.]))

    def test_unicode_char_and_byte_spans_have_same_alignment(self):
        f=fixtures(); t="Éli saw Max."
        f.update(source_text=t,source_sha256=hashlib.sha256(t.encode()).hexdigest(),
                 slr_tokens=[NumericSpacyToken(1,SourceSpan(0,3),1,1,1,2),
                             NumericSpacyToken(2,SourceSpan(4,7),1,1,1,2),
                             NumericSpacyToken(3,SourceSpan(8,11),1,1,1,2),
                             NumericSpacyToken(4,SourceSpan(11,12),1,1,1,2)])
        f["subword_spans"]=[SourceSpan(0,2),SourceSpan(2,3),SourceSpan(4,7),SourceSpan(8,11),SourceSpan(11,12)]
        char=align_slr_spacy_to_subwords(**f)
        f["offset_unit"]="utf8_bytes"
        f["subword_spans"]=[SourceSpan(0,3),SourceSpan(3,4),SourceSpan(5,8),SourceSpan(9,12),SourceSpan(12,13)]
        byte=align_slr_spacy_to_subwords(**f)
        torch.testing.assert_close(char.weights,byte.weights)
        f["subword_spans"]=[SourceSpan(1,3),SourceSpan(3,4),SourceSpan(5,8),SourceSpan(9,12),SourceSpan(12,13)]
        with self.assertRaisesRegex(ValueError,"UTF-8"):
            align_slr_spacy_to_subwords(**f)

    def test_fail_closed_source_identity_and_head(self):
        f=fixtures(); f["source_sha256"]=HEX
        with self.assertRaisesRegex(ValueError,"original source"):
            align_slr_spacy_to_subwords(**f)
        f=fixtures(); f["slr_tokens"][0]=NumericSpacyToken(101,SourceSpan(0,4),1,1,1,999)
        with self.assertRaisesRegex(ValueError,"dependency head"):
            align_slr_spacy_to_subwords(**f)
        f=fixtures(); f["visibility"]="unqualified"
        with self.assertRaisesRegex(ValueError,"visibility"):
            align_slr_spacy_to_subwords(**f)


    def test_prefix_parser_must_have_prefix_only_receipt(self):
        f=fixtures()
        f["visibility"]="prefix"
        with self.assertRaisesRegex(ValueError,"prefix-only"):
            align_slr_spacy_to_subwords(**f)
        # A shortened source-visible observation excludes Max and punctuation.
        f["slr_tokens"]=f["slr_tokens"][:2]
        f["subword_spans"]=f["subword_spans"][:3]
        f["prefix_visible_end_char"]=8
        f["parser_input_sha256"]=hashlib.sha256(
            f["source_text"][:8].encode()).hexdigest()
        a=align_slr_spacy_to_subwords(**f)
        self.assertEqual(a.weights.shape,(2,3))
        # A full-story parser receipt is not an admissible prefix parser.
        f["parser_input_sha256"]=f["source_sha256"]
        with self.assertRaisesRegex(ValueError,"prefix-only"):
            align_slr_spacy_to_subwords(**f)
        # Prefix parser may not report future tokens.
        f["parser_input_sha256"]=hashlib.sha256(
            f["source_text"][:8].encode()).hexdigest()
        f["slr_tokens"]=fixtures()["slr_tokens"]
        with self.assertRaisesRegex(ValueError,"future-derived"):
            align_slr_spacy_to_subwords(**f)

    def test_zero_is_missing_not_neutral(self):
        v=CompactSymbolVocabulary([100,20,100])
        self.assertEqual(v.size,3)
        self.assertEqual(v.input_id(None),0)
        self.assertEqual(v.target_id(None),-100)
        self.assertEqual(v.input_id(999),0)
        self.assertEqual(v.target_id(999),-100)
        self.assertEqual(v.input_id(20),1)
        with self.assertRaises(ValueError):CompactSymbolVocabulary([0])

    def test_all_word_targets_and_head_graph_train(self):
        torch.manual_seed(7)
        probe=SpacySyntacticProbe(8,4,5,4)
        h=torch.randn(4,8)
        y={"pos":torch.tensor([1,2,1,3]),"dep":torch.tensor([1,2,3,4]),
           "morph":torch.tensor([1,-100,1,3]),"head":torch.tensor([1,1,1,1])}
        opt=torch.optim.AdamW(probe.parameters(),lr=.04)
        before=probe.loss(probe(h),y).item()
        for i in range(40):
            opt.zero_grad();loss=probe.loss(probe(h),y);loss.backward();opt.step()
        after=probe.loss(probe(h),y).item()
        self.assertLess(after,before/3)
        y["pos"]=torch.tensor([0,2,1,3])
        with self.assertRaisesRegex(ValueError,"class zero"):
            probe.loss(probe(h),y)

    def test_actual_original_leech_source_dense_logit_parity(self):
        # The actual LeechGPT forward, not merely a synthetic embedding.
        # This checks default-off parser input has ZERO numerical effect.
        from models.model import LeechConfig, LeechGPT
        torch.manual_seed(19)
        model=LeechGPT(LeechConfig(vocab_size=64,n_layers=1,
                                   block_size=8,dropout=0.,
                                   lambda_geo=0.)).eval()
        adapter=ZeroGatedParserEmbedding(384,4,3,4)
        idx=torch.tensor([[1,5,2,7]])
        labels={"pos":torch.tensor([[1,2,0,1]]),
                "dep":torch.tensor([[1,1,0,2]]),
                "morph":torch.tensor([[2,1,0,1]])}
        with torch.no_grad():
            expected=model(idx)[0]
            got=adapter.run_original(model,idx,labels)[0]
        torch.testing.assert_close(got,expected,rtol=0,atol=0)
        self.assertFalse(bool(model.tok_emb._forward_hooks))

    def test_parser_zero_gate_exact_original_and_positive_gate_changes(self):
        class TinyBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.tok_emb=nn.Embedding(16,8)
                self.head=nn.Linear(8,16)
            def forward(self,idx):return self.head(self.tok_emb(idx))
        torch.manual_seed(12)
        model=TinyBackbone().eval()
        side=ZeroGatedParserEmbedding(8,4,3,4)
        x=torch.tensor([[1,2,3]])
        f={"pos":torch.tensor([[1,2,0]]),"dep":torch.tensor([[1,1,0]]),
           "morph":torch.tensor([[2,1,0]])}
        before=model(x).detach().clone()
        torch.testing.assert_close(side.run_original(model,x,f),before,rtol=0,atol=0)
        self.assertEqual(len(model.tok_emb._forward_hooks),0)
        side.gate.data.fill_(1)
        self.assertGreater((side.run_original(model,x,f)-before).abs().max().item(),1e-5)
        with self.assertRaisesRegex(ValueError,"cache"):
            side.run_original(model,x,f,use_cache=True)
        self.assertEqual(len(model.tok_emb._forward_hooks),0)

if __name__=="__main__":unittest.main()
