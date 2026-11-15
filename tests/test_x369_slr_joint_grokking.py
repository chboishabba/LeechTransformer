"""Original Leech FULL-WEIGHT grammar/grokking integration, CPU receipts.

Actual LeechGPT forward/backward, original source token IDs and parser spans.
Synthetic words/labels test code, not corpus learning or RX580 quality claims.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import hashlib
import unittest
import numpy as np
import torch

from models.model import LeechConfig,LeechGPT
from models.slr_spacy_parallel_surface import (
    AlignedSpacyDoc,SourceSpan,NumericSpacyToken,CompactSymbolVocabulary)
from scripts.train_x369_slr_joint_grokking import (
    ARMS,SyntaxScaffoldedLeech,word_to_token_label_distribution,
    validate_same_source,initial_zero_gate_weld,score_development,train_arm,
    spectrum_of_original_document_vectors,original_source_mechanism_snapshot)
from scripts.x369_long_grokking_receipts import source_sha


def fixture(split="train",text="Lily saw Max."):
    words=((0,4),(5,8),(9,12),(12,13))
    spans=tuple(SourceSpan(*v) for v in words)
    tokens=tuple(NumericSpacyToken(
        token_id=101+i,span=sp,pos_symbol_id=200+i,
        dependency_id=300+i,morph_set_id=400+i,head_token_id=102)
        for i,sp in enumerate(spans))
    w=torch.zeros(4,len(text),dtype=torch.float64)
    for i,(start,end) in enumerate(words):
        w[i,start:end]=1/(end-start)
    digest=hashlib.sha256(text.encode()).hexdigest()
    a=AlignedSpacyDoc(digest,"a"*64,"b"*64,"full_context",w,
                     tuple(t.token_id for t in tokens),(1,1,1,1),spans)
    ids=list(range(1,len(text)+1))
    return dict(id=split+"-doc",split=split,source_sha=digest,
                parser_receipt="c"*64,token_ids=ids,alignment=a,
                tokens=tokens)


def initialized_model():
    torch.manual_seed(13)
    return LeechGPT(LeechConfig(vocab_size=32,d_model=32,n_layers=1,
        n_heads=4,block_size=32,dropout=0.,lambda_geo=0.))


def vocabs(records):
    return {name:CompactSymbolVocabulary([
        getattr(t,name) for r in records if r["split"]=="train"
        for t in r["tokens"]])
        for name in ("pos_symbol_id","dependency_id","morph_set_id")}


class IntegratedOriginalJointGrokking(unittest.TestCase):
    def test_four_real_leech_arms_initial_source_logit_parity(self):
        docs=[fixture(),fixture("development")]
        source=initialized_model().eval()
        v=vocabs(docs)
        for arm in ARMS:
            with self.subTest(arm=arm):
                model=SyntaxScaffoldedLeech(source,v,arm).eval()
                self.assertTrue(initial_zero_gate_weld(model,docs[0],v,"cpu"))
                self.assertEqual(len(source.tok_emb._forward_hooks),0)

    def test_full_weight_gradient_and_gate_opening(self):
        docs=[fixture(),fixture("development")]
        v=vocabs(docs)
        for arm in ("oracle_context","shuffled_context","predicted_syntax"):
            with self.subTest(arm=arm):
                model=SyntaxScaffoldedLeech(initialized_model(),v,arm).train()
                opt=torch.optim.AdamW(model.parameters(),lr=0.005)
                opt.zero_grad()
                lm,aux,_=model.forward_document(docs[0],v,device="cpu")
                assert torch.isfinite(lm) and torch.isfinite(aux)
                loss=lm+0.2*aux
                loss.backward()
                self.assertIsNotNone(model.backbone.tok_emb.weight.grad)
                self.assertIsNotNone(model.side.gate.grad)
                self.assertTrue(torch.isfinite(model.side.gate.grad))
                self.assertNotEqual(float(model.side.gate.grad),0.)
                opt.step()
                self.assertNotEqual(float(model.side.gate.detach()),0.)

    def test_word_to_subword_soft_classes_missing_and_crossword(self):
        doc=fixture()
        label=torch.tensor([1,2,0,3],dtype=torch.long)
        d,m=word_to_token_label_distribution(doc["alignment"],label,
                                              classes=5,device="cpu",
                                              dtype=torch.float32)
        self.assertEqual(d.shape,(13,5))
        self.assertTrue(bool(m[0]))
        self.assertFalse(bool(m[8]))
        self.assertFalse(bool(m[10]))
        self.assertEqual(float(d[0,1]),1.)
        self.assertEqual(float(d[5,2]),1.)
        self.assertEqual(float(d[12,3]),1.)
        self.assertEqual(float(d[:,0].sum()),0.)

    def test_real_source_development_ce_all_four_arms(self):
        docs=[fixture(),fixture("development")]
        v=vocabs(docs)
        source=initialized_model()
        validate_same_source(docs,source)
        for arm in ARMS:
            model=SyntaxScaffoldedLeech(source,v,arm)
            metrics=score_development(model,docs[1:],v,"cpu",0.2,41)
            self.assertEqual(metrics["source_docs"],1)
            self.assertEqual(metrics["source_target_tokens"],12)
            self.assertTrue(np.isfinite(metrics["original_next_token_ce_nats"]))
        bad=dict(docs[1],token_ids=docs[1]["token_ids"]*5)
        with self.assertRaisesRegex(ValueError,"original dense context"):
            validate_same_source([bad],source)
        bad=dict(docs[0])
        bad["alignment"]=bad["alignment"].__class__(
            bad["alignment"].source_sha256,
            bad["alignment"].parser_contract_sha256,
            bad["alignment"].tokenizer_sha256,"prefix",
            bad["alignment"].weights,bad["alignment"].token_ids,
            bad["alignment"].head_positions,
            bad["alignment"].source_character_spans)
        with self.assertRaisesRegex(ValueError,"full-context"):
            validate_same_source([bad],source)

    def test_full_model_AdamW_document_cursor_resume_exact(self):
        train=fixture()
        dev=fixture("development")
        # Separate document source identity is mandatory in physical run;
        # these synthetic words deliberately reuse bytes to isolate the
        # end-to-end optimizer replay source logic.
        docs=[train,dev]
        v=vocabs(docs)
        identity={"model_sha256":"a"*64,"tokenizer_sha256":"b"*64,
                  "train_data_sha256":"c"*64,"code_revision":"unit-source"}
        source=initialized_model()
        with TemporaryDirectory() as root:
            d=Path(root)
            def args(path,steps,resume):
                return SimpleNamespace(output_dir=path,device="cpu",
                    seed=41,steps=steps,resume_step=resume,
                    checkpoint_first=2,lr=0.002,
                    aux_weight=0.2,grad_clip=1.)
            continuous=train_arm(arm="predicted_syntax",records=docs,
                original_model=source,vocabs=v,args=args(d/"continuous",4,0),
                identity=identity)
            partial=train_arm(arm="predicted_syntax",records=docs,
                original_model=source,vocabs=v,args=args(d/"resume",2,0),
                identity=identity)
            continued=train_arm(arm="predicted_syntax",records=docs,
                original_model=source,vocabs=v,args=args(d/"resume",4,2),
                identity=identity)
            self.assertEqual(continuous["final_step"],4)
            self.assertEqual(partial["final_step"],2)
            self.assertEqual(continued["final_step"],4)
            p1=d/"continuous"/"predicted_syntax"/"checkpoints"/"step_00000004.pt"
            p2=d/"resume"/"predicted_syntax"/"checkpoints"/"step_00000004.pt"
            a=torch.load(p1,map_location="cpu",weights_only=True)
            b=torch.load(p2,map_location="cpu",weights_only=True)
            self.assertEqual(a["data_cursor"],b["data_cursor"])
            self.assertEqual(a["optimizer_step"],b["optimizer_step"])
            for name,t in a["model_state_dict"].items():
                torch.testing.assert_close(t,b["model_state_dict"][name],
                                           rtol=0,atol=0)
            self.assertEqual(len((d/"resume"/"predicted_syntax"/
                  "per_step.jsonl").read_text().splitlines()),4)

    def test_source_document_gram_effective_rank_and_actual_hidden_capture(self):
        data=torch.tensor([[1.,0.],[0.,1.],[-1.,-1.]])
        result=spectrum_of_original_document_vectors(data)
        self.assertEqual(result["source_document_count"],3)
        self.assertEqual(result["nonzero_rank"],2)
        self.assertGreater(result["effective_rank"],1.)
        self.assertLessEqual(result["effective_rank"],2.)
        self.assertFalse(result["mp_outlier_significance_inferred"])
        single=spectrum_of_original_document_vectors(torch.ones(1,2))
        self.assertEqual(single["nonzero_rank"],0)
        with self.assertRaises(ValueError):
            spectrum_of_original_document_vectors(
                torch.tensor([[float("nan"),0.]]))
        docs=[fixture(),fixture("development")]
        v=vocabs(docs)
        model=SyntaxScaffoldedLeech(initialized_model(),v,
                                    "oracle_context")
        snapshot=original_source_mechanism_snapshot(
            model,docs,v,"cpu",41)
        self.assertEqual(snapshot["source_document_count"],2)
        self.assertEqual(snapshot["hidden_width"],32)
        self.assertTrue(snapshot["real_original_source_hidden_states"])
        self.assertFalse(snapshot["C3_character_action_measured"])
        self.assertFalse(bool(model.backbone.final_norm._forward_hooks))

    def test_parser_oracle_and_predicted_semantics_are_explicit(self):
        docs=[fixture(),fixture("development")]
        v=vocabs(docs)
        source=initialized_model()
        for arm in ARMS:
            model=SyntaxScaffoldedLeech(source,v,arm)
            self.assertEqual(model.arm,arm)
            out=model.forward_document(docs[0],v,device="cpu")
            self.assertEqual(out[2].shape,(1,13,32))
            if arm!="predicted_syntax":
                self.assertEqual(float(out[1]),0.)
            else:
                self.assertGreater(float(out[1]),0.)

if __name__=="__main__":
    unittest.main()
