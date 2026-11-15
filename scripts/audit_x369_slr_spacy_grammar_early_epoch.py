"""X369 T0/T1: original-tokenizer, SLR-numeric-tag grammar source/probe.

Input is an explicit SLR-exported JSONL, NOT spaCy re-parsing or an
independently authenticated SLR DB receipt. This is TRAIN/DEVELOPMENT only.
Original Leech checkpoint is frozen; a POS/DEP/MORPH/arc prediction head
fits original model hidden states. No reserved F5 or Monster quality claim.

JSONL: document_id, split ('train'|'development'), source_text,
source_sha256, parser_contract_sha256, parser_receipt_sha256, visibility,
slr_tokens=[{token_id,start_char,end_char,pos_symbol_id,dependency_id,
morph_set_id,head_token_id}], plus prefix_visible_end_char and
parser_input_sha256 if visibility='prefix'. SLR owns original numeric IDs.
No raw SLR database connection here; its source owner must export records.
SentencePiece immutable_proto supplies source-character offsets.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random
import torch

from models.slr_spacy_parallel_surface import (
    SourceSpan,NumericSpacyToken,CompactSymbolVocabulary,
    align_slr_spacy_to_subwords,SpacySyntacticProbe)

def sha_bytes(data): return hashlib.sha256(data).hexdigest()
def sha_file(path): return sha_bytes(Path(path).read_bytes())

def parse_original_doc(record,sp,tokenizer_sha):
    required=("document_id","split","source_text","source_sha256",
              "parser_contract_sha256","parser_receipt_sha256","visibility",
              "slr_tokens")
    if any(k not in record for k in required):
        raise ValueError("source/SLR parser record lacks obligatory fields")
    if record["split"] not in ("train","development"):
        raise ValueError("never consume reserved/test F5 rows")
    if not isinstance(record["document_id"],(str,int)) or not str(record["document_id"]):
        raise ValueError("missing original source document identity")
    if not isinstance(record["slr_tokens"],list) or not record["slr_tokens"]:
        raise ValueError("SLR numeric token export required")
    receipt=record["parser_receipt_sha256"]
    if not isinstance(receipt,str) or len(receipt)!=64 or any(
        c not in "0123456789abcdef" for c in receipt):
        raise ValueError("exact SLR parser receipt hash required")
    text=record["source_text"]
    prefix_end=record.get("prefix_visible_end_char")
    if record["visibility"]=="prefix" and (
        type(prefix_end) is not int or not 0<prefix_end<=len(text)):
        raise ValueError("prefix parser must specify original visible character boundary")
    visible=text if record["visibility"]=="full_context" else text[:prefix_end]
    # The SentencePiece proto (including immutable_proto on the pinned
    # legacy API) reports offsets in raw UTF-8 bytes, NOT Python string
    # codepoint positions. SLR parser token spans are Unicode characters.
    # align_slr_spacy_to_subwords performs the exact boundary conversion.
    pieces=sp.encode(visible,out_type="immutable_proto").pieces
    ids=[int(p.id) for p in pieces]
    if ids!=list(map(int,sp.encode(visible))):
        raise ValueError("SentencePiece token offsets and original encoded ids differ")
    if record.get("expected_token_ids") is not None and ids!=record["expected_token_ids"]:
        raise ValueError("original token IDs mismatch source-tokenizer receipt")
    toks=[]
    for t in record["slr_tokens"]:
        toks.append(NumericSpacyToken(int(t["token_id"]),
            SourceSpan(int(t["start_char"]),int(t["end_char"])),
            t.get("pos_symbol_id"),t.get("dependency_id"),
            t.get("morph_set_id"),t.get("head_token_id")))
    aligned=align_slr_spacy_to_subwords(source_text=text,
        source_sha256=record["source_sha256"],
        parser_contract_sha256=record["parser_contract_sha256"],
        tokenizer_sha256=tokenizer_sha,
        subword_spans=[SourceSpan(int(p.begin),int(p.end)) for p in pieces],
        slr_tokens=toks,offset_unit="utf8_bytes",
        visibility=record["visibility"],prefix_visible_end_char=prefix_end,
        parser_input_sha256=record.get("parser_input_sha256"))
    return dict(id=str(record["document_id"]),split=record["split"],
        source_sha=record["source_sha256"],parser_receipt=receipt,
        token_ids=ids,alignment=aligned,tokens=toks)

def load_docs(jsonl_path,sp,tokenizer_sha):
    records=[];ids=set();hashes={}
    for line in Path(jsonl_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():continue
        record=parse_original_doc(json.loads(line),sp,tokenizer_sha)
        if record["id"] in ids:raise ValueError("duplicate source document identity")
        ids.add(record["id"])
        if (record["source_sha"] in hashes and
            hashes[record["source_sha"]]!=record["split"]):
            raise ValueError("same source text spans train and development")
        hashes[record["source_sha"]]=record["split"]
        records.append(record)
    if not any(r["split"]=="train" for r in records) or not any(
         r["split"]=="development" for r in records):
        raise ValueError("both source-disjoint train and development required")
    return records

def compact_vocab(records,field):
    return CompactSymbolVocabulary([getattr(t,field)
        for r in records if r["split"]=="train" for t in r["tokens"]
        if getattr(t,field) is not None])

def word_targets(record,vocabs,device):
    return {field:torch.tensor([
        vocabs[source].target_id(getattr(t,source))
        for t in record["tokens"]],dtype=torch.long,device=device)
        for field,source in (("pos","pos_symbol_id"),
                             ("dep","dependency_id"),
                             ("morph","morph_set_id"))} | {
      "head":torch.tensor(record["alignment"].head_positions,
                          dtype=torch.long,device=device)}

def grammar_score(probe,cached_records):
    counts={k:[0,0] for k in ("pos","dep","morph","head")}
    confusion={k:Counter() for k in ("pos","dep","morph")}
    las=[0,0];samples=[]
    with torch.no_grad():
        for hidden,targets in cached_records:
            logits=probe(hidden)
            pred={k:logits[k].argmax(dim=-1) for k in counts}
            for k in counts:
                label=targets[k];valid=label!=-100
                if valid.any():
                    counts[k][0]+=int((pred[k][valid]==label[valid]).sum())
                    counts[k][1]+=int(valid.sum())
                    if k in confusion:
                        confusion[k].update(zip(label[valid].tolist(),
                                                 pred[k][valid].tolist()))
            both=(targets["head"]!=-100)&(targets["dep"]!=-100)
            las[0]+=int(((pred["head"]==targets["head"])&
                         (pred["dep"]==targets["dep"])&both).sum())
            las[1]+=int(both.sum())
            try:samples.append(float(probe.loss(logits,targets)))
            except ValueError:pass
    def macro_f1(field):
        cm=confusion[field]
        labels=set(x for x,y in cm)|set(y for x,y in cm)
        if not labels:return None
        vals=[]
        for label in labels:
            tp=cm.get((label,label),0)
            fp=sum(n for (true,pred),n in cm.items()
                   if pred==label and true!=label)
            fn=sum(n for (true,pred),n in cm.items()
                   if true==label and pred!=label)
            vals.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
        return sum(vals)/len(vals)
    return dict(mean_per_doc_objective=sum(samples)/len(samples) if samples else None,
       metrics={k:{"correct":v[0],"count":v[1],
                   "accuracy":v[0]/v[1] if v[1] else None,
                   "macro_f1":macro_f1(k) if k in confusion else None}
                for k,v in counts.items()},
       dependency_LAS=las[0]/las[1] if las[1] else None,
       dependency_LAS_count=las[1])


class MajorityGrammarBaseline:
    """Source train-only majority POS/DEP/MORPH and self-head prior.

    This is intentionally a cheap negative comparator, not a parser.
    """
    def __init__(self,train_cached_records,vocabs):
        self.winners={}
        self.widths={}
        for k,source in (("pos","pos_symbol_id"),
                         ("dep","dependency_id"),("morph","morph_set_id")):
            histogram=Counter(int(v) for _,targets in train_cached_records
                              for v in targets[k].tolist() if int(v)>0)
            if not histogram:
                raise ValueError("train grammar baseline has no source labels: "+k)
            self.winners[k]=sorted(histogram.items(),
                                    key=lambda z:(-z[1],z[0]))[0][0]
            self.widths[k]=vocabs[source].size

    def __call__(self,hidden):
        n=int(hidden.shape[0])
        out={}
        for k in ("pos","dep","morph"):
            logits=hidden.new_zeros(n,self.widths[k])
            logits[:,self.winners[k]]=1
            out[k]=logits
        out["head"]=torch.eye(n,dtype=hidden.dtype,device=hidden.device)
        return out

    @staticmethod
    def loss(logits,targets):
        # A non-probabilistic baseline; only classification counts apply.
        raise ValueError("no trainable objective for constant baseline")


def shuffled_word_hidden(hidden,*,seed,epoch,doc_index):
    """Deterministic word-to-observation mismatch; target order unchanged."""
    n=hidden.shape[0]
    if n<2:
        return hidden
    generator=torch.Generator(device="cpu")
    generator.manual_seed((int(seed)*65537+int(epoch)*4099+
                           int(doc_index)*17)%(2**63-1))
    perm=torch.randperm(n,generator=generator)
    # Force a nonidentity permutation on short documents; otherwise an
    # allegedly shuffled control may secretly use genuine aligned tokens.
    if torch.equal(perm,torch.arange(n)):
        perm=perm.roll(shifts=1)
    return hidden[perm.to(device=hidden.device)]


def run(args):
    import sentencepiece as spm
    if args.result.exists():raise FileExistsError("refuse overwrite receipt")
    tokenizer_sha=sha_file(args.tokenizer)
    sp=spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    if sp.get_piece_size()!=2048:
        raise ValueError("selected ORIGINAL Leech tokenizer must have 2048 pieces")
    records=load_docs(args.documents_jsonl,sp,tokenizer_sha)
    report=dict(schema=1,kind="x369_original_slr_spacy_grammar_early_epoch",
      tokenizer_sha256=tokenizer_sha,
      source_parser_export_sha256=sha_file(args.documents_jsonl),
      train_document_count=sum(r["split"]=="train" for r in records),
      development_document_count=sum(r["split"]=="development" for r in records),
      source_document_identities=[{k:r[k] for k in
          ("id","split","source_sha","parser_receipt")} for r in records],
      source_document_history_identity_independently_verified=False,
      original_slr_export_externally_authenticated=False,
      F5_reserved_token_bytes_opened=False,
      original_moe_or_geometry_trained=False,
      prospective_final_release_admitted=False)
    if args.original_checkpoint is not None:
        if sha_file(args.original_checkpoint)!=args.original_checkpoint_sha256:
            raise ValueError("original model source-checkpoint SHA mismatch")
        from models.model import LeechGPT,LeechConfig
        payload=torch.load(args.original_checkpoint,
                           map_location="cpu",weights_only=True)
        if "config" not in payload or "model_state_dict" not in payload:
            raise ValueError("checkpoint lacks original model config/state")
        model=LeechGPT(LeechConfig(**payload["config"]))
        model.load_state_dict(payload["model_state_dict"],strict=True)
        model.eval().to(args.device)
        for p in model.parameters():p.requires_grad_(False)
        vocabs={field:compact_vocab(records,field) for field in
                ("pos_symbol_id","dependency_id","morph_set_id")}
        cached={"train":[],"development":[]}
        for r in records:
            if len(r["token_ids"])>model.cfg.block_size:
                raise ValueError("original Leech context exceeded; no silent truncation")
            idx=torch.tensor([r["token_ids"]],
                              dtype=torch.long,device=args.device)
            with torch.no_grad():hidden=model(idx)[1][0]
            aligned=r["alignment"].word_hidden(hidden).detach()
            targets=word_targets(r,vocabs,args.device)
            cached[r["split"]].append((aligned,targets))
        torch.manual_seed(args.seed)
        probe=SpacySyntacticProbe(model.cfg.d_model,
            vocabs["pos_symbol_id"].size,vocabs["dependency_id"].size,
            vocabs["morph_set_id"].size).to(args.device)
        opt=torch.optim.AdamW(probe.parameters(),lr=args.lr)
        # Identical initial head, optimiser, source rows and epoch budget,
        # but source hidden WORDS are mismatched with their grammar targets.
        # This is an explicit supervised control for lexical coupling.
        negative=copy.deepcopy(probe)
        negative_opt=torch.optim.AdamW(negative.parameters(),lr=args.lr)
        majority=MajorityGrammarBaseline(cached["train"],vocabs)
        scores=[]
        negative_scores=[]
        baseline_development=grammar_score(majority,cached["development"])
        for epoch in range(args.epochs+1):
            probe.eval()
            negative.eval()
            scores.append(dict(epoch=epoch,
                train=grammar_score(probe,cached["train"]),
                development=grammar_score(probe,cached["development"])))
            negative_scores.append(dict(epoch=epoch,
                train=grammar_score(negative,cached["train"]),
                development=grammar_score(negative,cached["development"])))
            if epoch==args.epochs:break
            probe.train()
            negative.train()
            order=list(range(len(cached["train"])))
            random.Random(args.seed+epoch).shuffle(order)
            for i in order:
                h,y=cached["train"][i]
                opt.zero_grad();loss=probe.loss(probe(h),y)
                loss.backward();torch.nn.utils.clip_grad_norm_(probe.parameters(),1.)
                opt.step()
                h_control=shuffled_word_hidden(h,seed=args.seed,
                                                epoch=epoch,doc_index=i)
                negative_opt.zero_grad()
                control_loss=negative.loss(negative(h_control),y)
                control_loss.backward()
                torch.nn.utils.clip_grad_norm_(negative.parameters(),1.)
                negative_opt.step()
        report.update(original_checkpoint_sha256=sha_file(args.original_checkpoint),
                      trainable_probe_only=True,
                      compact_vocabs={k:len(v.lookup) for k,v in vocabs.items()},
                      seed=args.seed,epochs=args.epochs,
                      source_hidden_probe_results=scores,
                      train_label_majority_self_head_baseline=baseline_development,
                      word_mismatched_training_negative_control=negative_scores,
                      word_mismatch_control_eligible_train_docs=sum(
                          h.shape[0]>=2 for h,_ in cached["train"]),
                      parser_assisted_pretraining_quality_measured=False)
    else:
        report["source_validation_only_no_model_execution"]=True
    args.result.parent.mkdir(parents=True,exist_ok=True)
    args.result.write_text(json.dumps(report,indent=2,sort_keys=True,
                                     allow_nan=False)+"\n")
    print(json.dumps({k:report[k] for k in
        ("kind","train_document_count","development_document_count")},indent=2))
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ("documents-jsonl","tokenizer","result"):
        p.add_argument("--"+flag,type=Path,required=True)
    p.add_argument("--original-checkpoint",type=Path)
    p.add_argument("--original-checkpoint-sha256")
    p.add_argument("--device",default="cpu")
    p.add_argument("--epochs",type=int,default=8)
    p.add_argument("--lr",type=float,default=.001)
    p.add_argument("--seed",type=int,default=41)
    args=p.parse_args()
    if (args.epochs<0 or args.epochs>256 or not 0<args.lr<1 or
        (args.original_checkpoint is None)!=(args.original_checkpoint_sha256 is None)):
        p.error("bounded optimizer config and complete source-checkpoint pair required")
    run(args)

if __name__=="__main__":main()
