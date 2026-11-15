"""X369 T1/G1: matched original-Leech full-weight grammar-scaffold training.

This is a REAL optimizer runner on source-authenticated SLR parser documents;
not a frozen-head toy probe. Each arm starts from the identical dense Leech
checkpoint, tokenizer and document order, with equal full-model optimizer
updates; only the grammar side condition differs.

Arms:
  token_only:       original Leech, no grammar feature input
  oracle_context:   full-story spaCy numeric categories, an ORACLE upper bound
  shuffled_context: same sidecar/parameters, deliberately misaligned word tags
  aux_only:         same predicted POS/DEP/MORPH + arc supervision but NO
                    syntax injection; isolates auxiliary multitask benefit
  predicted_syntax: token-local learned syntax distributions, never oracle
                    labels as next-token *inputs*, auxiliary source syntax loss

The full-context input arms ARE NOT deployable prefix-causal LM comparisons.
They are explicit parser-oracle counterfactuals. No F5, heldout/test
documents, or Monster/Griess identification is used. Do not infer
improvements without executing against real original source receipts.

All source documents fit the original context or the run fails closed; no
silent truncation or forged head references. One exact document per step;
large long-horizon runs may require a separate batching performance weld.
"""
from __future__ import annotations
import argparse,copy,json,hashlib,math,random,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from models.slr_spacy_parallel_surface import (
    ZeroGatedParserEmbedding,SpacySyntacticProbe,word_feature_inputs)
from scripts.audit_x369_slr_spacy_grammar_early_epoch import (
    load_docs,compact_vocab,word_targets,sha_file)
from scripts.lila_dense_training_parity import load_dense_teacher
from scripts.train_lila_dense_gpu_smoke import deterministic_indices
from scripts.x369_long_grokking_receipts import (
    GrokkingStepLedger,save_exact_replay,load_exact_replay,
    is_full_mechanism_step,source_sha)

ARMS=("token_only","oracle_context","shuffled_context","aux_only","predicted_syntax")
LABELS=(("pos","pos_symbol_id"),("dep","dependency_id"),
        ("morph","morph_set_id"))


def validate_same_source(records,model):
    if model.cfg.moe_experts!=0 or model.cfg.moe_guarded_ffn:
        raise ValueError("first grammar ablation requires original dense source")
    for r in records:
        ids=r["token_ids"]
        if (len(ids)<2 or len(ids)>model.cfg.block_size or
            min(ids)<0 or max(ids)>=model.cfg.vocab_size or
            r["alignment"].weights.shape[1]!=len(ids)):
            raise ValueError("document not an intact original dense context")
        if r["split"] not in ("train","development"):
            raise ValueError("not a permitted original development source")
        if r["alignment"].visibility!="full_context":
            raise ValueError("full-story oracle experiment needs full-context source declaration")


def word_to_token_label_distribution(alignment,word_ids,classes,device,dtype):
    """Soft category targets respecting exact cross-word source subword spans."""
    if word_ids.ndim!=1 or classes<2:
        raise ValueError("invalid source grammar label array")
    w=alignment.weights.to(device=device,dtype=dtype)
    valid=word_ids>0
    if not bool(valid.any()):
        return w.new_zeros(w.shape[1],classes),torch.zeros(
            w.shape[1],dtype=torch.bool,device=device)
    ids=word_ids.to(device=device)
    observed=w*valid.to(dtype=dtype).unsqueeze(1)
    weights=observed.sum(dim=0)
    labels=F.one_hot(ids.clamp_min(0),classes).to(dtype=dtype)
    distribution=(observed.T @ labels)/weights.clamp_min(1.e-12).unsqueeze(-1)
    distribution[:,0]=0
    return distribution,weights>0


def masked_soft_grammar_loss(logits,alignment,word_ids,classes):
    target,available=word_to_token_label_distribution(
        alignment,word_ids,classes,logits.device,logits.dtype)
    if not bool(available.any()):
        return None
    return -(target[available]*
             F.log_softmax(logits[0,available,:],dim=-1)).sum(-1).mean()


class SyntaxScaffoldedLeech(nn.Module):
    """Real LeechGPT parameters and differentiable syntax side surface."""
    def __init__(self,backbone,vocabs,arm:str):
        super().__init__()
        if arm not in ARMS:raise ValueError("unknown original source ablation arm")
        self.backbone=backbone
        self.arm=arm
        width=backbone.cfg.d_model
        dimensions=tuple(vocabs[k].size for _,k in LABELS)
        self.side=ZeroGatedParserEmbedding(width,*dimensions)
        self.predictors=nn.ModuleDict({
            k:nn.Linear(width,dim) for (k,_),dim in zip(LABELS,dimensions)})
        self.dep_graph=SpacySyntacticProbe(width,*dimensions)
        self.dimensions=dict(zip(("pos","dep","morph"),dimensions))

    def _predicted_source_side(self,idx,alignment,*,inject):
        # Token LOCAL embeddings only; predicted labels at position i do not
        # inspect suffix tokens. Supervision may use completed-story labels.
        original=self.backbone.tok_emb(idx)
        predictions={k:self.predictors[k](original) for k,_ in LABELS}
        delta=0.
        for k,_ in LABELS:
            # Category 0 is explicitly missing/no-grammatical-label. We
            # exclude the padding row even under softmax mixing, because a
            # raw matrix multiply bypasses nn.Embedding.padding_idx's
            # gradient suppression and could silently learn a fake zero.
            probs=torch.softmax(predictions[k],dim=-1)
            delta=delta+probs[...,1:] @ getattr(self.side,k).weight[1:]
        # This hook does not persist after one synchronous original forward.
        def hook(module,inputs,output):
            return output+self.side.gate*delta
        if not inject:
            return self.backbone(idx)[0],predictions,original
        handle=self.backbone.tok_emb.register_forward_hook(hook)
        try:
            logits=self.backbone(idx)[0]
        finally:
            handle.remove()
        return logits,predictions,original

    def forward_document(self,record,vocabs,*,device,source_seed=41):
        idx=torch.tensor(record["token_ids"],device=device,
                         dtype=torch.long).unsqueeze(0)
        aligned=record["alignment"]
        ids=word_feature_inputs(record["tokens"],vocabs,device=device)
        if self.arm=="token_only":
            logits=self.backbone(idx)[0]
            predictions=None
            original=None
        elif self.arm in ("predicted_syntax","aux_only"):
            logits,predictions,original=self._predicted_source_side(
                idx,aligned,inject=self.arm=="predicted_syntax")
        else:
            if self.arm=="shuffled_context":
                # Same documents, same category histograms, same sidecar
                # architecture; only original word-label correspondence lost.
                n=len(record["tokens"])
                if n<2:raise ValueError("no actual word-label mismatch possible")
                generator=torch.Generator(device="cpu")
                key=int(hashlib.sha256(record["source_sha"].encode()).hexdigest()[:12],16)
                generator.manual_seed((key+source_seed*65537)%(2**63-1))
                perm=torch.randperm(n,generator=generator)
                if torch.equal(perm,torch.arange(n)):perm=perm.roll(1)
                ids={k:v[perm.to(device=v.device)] for k,v in ids.items()}
            logits=self.side.run_original_word_aligned(
                self.backbone,idx,aligned,ids)[0]
            predictions=None
            original=None
        if logits.shape[1]<2:
            raise ValueError("not enough original source tokens for LM loss")
        # Causal next-token objective on original Leech; all source tokens
        # are fed as input solely to preserve the last word's alignment.
        # The output logit at last position is deliberately unscored.
        lm=F.cross_entropy(logits[:,:-1,:].reshape(-1,logits.shape[-1]),
                           idx[:,1:].reshape(-1))
        aux=lm.new_zeros(())
        if predictions is not None:
            terms=[]
            for k,src in LABELS:
                v=ids[k]
                p=masked_soft_grammar_loss(
                    predictions[k],aligned,v,self.dimensions[k])
                if p is not None:terms.append(p)
            # Word-level dependency arc supervision is a TRAINING auxiliary
            # objective only, not an inference-time future-looking signal.
            word_hidden=aligned.word_hidden(original[0])
            word_graph=self.dep_graph(word_hidden)
            targets=word_targets(record,vocabs,device)
            try:
                terms.append(self.dep_graph.loss(word_graph,targets))
            except ValueError:
                pass
            if not terms:raise ValueError("no supervised original grammar targets")
            aux=torch.stack(terms).mean()
        return lm,aux,logits


@torch.no_grad()
def score_development(model,docs,vocabs,device,aux_weight,seed):
    was_training=model.training
    model.eval()
    token_loss=0.;total_tokens=0;aux_loss=0.;scored=0
    for r in docs:
        lm,aux,_=model.forward_document(
            r,vocabs,device=device,source_seed=seed)
        n=len(r["token_ids"])-1
        token_loss+=float(lm)*n
        total_tokens+=n
        aux_loss+=float(aux)
        scored+=1
    model.train(was_training)
    return dict(source_docs=scored,source_target_tokens=total_tokens,
        original_next_token_ce_nats=token_loss/total_tokens,
        grammar_auxiliary_objective=aux_loss/scored,
        total_objective_ce_plus_aux=token_loss/total_tokens+
               aux_weight*aux_loss/scored)



def spectrum_of_original_document_vectors(vectors):
    """Exact empirical document Gram eigenmass (NOT an iid MP null claim).

    One pooled original hidden vector per source document is a single
    cluster. Document correlation and network training invalidate blind
    Marchenko-Pastur inference; these are descriptive source statistics.
    """
    a=torch.as_tensor(vectors,dtype=torch.float64,device="cpu")
    if a.ndim!=2 or a.shape[0]<1 or a.shape[1]<1 or not bool(torch.isfinite(a).all()):
        raise ValueError("finite source document hidden matrix required")
    n,width=a.shape
    if n<2:
        return dict(source_document_count=n,hidden_width=width,
                    effective_rank=0.,dominant_eigenmass_fraction=None,
                    nonzero_rank=0,empirical_gram_eigenvalues=[],
                    mp_outlier_significance_inferred=False)
    centered=a-a.mean(0,keepdim=True)
    eigen=torch.linalg.eigvalsh(centered @ centered.T / (n-1))
    eigen=torch.clamp(eigen,min=0)
    total=float(eigen.sum())
    if total<=0:
        rank=0;eff=0.;dom=None
    else:
        p=eigen/total
        p=p[p>1e-14]
        rank=len(p)
        eff=float(torch.exp(-(p*torch.log(p)).sum()))
        dom=float(eigen.max()/eigen.sum())
    return dict(source_document_count=n,hidden_width=width,
                effective_rank=eff,dominant_eigenmass_fraction=dom,
                nonzero_rank=rank,empirical_gram_eigenvalues=[
                    float(x) for x in eigen.tolist()],
                mp_outlier_significance_inferred=False)


@torch.no_grad()
def original_source_mechanism_snapshot(model,docs,vocabs,device,seed):
    """Actual source-final-norm hidden states, including the selected adapter."""
    old_mode=model.training
    model.eval()
    vectors=[]
    for record in docs:
        observed=[]
        def capture(module,inputs,output):
            observed.append(output.detach().float().mean(dim=1).cpu()[0])
        hook=model.backbone.final_norm.register_forward_hook(capture)
        try:
            model.forward_document(record,vocabs,device=device,source_seed=seed)
        finally:
            hook.remove()
        if len(observed)!=1:
            raise RuntimeError("source Leech final norm activation capture incomplete")
        vectors.append(observed[0])
    model.train(old_mode)
    summary=spectrum_of_original_document_vectors(torch.stack(vectors))
    summary["parser_gate_value"]=float(model.side.gate.detach())
    summary["real_original_source_hidden_states"]=True
    summary["C3_character_action_measured"]=False
    summary["model_semantic_world_consumer_measured"]=False
    return summary


def initial_zero_gate_weld(model,doc,vocabs,device):
    model.eval()
    idx=torch.tensor(doc["token_ids"],dtype=torch.long,
                     device=device).unsqueeze(0)
    if float(model.side.gate.detach())!=0.:
        raise ValueError("source parser gate must initialise at exactly zero")
    with torch.no_grad():
        baseline=model.backbone(idx)[0]
        actual=model.forward_document(doc,vocabs,device=device)[2]
    if not torch.equal(baseline,actual):
        raise AssertionError("original source checkpoint logits changed at zero parser gate")
    return True


def train_arm(*,arm,records,original_model,vocabs,args,identity):
    output=args.output_dir/arm
    output.mkdir(parents=True,exist_ok=True)
    device=args.device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    model=SyntaxScaffoldedLeech(
        copy.deepcopy(original_model),vocabs,arm).to(device)
    train=[r for r in records if r["split"]=="train"]
    dev=[r for r in records if r["split"]=="development"]
    if arm=="shuffled_context" and any(len(r["tokens"])<2 for r in train+dev):
        raise ValueError("mixed-source sham needs >=2 words per original document")
    initial_zero_gate_weld(model,train[0],vocabs,device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr)
    checkpoint_dir=output/"checkpoints"
    step=0
    if args.resume_step:
        resume=checkpoint_dir/f"step_{args.resume_step:08d}.pt"
        step,cursor=load_exact_replay(resume,model=model,optimizer=optimizer,
                                      expected_source_identity=identity,
                                      expected_sha256=source_sha(resume))
        if (step!=args.resume_step or cursor.get("last_doc_index")!=
            deterministic_indices(len(train),1,step,args.seed)[0]):
            raise ValueError("original data cursor and deterministic source mismatch")
    ledger=GrokkingStepLedger(output/"per_step.jsonl",identity)
    if ledger.last_step!=-1 and ledger.last_step!=step:
        raise ValueError("per-step ledger does not agree with resumed source step")
    if ledger.last_step==-1 and step!=0:
        raise ValueError("missing original per-step ledger on resume")
    initial_dev=score_development(model,dev,vocabs,device,args.aux_weight,args.seed)
    initial_spectrum=original_source_mechanism_snapshot(
        model,dev,vocabs,device,args.seed)
    history=[dict(step=step,development=initial_dev,
                  source_fit_tokens=(ledger.last_token_count
                      if ledger.last_token_count>=0 else 0),
                  source_mechanism=initial_spectrum)]
    for k in range(step+1,args.steps+1):
        document_index=deterministic_indices(len(train),1,k,args.seed)[0]
        record=train[document_index]
        model.train()
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        started=time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        lm,aux,_=model.forward_document(record,vocabs,
            device=device,source_seed=args.seed)
        loss=lm+args.aux_weight*aux
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite original joint grammar model objective")
        loss.backward()
        grad=nn.utils.clip_grad_norm_(
            model.parameters(),args.grad_clip,error_if_nonfinite=True)
        optimizer.step()
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        optimizer_step_seconds=time.perf_counter()-started
        ledger.record(k,token_count=(
            (ledger.last_token_count if ledger.last_token_count>=0 else 0)+
            len(record["token_ids"])-1),
            train_ce=float(lm.detach()),learning_rate=args.lr,
            grad_norm=float(grad.detach()),extra_scalars={
                "syntax_aux_loss":float(aux.detach()),
                "syntax_parser_gate":float(model.side.gate.detach()),
                "total_fit_objective":float(loss.detach()),
                "source_optimizer_step_seconds":optimizer_step_seconds,
                "source_target_tokens_per_second":(
                    (len(record["token_ids"])-1)/max(
                        optimizer_step_seconds,1e-12))})
        if is_full_mechanism_step(k,first=args.checkpoint_first) or k==args.steps:
            cp=checkpoint_dir/f"step_{k:08d}.pt"
            save_exact_replay(cp,step=k,model=model,optimizer=optimizer,
                data_cursor=dict(last_doc_index=document_index,
                                 source_document_id=record["id"],
                                 optimizer_step=k,
                                 deterministic_index_algorithm="absolute_step_epoch_permutation_v1",
                                 original_token_count=ledger.last_token_count),
                source_identity=identity,require_gpu_rng=str(device).startswith("cuda"))
            dev_metric=score_development(
                model,dev,vocabs,device,args.aux_weight,args.seed)
            snapshot=original_source_mechanism_snapshot(
                model,dev,vocabs,device,args.seed)
            history.append(dict(step=k,development=dev_metric,
                train_ce=float(lm.detach()),grammar_loss=float(aux.detach()),
                gate=float(model.side.gate.detach()),
                source_mechanism=snapshot,
                source_fit_tokens=ledger.last_token_count,
                checkpoint_sha256=source_sha(cp)))
    return dict(arm=arm,initial_source_parity=True,
        original_source_checkpoint=identity["model_sha256"],
        parser_condition="full_context_oracle_upper_bound" if arm in (
            "oracle_context","shuffled_context") else (
            "prefix_available_grammar_auxiliary_without_feedback"
            if arm=="aux_only" else (
            "prefix_available_token_local_predicted_syntax_feedback"
            if arm=="predicted_syntax" else "none")),
        source_development_history=history,
        final_step=args.steps,
        F5_reserved_rows_loaded=False,
        trained_original_backbone=True,
        original_C3_Monster_physical_action_verified=False)



def compare_matched_development_arms(results,*,target_ce=None):
    """Paired source/step contrasts; no cross-arm model-selection leakage.

    Changes in grammar feedback vs AUX-only isolate input fusion from syntax
    supervision. The full-context oracle and shuffled oracle are upper-bound
    information regimes, not causal deployment measurements.
    """
    runs={row["arm"]:row for row in results}
    if set(runs)!=set(ARMS):
        return dict(complete_five_arm_comparison=False,
                    missing_arms=sorted(set(ARMS)-set(runs)),
                    no_F5_released=True)
    by_step={}
    for arm,receipt in runs.items():
        trace=receipt["source_development_history"]
        steps=[row["step"] for row in trace]
        if len(set(steps))!=len(steps):
            raise ValueError("duplicate optimizer checkpoint in matched arm")
        by_step[arm]={row["step"]:row for row in trace}
    step_sets=[set(z) for z in by_step.values()]
    if any(z!=step_sets[0] for z in step_sets[1:]):
        raise ValueError("all five arms require the same original step schedule")
    def ce(arm,step):
        v=by_step[arm][step]["development"]["original_next_token_ce_nats"]
        if not isinstance(v,(int,float)) or not math.isfinite(v):
            raise ValueError("nonfinite or malformed matched development CE")
        return v
    trajectories=[]
    for step in sorted(step_sets[0]):
        base=ce("token_only",step)
        oracle=ce("oracle_context",step)
        shuffled=ce("shuffled_context",step)
        aux=ce("aux_only",step)
        predicted=ce("predicted_syntax",step)
        trajectories.append(dict(
            optimizer_step=step,
            source_LM_CE={
                arm:ce(arm,step) for arm in ARMS},
            oracle_minus_token_only=oracle-base,
            oracle_minus_shuffled_control=oracle-shuffled,
            predicted_minus_token_only=predicted-base,
            predicted_minus_aux_only=predicted-aux,
            aux_only_minus_token_only=aux-base,
            original_oracle_access_is_full_story=True))
    first_crossings={}
    if target_ce is not None:
        if not isinstance(target_ce,(int,float)) or not 0<target_ce<float("inf"):
            raise ValueError("positive CE target required")
        for arm in ARMS:
            crossed=next((row["step"] for row in runs[arm][
                "source_development_history"] if
                ce(arm,row["step"])<=target_ce),None)
            first_crossings[arm]=crossed
    return dict(complete_five_arm_comparison=True,
                paired_source_optimizer_step_curves=trajectories,
                predeclared_target_dev_ce=target_ce,
                first_step_reaching_target_dev_ce=first_crossings,
                no_independent_heldout_or_F5_scored=True,
                interpretation="descriptive train/development paired ablation; full-context oracle is not causal deployment",
                parser_feedback_effect_proved=False)


def run(args):
    import sentencepiece as spm
    if args.output_dir.exists() and not args.resume_step and any(
        args.output_dir.iterdir()):
        raise FileExistsError("refuse to overwrite original long-run ablation")
    source_sha_file=sha_file(args.documents_jsonl)
    sp=spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    docs=load_docs(args.documents_jsonl,sp,sha_file(args.tokenizer))
    model=load_dense_teacher(args.original_checkpoint,args.config)
    if sp.get_piece_size()!=model.cfg.vocab_size:
        raise ValueError("source tokenizer differs from original Leech checkpoint vocabulary")
    validate_same_source(docs,model)
    vocabs={src:compact_vocab(docs,src) for _,src in LABELS}
    identity=dict(model_sha256=sha_file(args.original_checkpoint),
         tokenizer_sha256=sha_file(args.tokenizer),
         train_data_sha256=source_sha_file,code_revision=args.code_revision)
    if args.arm=="all":
        arms=ARMS
    else:
        arms=(args.arm,)
    results=[]
    for arm in arms:
        results.append(train_arm(arm=arm,records=docs,
            original_model=model,vocabs=vocabs,args=args,identity=identity))
    summary=dict(schema=1,kind="x369_slr_joint_original_model_grokking",
        source_identity=identity,output_arms=results,
        paired_five_arm_development_contrast=compare_matched_development_arms(
            results,target_ce=args.target_dev_ce),
        all_arms_same_original_checkpoint=True,
        matched_original_train_docs=True,
        matched_deterministic_document_schedule=True,
        full_context_spacy_oracle_not_deployable_causal=True,
        reserved_F5_never_loaded=True,
        live_PNF_world_state_training_performed=False,
        F3_B_RX580_C3_physical_result_included=False)
    dst=args.output_dir/"matched_arm_summary.json"
    if dst.exists():raise FileExistsError("original summary exists")
    dst.write_text(json.dumps(summary,sort_keys=True,indent=2,allow_nan=False)+"\n")
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for item in ("documents-jsonl","original-checkpoint","tokenizer","output-dir"):
        p.add_argument("--"+item,required=True,type=Path)
    p.add_argument("--config",type=Path)
    p.add_argument("--code-revision",required=True)
    p.add_argument("--arm",choices=("all",)+ARMS,default="all")
    p.add_argument("--device",default="cpu")
    p.add_argument("--seed",type=int,default=41)
    p.add_argument("--steps",type=int,default=2048)
    p.add_argument("--resume-step",type=int,default=0)
    p.add_argument("--checkpoint-first",type=int,default=128)
    p.add_argument("--lr",type=float,default=3e-4)
    p.add_argument("--aux-weight",type=float,default=0.2)
    p.add_argument("--grad-clip",type=float,default=1.)
    p.add_argument("--target-dev-ce",type=float,
       help="optional PREDECLARED development CE threshold for time-to-quality")
    args=p.parse_args()
    if (not args.code_revision.strip() or args.steps<1 or
        args.resume_step<0 or args.resume_step>=args.steps or
        args.lr<=0 or args.aux_weight<0 or args.grad_clip<=0 or
        (args.target_dev_ce is not None and args.target_dev_ce<=0) or
        args.checkpoint_first<1 or
        args.checkpoint_first & (args.checkpoint_first-1)):
        p.error("invalid original-source run/optimizer/checkpoint contract")
    run(args)


if __name__=="__main__":main()
