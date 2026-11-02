"""Counterfactual layer-1/0 MoE failure ablations on REAL saved checkpoints.

Reuse the existing strict loader, model-level CE evaluator, trained
history-bearing MoE and even-channel owner. All variants retain original
trained weights; interventions are temporary and always restored.

Data-selection contract: use TRAIN token IDs for diagnosis. Do not select
architectures or gates using final heldout. The teacher-forced oracle
scalar fits and scores on the SAME probe, therefore its reported RMSE
is an *optimistic diagnostic bound*, never a quality certificate.

Hypotheses: even/odd split, history accumulation, routing starvation,
orientation, output magnitude vs output direction, and depth propagation.
A positive result from modifying a saved model is NOT evidence that
the corresponding student could have learned that intervention.

Original Leech-LILA: Anatolii Kornienko,
DOI 10.5281/zenodo.18784424.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from models.dense_to_signed_experts import canonical_even_input
from scripts.diagnose_moe_quality_collapse import (
    load_pair, token_cross_entropy, substituted_ffns)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class ZeroOddMoE(nn.Module):
    """Keeps even learner only; no learned expert computation or history."""
    def __init__(self, original):
        super().__init__()
        self.d_model = original.d_model
        self.num_experts = original.num_experts
        self.top_k = original.top_k

    def forward(self, x, history=None, train_sign=False):
        b, t, d = x.shape
        h = (x.new_zeros(b, self.num_experts, d)
             if history is None else history)
        ids = torch.arange(self.top_k,device=x.device).expand(b,t,-1)
        weights = x.new_full((b,t,self.top_k),1/self.top_k)
        return x.new_zeros(x.shape),h,ids,x.new_zeros(weights.shape),weights


class ResetHistoryMoE(nn.Module):
    """Reset every expert history before EACH token; preserve live experts.

    It is deliberately not the original recurrent architecture. It
    distinguishes accumulated state from a stateless selected expert.
    """
    def __init__(self, original):
        super().__init__()
        self.original = original

    def forward(self, x, history=None, train_sign=False):
        if x.ndim!=3:
            raise ValueError("reset history requires [B,T,d]")
        b,t,d=x.shape
        outputs,ids,signs,weights=[],[],[],[]
        for index in range(t):
            zero=x.new_zeros(b,self.original.num_experts,d)
            y,_,routing,orientation,mix=self.original.step(
                x[:,index],zero,train_sign=train_sign)
            outputs.append(y);ids.append(routing)
            signs.append(orientation);weights.append(mix)
        return (torch.stack(outputs,dim=1),
                x.new_zeros(b,self.original.num_experts,d),
                torch.stack(ids,dim=1),torch.stack(signs,dim=1),
                torch.stack(weights,dim=1))


@contextmanager
def route_orientation_ablation(moe, variant):
    """Temporarily modify a real router/sign scorer without rewriting state.

    balanced_router chooses the same top-k by batch row at every token,
    deliberately abolishing learned routing. Its batch dependence is
    part of the negative-control definition (not a deployable router).
    zero_orientation uses the existing first-nonzero tie-break.
    flip_orientation reverses the learned decision hyperplane.
    """
    core = getattr(moe,"core",moe)
    if variant not in ("normal","balanced_router","zero_orientation",
                       "flip_orientation"):
        raise ValueError("unknown router/orientation counterfactual")
    if variant == "normal":
        yield
        return
    if variant == "balanced_router":
        router=core.router
        if "forward" in router.__dict__:
            original_forward=router.__dict__["forward"]
            inherited=False
        else:
            original_forward=None
            inherited=True

        def balanced_forward(features):
            b,e,_=features.shape
            k=core.top_k
            chosen=(torch.arange(b,device=features.device)[:,None]+
                    torch.arange(k,device=features.device)[None,:])%e
            # core.step adds learned expert_bias AFTER router.forward.
            # Subtract that bias, ensuring exactly the predetermined rank.
            logits=(torch.full((b,e),-1000.,device=features.device,
                               dtype=features.dtype)-
                    core.expert_bias[None,:].detach())
            ranks=torch.arange(k,0,-1,device=features.device,
                               dtype=features.dtype).expand(b,-1)
            return logits.scatter(1,chosen,ranks)[:,:,None]
        router.forward=balanced_forward
        try:
            yield
        finally:
            if inherited:
                del router.forward
            else:
                router.forward=original_forward
        return
    weight=core.orientation.weight
    old=weight.detach().clone()
    try:
        with torch.no_grad():
            if variant=="zero_orientation":
                weight.zero_()
            else:
                weight.neg_()
        yield
    finally:
        with torch.no_grad():
            weight.copy_(old)


@contextmanager
def selected_variant(teacher, student, layer: int, variant: str):
    """Intervene at exactly one block; restore ALL references and tensors."""
    if not 0<=layer<len(teacher.blocks):
        raise ValueError("layer out of range")
    variants=("full","even_only","odd_only","reset_history",
              "balanced_router","zero_orientation","flip_orientation")
    if variant not in variants:
        raise ValueError("unknown counterfactual variant")
    with substituted_ffns(teacher,student,[layer]):
        block=teacher.blocks[layer]
        old_moe,old_even=block.moe,block.even_student
        try:
            if variant=="even_only":
                block.moe=ZeroOddMoE(old_moe)
            elif variant=="odd_only":
                block.even_student=None
            elif variant=="reset_history":
                block.moe=ResetHistoryMoE(old_moe)
            with route_orientation_ablation(
                    old_moe,variant if variant in
                    ("balanced_router","zero_orientation","flip_orientation")
                    else "normal"):
                yield
        finally:
            block.moe=old_moe
            block.even_student=old_even


@torch.no_grad()
def teacher_normalized_input(teacher, rows, layer, *, device="cpu"):
    observed=[]
    handle=teacher.blocks[layer].ln2.register_forward_hook(
        lambda module,inputs,output: observed.append(output.detach().clone()))
    try:
        teacher.to(device).eval()
        t=torch.as_tensor(rows,device=device,dtype=torch.long)
        teacher(t)
    finally:
        handle.remove()
    if len(observed)!=1:
        raise AssertionError("normalization hook did not fire exactly once")
    return observed[0]


@torch.no_grad()
def same_input_metrics(reference, predicted):
    if reference.shape!=predicted.shape:
        raise ValueError("reference/predicted shape mismatch")
    a=reference.detach().double().reshape(-1)
    b=predicted.detach().double().reshape(-1)
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise ValueError("nonfinite layer outputs")
    a2=float(a.dot(a));b2=float(b.dot(b));ab=float(a.dot(b))
    count=a.numel()
    mse=float((a-b).square().mean())
    scale=ab/b2 if b2>0 else 0.
    optim=float((a-scale*b).square().mean())
    rms_ref=(a2/count)**.5
    rms_pred=(b2/count)**.5
    return {
        "reference_rms":rms_ref,
        "student_rms":rms_pred,
        "rmse":mse**.5,
        "relative_mse":mse/max(a2/count,1e-30),
        "uncentered_cosine":ab/(a2*b2)**.5 if a2>0 and b2>0 else None,
        "same_probe_least_squares_scalar":scale,
        "same_probe_oracle_scaled_rmse":optim**.5,
        "oracle_warning":"Scalar is fitted and evaluated on identical data; it cannot certify unseen performance or restore lost direction."
    }


@torch.no_grad()
def logit_nll_stability(reference_logits, candidate_logits, tokens):
    """F08 numeric witness for |NLL_a-NLL_b| <= 2*||logits_a-logits_b||∞.

    The bound holds pointwise on each token because logsumexp is
    1-Lipschitz under the infinity norm, plus the target-logit term.
    Float64 evaluation is used for diagnostic stability; it is NOT
    an Agda proof or a bound from one neural layer to all others.
    """
    if (reference_logits.shape!=candidate_logits.shape or
        reference_logits.ndim!=3 or
        tokens.shape!=reference_logits.shape[:2] or
        reference_logits.shape[1]<2):
        raise ValueError("same-model full logits and token targets required")
    a=reference_logits[:,:-1,:].double()
    b=candidate_logits[:,:-1,:].double()
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise ValueError("nonfinite logits")
    targets=tokens[:,1:].long().to(a.device)
    if bool((targets<0).any()) or bool((targets>=a.shape[-1]).any()):
        raise ValueError("invalid target token")
    a_nll=-F.log_softmax(a,dim=-1).gather(-1,targets[:,:,None]).squeeze(-1)
    b_nll=-F.log_softmax(b,dim=-1).gather(-1,targets[:,:,None]).squeeze(-1)
    delta=(a-b).abs().amax(-1)
    nll_delta=(a_nll-b_nll).abs()
    bound=2*delta
    slack=bound-nll_delta
    if bool((slack < -1e-9).any()):
        raise AssertionError("pointwise logit infinity/CE stability violated")
    return {
        "probe_target_count":int(nll_delta.numel()),
        "max_pointwise_nll_abs_delta":float(nll_delta.max()),
        "max_pointwise_twice_logit_linf_bound":float(bound.max()),
        "min_pointwise_bound_slack":float(slack.min()),
        "sample_mean_nll_abs_delta":float(nll_delta.mean()),
        "verified_float64_on_this_probe":True,
        "scope":"full-model same-token pointwise logit NLL bound, not a layerwise Lipschitz certificate"}


@torch.no_grad()
def ablate(teacher, student, rows, *, layer=1, batch=2,
           teacher_forced_rows=2, device="cpu", progress=None):
    if (rows.ndim!=2 or len(rows)<2 or rows.shape[1]<2 or
        rows.shape[1]>teacher.cfg.block_size or
        not np.issubdtype(rows.dtype,np.integer) or
        np.any(rows<0) or np.any(rows>=teacher.cfg.vocab_size)):
        raise ValueError("expected real [N,T] numeric training token IDs")
    if not 0<=layer<len(teacher.blocks) or teacher_forced_rows<1 or batch<1:
        raise ValueError("invalid layer/probe/batch")
    # Check the shared transformer backbone, not merely matching shapes.
    teacher_state=teacher.state_dict()
    student_state=student.state_dict()
    shared=[key for key in teacher_state if key in student_state and
            ".ffn." not in key]
    if not shared or any(not torch.equal(teacher_state[key],student_state[key])
                         for key in shared):
        raise ValueError("student shared backbone differs from frozen teacher")
    baseline=token_cross_entropy(teacher,rows,batch=batch,device=device)
    original_full=token_cross_entropy(student,rows,batch=batch,device=device)
    x=teacher_normalized_input(teacher,rows[:teacher_forced_rows],
                               layer,device=device)
    teacher.eval();student.eval()
    reference=teacher.blocks[layer].ffn(x)
    probe_ids=torch.as_tensor(rows[:teacher_forced_rows],
                              device=device,dtype=torch.long)
    reference_logits=teacher(probe_ids)[0]
    cases={}
    variants=("full","even_only","odd_only","reset_history",
              "balanced_router","zero_orientation","flip_orientation")
    for variant in variants:
        if progress:progress("layer %d: %s"%(layer,variant))
        with selected_variant(teacher,student,layer,variant):
            candidate=teacher.blocks[layer]
            odd=candidate.moe(x)[0]
            even=(candidate.even_student(canonical_even_input(x))
                  if candidate.even_student is not None else
                  torch.zeros_like(odd))
            metrics=same_input_metrics(reference,odd+even)
            candidate_probe_logits=teacher(probe_ids)[0]
            logit_bound=logit_nll_stability(
                reference_logits,candidate_probe_logits,probe_ids)
            entry=token_cross_entropy(teacher,rows,batch=batch,device=device)
        cases[variant]={**metrics,**entry,
                        "same_probe_logit_nll_bound":logit_bound,
                        "delta_vs_dense_ce":(entry["ce_nats"]-baseline["ce_nats"]
                            if entry["ce_nats"] is not None else None)}
    if teacher.blocks[layer].moe is not None:
        raise AssertionError("teacher MoE intervention leaked past restoration")
    if teacher.blocks[layer].even_student is not None:
        raise AssertionError("teacher even intervention leaked past restoration")
    return {
        "baseline_dense":baseline,
        "original_complete_student":original_full,
        "layer":layer,"variants":cases,
        "teacher_forced_probe_sequences":min(teacher_forced_rows,len(rows)),
        "ce_sequences":len(rows),
        "source_split":"train_only",
        "shared_backbone_identical":True,
        "interpretation":"All CE counterfactuals use the same source inputs and same frozen teacher backbone. Teacher-forced oracle scalar is not trained or deployed. Routing/sign interventions do not isolate model retrainability.",
        "status":"diagnostic only; no student weights trained, no quality gates admitted"}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--teacher",required=True,type=Path)
    p.add_argument("--student",required=True,type=Path)
    p.add_argument("--config",type=Path)
    p.add_argument("--train-ids",required=True,type=Path,
                   help="train-only token IDs, never final heldout for ablation selection")
    p.add_argument("--layer",type=int,default=1)
    p.add_argument("--batch",type=int,default=2)
    p.add_argument("--probe-rows",type=int,default=2)
    p.add_argument("--device",default="cpu")
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    teacher,student=load_pair(a.teacher,a.student,a.config)
    rows=np.load(a.train_ids,allow_pickle=False)
    result=ablate(teacher,student,rows,layer=a.layer,batch=a.batch,
                  teacher_forced_rows=a.probe_rows,device=a.device,
                  progress=lambda message:print("[ablate] "+message,flush=True))
    result["sources"]={
        "teacher_checkpoint_sha256":sha256_file(a.teacher),
        "student_checkpoint_sha256":sha256_file(a.student),
        "train_ids_sha256":sha256_file(a.train_ids)}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({
        "output":str(a.output),
        "layer":a.layer,
        "dense_ce":result["baseline_dense"]["ce_nats"],
        "candidate_ce":result["original_complete_student"]["ce_nats"],
        "variants":{key:row["delta_vs_dense_ce"]
                    for key,row in result["variants"].items()}
    },indent=2))


if __name__=="__main__":
    main()
