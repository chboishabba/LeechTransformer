"""Matched-parameter dense FFN negative control for C2/E8 early-layer rescue.

A dense GELU MLP, randomly initialised, replaces exactly ONE teacher
FFN. All original attention, embeddings, norms and other FFNs stay
frozen. Match the parameter count of the candidate layer's trainable
odd-MoE plus even student (to within <2*d+1 parameters).

Train on fit only; evaluate on TRAIN-derived development only. This
tests the steelman 'maybe any student fails at this budget/steps',
not the actual geometric model, real E8 invariance or teacher-free
12-layer promotion.

Original Leech-LILA source: Anatolii Kornienko,
DOI 10.5281/zenodo.18784424.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.diagnose_moe_quality_collapse import (
    load_pair, token_cross_entropy)
from scripts.train_signed_wave_experts import layer_inputs


def sha256_file(path: Path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1 << 20),b""):
            h.update(block)
    return h.hexdigest()


def dense_width_for_budget(d_model:int,budget:int):
    if d_model<1 or budget<=d_model:
        raise ValueError("insufficient target student parameters")
    width=(budget-d_model)//(2*d_model+1)
    if width<1:
        raise ValueError("target budget cannot fit dense FFN")
    return width


def make_control(teacher,student,layer:int):
    if not 0<=layer<teacher.cfg.n_layers:
        raise ValueError("layer outside teacher")
    if student.blocks[layer].moe is None or student.blocks[layer].even_student is None:
        raise ValueError("comparison requires odd MoE plus even student")
    target_params=(sum(p.numel() for p in student.blocks[layer].moe.parameters())+
                   sum(p.numel() for p in student.blocks[layer].even_student.parameters()))
    d=teacher.cfg.d_model
    width=dense_width_for_budget(d,target_params)
    control=copy.deepcopy(teacher)
    control.blocks[layer].ffn=nn.Sequential(
        nn.Linear(d,width),nn.GELU(),nn.Linear(width,d),
        nn.Dropout(0.))
    for param in control.parameters():
        param.requires_grad_(False)
    for param in control.blocks[layer].ffn.parameters():
        param.requires_grad_(True)
    actual=sum(p.numel() for p in control.blocks[layer].ffn.parameters())
    if not 0<=target_params-actual<2*d+1:
        raise AssertionError("dense comparison parameter match failed")
    return control,dict(
        matched_student_trainable_params=target_params,
        dense_control_params=actual,
        unmatched_parameter_slack=target_params-actual,
        dense_hidden_width=width,
        original_teacher_ffn_params=sum(
            p.numel() for p in teacher.blocks[layer].ffn.parameters()))


def _validate_rows(rows,model):
    if (rows.ndim!=2 or len(rows)<2 or not 2<=rows.shape[1]<=model.cfg.block_size
        or not np.issubdtype(rows.dtype,np.integer) or
        np.any(rows<0) or np.any(rows>=model.cfg.vocab_size)):
        raise ValueError("invalid consecutive original tokenizer IDs")


def train_control(teacher,student,fit,development,*,layer=1,epochs=3,
                  lr=3e-4,batch=2,seed=41,device="cpu",
                  local_mse_weight=1.,global_ce_weight=1.,
                  global_kl_weight=1.,progress=None):
    _validate_rows(fit,teacher)
    _validate_rows(development,teacher)
    if (set(row.tobytes() for row in fit)&
            set(row.tobytes() for row in development)):
        raise ValueError("fit/development token windows overlap")
    if (epochs<1 or batch<1 or lr<=0 or local_mse_weight<0 or
        global_ce_weight<0 or global_kl_weight<0 or
        local_mse_weight+global_ce_weight+global_kl_weight<=0):
        raise ValueError("bad dense negative-control hyperparameters")
    torch.manual_seed(seed)
    control,capacity=make_control(teacher,student,layer)
    control.to(device)
    teacher.to(device).eval()
    control.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    params=list(control.blocks[layer].ffn.parameters())
    optimizer=torch.optim.AdamW(params,lr=lr)
    if progress:
        progress("measuring dense and initial-control baselines")
    reference_ce=token_cross_entropy(teacher,development,batch=batch,
                                     device=device)["ce_nats"]
    before=token_cross_entropy(control,development,batch=batch,
                               device=device)["ce_nats"]
    history=[]
    for epoch in range(epochs):
        if progress:
            progress("epoch %d/%d: fitting %d rows" % (epoch + 1, epochs, len(fit)))
        # Only the substituted FFN is trainable; dropout frozen/eval to
        # maintain teacher/state comparability and deterministic losses.
        control.eval()
        order=np.random.default_rng(seed+epoch).permutation(len(fit))
        losses=[]
        for start in range(0,len(fit),batch):
            if progress and (start == 0 or start + batch >= len(fit) or
                             (start // batch) % 10 == 0):
                progress("epoch %d/%d: batch %d/%d" % (
                    epoch + 1, epochs, min(start // batch + 1, (len(fit) + batch - 1) // batch),
                    (len(fit) + batch - 1) // batch))
            ids=torch.as_tensor(fit[order[start:start+batch]],dtype=torch.long,
                                device=device)
            with torch.no_grad():
                teacher_logits=teacher(ids)[0]
                x=layer_inputs(teacher,ids,layer)
                local_target=teacher.blocks[layer].ffn(x)
            optimizer.zero_grad(set_to_none=True)
            local=F.mse_loss(control.blocks[layer].ffn(x),local_target)
            logits=control(ids)[0]
            next_logits=logits[:,:-1,:].reshape(-1,logits.shape[-1])
            next_tokens=ids[:,1:].reshape(-1)
            ce=F.cross_entropy(next_logits,next_tokens)
            teacher_probs=F.softmax(teacher_logits.detach(),dim=-1)
            kl=F.kl_div(F.log_softmax(logits,dim=-1),teacher_probs,
                        reduction="none").sum(-1).mean()
            loss=local_mse_weight*local+global_ce_weight*ce+global_kl_weight*kl
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite dense negative control")
            loss.backward()
            nn.utils.clip_grad_norm_(params,1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        dev=token_cross_entropy(control,development,batch=batch,device=device)
        history.append({
            "epoch":epoch+1,"training_loss_avg":sum(losses)/len(losses),
            "train_only_development_ce_nats":dev["ce_nats"],
            "target_tokens":dev["target_tokens"],
            "dense_teacher_development_ce_nats":reference_ce})
        if progress:
            progress("epoch %d/%d: development scoring complete" %
                     (epoch + 1, epochs))
    return control,{
        "layer":layer,**capacity,
        "source":"capacity-matched, single-FFN dense GELU negative control",
        "dense_teacher_development_ce_nats":reference_ce,
        "initial_control_development_ce_nats":before,
        "final_control_development_ce_nats":history[-1]["train_only_development_ce_nats"],
        "epochs":history,
        "fit_rows":len(fit),"development_rows":len(development),
        "selection_scope":"training-only development; not independent final heldout",
        "full_teacher_free_promotion":False,
        "limits":"One trained replacement FFN; cannot claim generalisation, E8 parity, 12-layer teacher-free quality or WAN performance."}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--teacher",required=True,type=Path)
    p.add_argument("--student",required=True,type=Path)
    p.add_argument("--config",type=Path)
    p.add_argument("--fit-ids",required=True,type=Path)
    p.add_argument("--development-ids",required=True,type=Path)
    p.add_argument("--layer",type=int,default=1)
    p.add_argument("--epochs",type=int,default=3)
    p.add_argument("--batch",type=int,default=2)
    p.add_argument("--learning-rate",type=float,default=3e-4)
    p.add_argument("--seed",type=int,default=41)
    p.add_argument("--device",default="cpu")
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    teacher,student=load_pair(a.teacher,a.student,a.config)
    fit=np.load(a.fit_ids,allow_pickle=False)
    development=np.load(a.development_ids,allow_pickle=False)
    model,report=train_control(teacher,student,fit,development,
        layer=a.layer,epochs=a.epochs,batch=a.batch,seed=a.seed,
        device=a.device,lr=a.learning_rate,
        progress=lambda message:print("[dense-control] " + message, flush=True))
    report.update({"teacher_checkpoint_sha256":sha256_file(a.teacher),
                   "student_checkpoint_sha256":sha256_file(a.student),
                   "fit_ids_sha256":sha256_file(a.fit_ids),
                   "development_ids_sha256":sha256_file(a.development_ids)})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"model_state_dict":model.cpu().state_dict(),
                "config":dataclasses.asdict(model.cfg),
                "dense_control_receipt":report},a.output)
    a.output.with_suffix(".json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
