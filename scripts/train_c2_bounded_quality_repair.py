"""Bounded native C2 repair warm-start, no 80-minute curriculum replay.

Initialises from the user's original full trained guarded checkpoint,
preserving ALL source weights. Trains original layer0/1 even branches and
layer10/11 transitions with a config-persistent radial cap. A matched
identity/cap/early-even control is separately measured, not inferred.

Development-only fit and checkpoint selection; NO heldout data read.
Outputs a full guarded source checkpoint with the repair config encoded,
and a separate fail-closed receipt. Release promotion remains FALSE.
"""
from __future__ import annotations
import argparse
import dataclasses
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from models.model import LeechGPT
from scripts.reproduce_c2_quality_and_layers import load_trained
from scripts.train_guarded_moe_continuation import (
    student_rollout_cross_entropy,sha256_file)
from scripts.verify_trained_federated_moe import _load_checkpoint_weights_only

def recover_split(original,train):
    if original.get("kind")!="native_gpu_cumulative_student_quality_curriculum":
        raise ValueError("not original native curriculum receipt")
    fit=original.get("train_fit_indices")
    dev=original.get("train_development_indices")
    if (not isinstance(fit,list) or not isinstance(dev,list)
       or not fit or not dev or set(fit)&set(dev)
       or len(set(fit))!=len(fit) or len(set(dev))!=len(dev)
       or any(type(i)!=int or i<0 or i>=len(train) for i in fit+dev)):
        raise ValueError("invalid original fit/dev split")
    if {train[i].tobytes() for i in fit}&{train[i].tobytes() for i in dev}:
        raise ValueError("original fit/dev window identity leakage")
    return train[fit],train[dev],fit,dev

def configure_original(source,*,cap,even0,even1):
    if cap is not None and (cap<=0 or not math.isfinite(cap)):
        raise ValueError("finite positive source cap required")
    if any(not math.isfinite(x) or x<0 for x in (even0,even1)):
        raise ValueError("finite nonnegative even scales required")
    config=dataclasses.replace(source.cfg,moe_late_history_cap=cap,
        moe_even_layer0_scale=even0,moe_even_layer1_scale=even1)
    model=LeechGPT(config).eval()
    model.load_state_dict(source.state_dict(),strict=True)
    return model

def trainable_parameters(model,*,late):
    for p in model.parameters():
        p.requires_grad_(False)
        p.grad=None
    params=[]
    for index in (0,1):
        branch=model.blocks[index].even_student
        if branch is None:
            raise ValueError("original even branch absent")
        for p in branch.parameters():
            p.requires_grad_(True)
            params.append(p)
    if late:
        for index in (10,11):
            for p in model.blocks[index].moe.transitions.parameters():
                p.requires_grad_(True)
                params.append(p)
    return params

@torch.no_grad()
def measure(model,dev,batch,device):
    model.eval()
    result=student_rollout_cross_entropy(model,dev,batch=batch,
                                          device=device)
    if result is None or not math.isfinite(result):
        raise ValueError("nonfinite source development CE")
    return float(result)

def run(a):
    if a.output.exists() or a.output.with_suffix(".json").exists():
        raise FileExistsError("never overwrite existing candidate or receipt")
    if a.device not in ("cuda","cuda:0") or not torch.cuda.is_available():
        raise RuntimeError("real native HIP GPU required")
    if a.epochs<1 or a.batch<1 or a.lr<=0 or a.max_steps<1:
        raise ValueError("invalid bounded repair budget")
    if (a.history_cap is None or a.history_cap<=0 or
        not math.isfinite(a.history_cap)):
        raise ValueError("explicit finite history cap required")
    orig=json.loads(a.original_receipt.read_text())
    if sha256_file(a.train_ids)!=orig.get("train_ids_sha256"):
        raise ValueError("original train token SHA mismatch")
    if sha256_file(a.teacher)!=orig.get("source_teacher_sha256"):
        raise ValueError("original teacher SHA mismatch")
    embedded=_load_checkpoint_weights_only(a.candidate)
    embedded_receipt=embedded.get("quality_curriculum_receipt")
    if not isinstance(embedded_receipt,dict) or embedded_receipt.get(
           "train_ids_sha256")!=orig.get("train_ids_sha256"):
        raise ValueError("source candidate/receipt mismatch")
    train=np.load(a.train_ids,allow_pickle=False)
    if train.ndim!=2 or not np.issubdtype(train.dtype,np.integer):
        raise ValueError("invalid original token array")
    fit,dev,fit_indices,dev_indices=recover_split(orig,train)
    if len(fit)>a.max_fit_rows:fit=fit[:a.max_fit_rows]
    _,original=load_trained(a.candidate,a.teacher,a.config)
    original.eval().to(a.device)
    if len(original.blocks)<12:raise ValueError("12 C2 layers required")
    teacher=measure(original,dev,a.batch,a.device)
    # Compare *same weights, same data* with and without source cap before
    # training; this control is never used to alter original checkpoint.
    model=configure_original(original,cap=a.history_cap,
        even0=a.layer0_even_scale,even1=a.layer1_even_scale).to(a.device).eval()
    baseline=measure(model,dev,a.batch,a.device)
    if a.layer0_even_scale==0.:
        raise ValueError("zero even layer0 scale blocks branch gradients; use 0.75 or a separate frozen ablation")
    parameters=trainable_parameters(model,late=a.train_late_transitions)
    optimiser=torch.optim.AdamW(parameters,lr=a.lr)
    best_dev=baseline
    best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    best_step=0
    history=[]
    rng=np.random.default_rng(a.seed)
    start=time.perf_counter()
    step=0
    for epoch in range(a.epochs):
        permutation=rng.permutation(len(fit))
        for start_idx in range(0,len(fit),a.batch):
            if step>=a.max_steps:break
            ids=torch.as_tensor(fit[permutation[start_idx:start_idx+a.batch]],
                                device=a.device,dtype=torch.long)
            optimiser.zero_grad(set_to_none=True)
            # Eval-mode preserves original signed-orientation and dropout
            # semantics while autograd still differentiates permitted
            # source even branches and transition weights.
            model.eval()
            logits=model(ids,student_only=True)[0]
            loss=F.cross_entropy(logits[:,:-1,:].reshape(-1,logits.shape[-1]),
                                 ids[:,1:].reshape(-1))
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite actual full-student training CE")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters,a.gradient_clip)
            optimiser.step()
            step+=1
            if step==1 or step%a.dev_every==0 or step==a.max_steps:
                dev_score=measure(model,dev,a.batch,a.device)
                if dev_score<best_dev:
                    best_dev=dev_score
                    best_step=step
                    best_state={k:v.detach().cpu().clone()
                                for k,v in model.state_dict().items()}
                history.append({"step":step,"epoch":epoch+1,
                    "fit_loss":float(loss.detach()),"development_ce":dev_score,
                    "best_development_ce":best_dev})
        if step>=a.max_steps:break
    if step<1:raise ValueError("no source train steps executed")
    model.load_state_dict(best_state,strict=True)
    recovered=measure(model,dev,a.batch,a.device)
    if abs(recovered-best_dev)>1e-5:
        raise AssertionError("best-weight rollback dev CE mismatch")
    receipt={"schema":1,"kind":"native_c2_bounded_early_even_warmstart",
      "source_candidate_sha256":sha256_file(a.candidate),
      "source_teacher_sha256":sha256_file(a.teacher),
      "source_train_sha256":sha256_file(a.train_ids),
      "source_heldout_sha256":orig["heldout_ids_sha256"],
      "source_teacher_heldout_ce":orig["teacher_heldout_ce"],
      "heldout_scored_during_repair":False,
      "original_receipt_sha256":sha256_file(a.original_receipt),
      "original_fit_indices":fit_indices,
      "original_dev_indices":dev_indices,
      "fit_rows":len(fit),"dev_rows":len(dev),
      "source_student_unmodified_development_ce":teacher,
      "intervened_pretrain_development_ce":baseline,
      "best_warmstarted_development_ce":best_dev,
      "best_step":best_step,"optimizer_steps":step,
      "history_cap":a.history_cap,
      "layer0_even_scale":a.layer0_even_scale,
      "layer1_even_scale":a.layer1_even_scale,
      "trainable_scope":"even_layer0_1_plus_late_transition10_11" if a.train_late_transitions else "even_layer0_1",
      "gradient_clip":a.gradient_clip,
      "model_train_mode":"eval_mode_with_autograd",
      "history":history,"original_weights_mutated":False,
      "independent_heldout_used":False,"quality_promotion":False,
      "release_admitted":False,
      "seconds":time.perf_counter()-start}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"model_state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},
                "config":dataclasses.asdict(model.cfg),
                "warmstart_repair_receipt":receipt},a.output)
    receipt["output_sha256"]=sha256_file(a.output)
    a.output.with_suffix(".json").write_text(
        json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in receipt.items()
                      if k not in ("history","original_fit_indices","original_dev_indices")},
                     indent=2,sort_keys=True,allow_nan=False))
    return receipt

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate",type=Path,required=True)
    p.add_argument("--teacher",type=Path,required=True)
    p.add_argument("--config",type=Path)
    p.add_argument("--train-ids",type=Path,required=True)
    p.add_argument("--original-receipt",type=Path,required=True)
    p.add_argument("--history-cap",type=float,default=64.)
    p.add_argument("--layer0-even-scale",type=float,default=.75)
    p.add_argument("--layer1-even-scale",type=float,default=1.)
    p.add_argument("--train-late-transitions",action="store_true")
    p.add_argument("--epochs",type=int,default=2)
    p.add_argument("--batch",type=int,default=1)
    p.add_argument("--max-fit-rows",type=int,default=256)
    p.add_argument("--max-steps",type=int,default=24)
    p.add_argument("--dev-every",type=int,default=4)
    p.add_argument("--gradient-clip",type=float,default=1.)
    p.add_argument("--lr",type=float,default=1e-4)
    p.add_argument("--seed",type=int,default=41)
    p.add_argument("--device",default="cuda")
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.max_fit_rows<2 or a.dev_every<1 or a.gradient_clip<=0:
        raise ValueError("invalid original repair training budget")
    run(a)
if __name__=="__main__":main()
