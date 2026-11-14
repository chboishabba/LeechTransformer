"""Read-only original C2 student stabilisation / early even-channel dev ablation.

Uses original trained candidate weights with reversible PyTorch forward hooks:
  - selected source MoE transition output is norm-capped or scaled;
  - layer-0/1 even branch output is scaled without changing learned weights.
This is a reversible inference *intervention* and not a trained repair,
spectral-radius proof, or teacher-free quality promotion.

All candidates are ranked on the original train-derived DEVELOPMENT subset;
never use the independent heldout for selection. Saves no model weights.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path

import numpy as np
import torch
from scripts.reproduce_c2_quality_and_layers import load_trained
from scripts.train_guarded_moe_continuation import (
    student_rollout_cross_entropy,sha256_file)

def bounded_transition(y,cap,scale):
    if cap is not None:
        # Compute norms in fp64, avoiding infinity in square reduction on
        # enormous but still finite state vectors. Positive radial bound is
        # O(d)-equivariant and odd: cap(-y)=-cap(y).
        norms=torch.linalg.vector_norm(y.double(),dim=-1,keepdim=True)
        factor=torch.clamp(float(cap)/norms.clamp_min(1e-30),max=1.)
        y=y*factor.to(dtype=y.dtype)
    return y*scale

@contextmanager
def interventions(model,*,late_cap=None,late_scale=1.,even_scales=None):
    if late_cap is not None and (late_cap<=0 or not math.isfinite(late_cap)):
        raise ValueError("invalid late history cap")
    if late_scale<=0 or not math.isfinite(late_scale):
        raise ValueError("invalid late state scale")
    even_scales=even_scales or {}
    hooks=[]
    try:
        for index in (10,11):
            if index>=len(model.blocks):continue
            moe=model.blocks[index].moe
            for trans in moe.transitions:
                hooks.append(trans.register_forward_hook(
                    lambda module,args,out,cap=late_cap,scale=late_scale:
                        bounded_transition(out,cap,scale)))
        for index,scale in even_scales.items():
            if not 0<=index<len(model.blocks) or scale<0 or not math.isfinite(scale):
                raise ValueError("invalid early even output scale")
            branch=model.blocks[index].even_student
            if branch is None:raise ValueError("missing original even branch")
            hooks.append(branch.register_forward_hook(
                lambda module,args,out,alpha=scale:out*alpha))
        yield
    finally:
        for hook in hooks:hook.remove()

def cases():
    # Source C2 is the identity control. Each local intervention is applied
    # to the same frozen checkpoint; do not cascade mutable modifications.
    candidates=[{"name":"baseline","late_cap":None,"late_scale":1.,
                 "even_scales":{}}]
    for cap in (1.,4.,16.,64.):
        candidates.append({"name":f"late_cap_{cap:g}","late_cap":cap,
             "late_scale":1.,"even_scales":{}})
    for alpha in (.1,.5):
        candidates.append({"name":f"late_scale_{alpha:g}","late_cap":None,
             "late_scale":alpha,"even_scales":{}})
    for layer in (0,1):
        for scale in (0.,.5,.75,1.25,1.5):
            candidates.append({"name":f"even_{layer}_scale_{scale:g}",
                "late_cap":None,"late_scale":1.,
                "even_scales":{layer:scale}})
    # A small factorial subset captures obvious interaction/confounding.
    for cap in (4.,16.):
        for alpha in (.5,1.25):
            candidates.append({"name":f"late_cap_{cap:g}_even_01_{alpha:g}",
                "late_cap":cap,"late_scale":1.,
                "even_scales":{0:alpha,1:alpha}})
    return candidates

@torch.no_grad()
def evaluate(model,dev,*,batch,device,case):
    with interventions(model,late_cap=case["late_cap"],
                       late_scale=case["late_scale"],
                       even_scales=case["even_scales"]):
        ce=student_rollout_cross_entropy(
            model,dev,batch=batch,device=device)
    if ce is None or not math.isfinite(ce):
        raise ValueError("nonfinite actual source dev CE")
    return float(ce)

def run(args):
    if args.output.exists():raise FileExistsError("refuse overwrite source repair receipt")
    receipt=json.loads(args.curriculum_receipt.read_text())
    if receipt.get("kind")!="native_gpu_cumulative_student_quality_curriculum":
        raise ValueError("wrong original curriculum receipt")
    if sha256_file(args.train_ids)!=receipt.get("train_ids_sha256"):
        raise ValueError("train source SHA differs from original admitted split")
    if sha256_file(args.teacher)!=receipt.get("source_teacher_sha256"):
        raise ValueError("teacher SHA differs from original admitted split")
    train=np.load(args.train_ids,allow_pickle=False)
    indices=receipt.get("train_development_indices")
    if not isinstance(indices,list) or not indices or len(indices)!=len(set(indices)):
        raise ValueError("missing independent original train-derived development indices")
    if any(type(i)!=int or i<0 or i>=len(train) for i in indices):
        raise ValueError("development split out of range")
    dev=train[indices]
    if args.rows:
        dev=dev[:args.rows]
    if len(dev)<1 or args.batch<1:raise ValueError("no original development rows")
    _,model=load_trained(args.candidate,args.teacher,args.config)
    if model.cfg.n_layers<12:
        raise ValueError("original late-layer intervention requires at least 12 blocks")
    model.to(args.device).eval()
    outcomes=[]
    for case in cases():
        ce=evaluate(model,dev,batch=args.batch,device=args.device,case=case)
        outcomes.append({"name":case["name"],"development_ce":ce,
            "intervention":case})
    baseline=outcomes[0]["development_ce"]
    for row in outcomes:
        row["delta_ce_vs_identity"]=row["development_ce"]-baseline
    ranked=sorted(outcomes,key=lambda x:x["development_ce"])
    result={"kind":"frozen_original_student_dev_intervention_screen",
        "source_candidate_sha256":sha256_file(args.candidate),
        "original_train_sha256":sha256_file(args.train_ids),
        "original_development_indices":indices[:len(dev)],
        "development_rows_scored":len(dev),
        "baseline_dev_ce":baseline,"ranked":ranked,
        "best_dev_candidate":ranked[0]["name"],
        "candidate_weights_modified":False,
        "source_C2_odd_history_sign_preserved_by_radial_cap":True,
        "teacher_ffn_used_during_student_score":False,
        "independent_heldout_scored_or_selected":False,
        "quality_promoted":False,
        "note":"A lower dev CE is an observational causal intervention on frozen weights, NOT a learned stability theorem or heldout generalisation."}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="ranked"},
                     indent=2,sort_keys=True))
    print("Top development-only controls:")
    for r in ranked[:7]:
        print(f'{r["name"]}: CE={r["development_ce"]:.6f}, delta={r["delta_ce_vs_identity"]:+.6f}')
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate",required=True,type=Path)
    p.add_argument("--teacher",required=True,type=Path)
    p.add_argument("--config",type=Path)
    p.add_argument("--train-ids",required=True,type=Path)
    p.add_argument("--curriculum-receipt",required=True,type=Path)
    p.add_argument("--rows",type=int,default=0,
        help="0 uses every original train-derived development row")
    p.add_argument("--batch",type=int,default=1)
    p.add_argument("--device",default="cuda")
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    if a.rows<0:raise ValueError("negative source-row limit")
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("native HIP GPU requested but unavailable")
    run(a)
if __name__=="__main__":main()
