"""Matched C2 vs C2xC3 routing ablation on ORIGINAL bounded student.

Requires a new previously unused validation token file. Rejects exact rows and
>=16-token contiguous overlap with BOTH original fit/development and the
previously viewed independent heldout source; candidate data must come from an
independent corpus/split. The previously viewed heldout is never scored.

Both geometry arms receive the SAME original repaired weights, parameter count,
fit data and step schedule. Only routers and expert_bias are updated. The
history cap / even gains stay config-persistent in both arms. This tests
whether C3-sensitive routing features add value; it does NOT prove semantic
triality, a lattice intertwiner, or a learned modular algorithm.
"""
from __future__ import annotations
import argparse
import dataclasses
import hashlib
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

def rows_from_file(path):
    a=np.load(path,allow_pickle=False)
    if a.ndim!=2 or not len(a) or a.shape[1]<18 or not np.issubdtype(a.dtype,np.integer):
        raise ValueError("require nonempty integer [rows,tokens] length>=18")
    return a

def signatures(rows,width=16):
    """Conservative exact-overlap guard; NOT a semantic near-duplicate detector."""
    seen=set()
    for row in rows:
        values=row.tolist()
        for i in range(len(values)-width+1):
            # Python tuple exact token identity; no random hashes.
            seen.add(tuple(int(x) for x in values[i:i+width]))
    return seen

def validate_fresh(fresh,original,heldout,window=16):
    if fresh.shape[1]!=original.shape[1] or fresh.shape[1]!=heldout.shape[1]:
        raise ValueError("fresh validation token width differs from original")
    old=signatures(original,window)
    old.update(signatures(heldout,window))
    fresh_windows=signatures(fresh,window)
    overlap=old.intersection(fresh_windows)
    if overlap:
        raise ValueError(f"fresh validation has {len(overlap)} overlapping "
                         f"{window}-token windows against seen sources")
    if len({tuple(map(int,row)) for row in fresh})!=len(fresh):
        raise ValueError("duplicate validation rows")
    return len(fresh_windows)

def clone_with_geometry(source,geometry):
    cfg=dataclasses.replace(source.cfg,moe_route_geometry=geometry)
    model=LeechGPT(cfg)
    model.load_state_dict(source.state_dict(),strict=True)
    # Source guarded teacher FFNs remain loaded but never used in student_only.
    if cfg.moe_late_history_cap is None:
        raise ValueError("expected trained bounded source checkpoint")
    for i in (10,11):
        if model.blocks[i].moe.history_cap!=cfg.moe_late_history_cap:
            raise AssertionError("bounded history cap not imported")
    return model

def router_params_only(model):
    for p in model.parameters():
        p.requires_grad_(False)
        p.grad=None
    params=[]
    for block in model.blocks:
        if block.moe is None:raise ValueError("original MoE layer missing")
        for p in block.moe.router.parameters():
            p.requires_grad_(True)
            params.append(p)
        block.moe.expert_bias.requires_grad_(True)
        params.append(block.moe.expert_bias)
    if len(params)!=len(model.blocks)*3:
        raise AssertionError("unexpected source router parameter count")
    return params

@torch.no_grad()
def evaluate(model,data,batch,device):
    model.eval()
    ce=student_rollout_cross_entropy(model,data,batch=batch,device=device)
    if ce is None or not math.isfinite(float(ce)):
        raise ValueError("nonfinite validation CE")
    return float(ce)

def train_arm(model,train,validation,*,args,geometry,seed):
    model=model.to(args.device).eval()
    params=router_params_only(model)
    count=sum(p.numel() for p in params)
    opt=torch.optim.AdamW(params,lr=args.lr,weight_decay=0.)
    rng=np.random.default_rng(seed)
    initial=evaluate(model,validation,args.batch,args.device)
    history=[{"step":0,"validation_ce":initial}]
    best=initial
    best_step=0
    best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    start=time.perf_counter()
    step=0
    while step<args.steps:
        order=rng.permutation(len(train))
        for j in range(0,len(order),args.batch):
            if step>=args.steps:break
            sample=torch.as_tensor(train[order[j:j+args.batch]],
                                    dtype=torch.long,device=args.device)
            opt.zero_grad(set_to_none=True)
            # eval() with autograd ensures dropout-free matched physics,
            # but gradients still update only source router and expert_bias.
            model.eval()
            logits=model(sample,student_only=True)[0]
            loss=F.cross_entropy(
                logits[:,:-1,:].reshape(-1,logits.shape[-1]),
                sample[:,1:].reshape(-1))
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite train loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params,args.grad_clip)
            opt.step()
            step+=1
            if step==args.steps or step%args.eval_every==0:
                val=evaluate(model,validation,args.batch,args.device)
                history.append({"step":step,"fit_loss":float(loss.detach()),
                                "validation_ce":val})
                if val<best:
                    best=val
                    best_step=step
                    best_state={k:v.detach().cpu().clone()
                                for k,v in model.state_dict().items()}
        if len(train)<1:raise AssertionError("empty fit rows")
    model.load_state_dict(best_state,strict=True)
    if abs(evaluate(model,validation,args.batch,args.device)-best)>1e-5:
        raise AssertionError("rollback does not reproduce best validation")
    return {"geometry":geometry,"seed":seed,"initial_fresh_ce":initial,
            "best_fresh_ce":best,"best_step":best_step,
            "trained_parameter_count":count,"steps":step,
            "seconds":time.perf_counter()-start,"history":history}

def run(args):
    if args.output.exists():raise FileExistsError("refuse overwrite comparison")
    if args.models_dir is not None:
        if args.models_dir.exists() and any(args.models_dir.iterdir()):
            raise FileExistsError("refuse writing into nonempty model output directory")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested physical HIP GPU unavailable")
    if args.steps<1 or args.eval_every<1 or args.seeds<1 or args.batch<1:
        raise ValueError("invalid matched training budget")
    if args.max_fit_rows<1 or args.lr<=0 or args.grad_clip<=0:
        raise ValueError("invalid optimiser controls")
    originals=rows_from_file(args.original_train_ids)
    previous_heldout=rows_from_file(args.original_heldout_ids)
    fresh=rows_from_file(args.fresh_validation_ids)
    # Critical: compare original and heldout not only by path but by actual
    # shared 16-grams, to stop leakage from overlapping sliding windows.
    if args.include_c3_square:
        if any(path is None for path in (
                args.prior_f4_development_ids,args.prior_f5_lockbox_ids,
                args.prior_f4_screen_receipt,args.prior_f5_final_receipt)):
            raise ValueError("F6 requires prior F4/F5 data and both original receipts")
        old_screen=json.loads(args.prior_f4_screen_receipt.read_text())
        old_final=json.loads(args.prior_f5_final_receipt.read_text())
        if old_screen.get("kind")!="original_bounded_c2_vs_c2c3_router_matched_screen" or (
                old_final.get("kind")!="frozen_bounded_c2_c2c3_one_time_token_lockbox"):
            raise ValueError("F6 requires original paired F4 and frozen F5 official receipts")
        if old_screen.get("fresh_validation_sha256")!=sha256_file(
                args.prior_f4_development_ids) or (
                old_final.get("locked_test_sha256")!=sha256_file(args.prior_f5_lockbox_ids)) or (
                old_final.get("screen_receipt_sha256")!=sha256_file(
                    args.prior_f4_screen_receipt)) or (
                old_screen.get("source_repaired_checkpoint_sha256")!=sha256_file(args.candidate)):
            raise ValueError("F6 prior F4/F5 receipts do not bind source/data hashes")
        prior_dev=rows_from_file(args.prior_f4_development_ids)
        prior_lockbox=rows_from_file(args.prior_f5_lockbox_ids)
        if not (prior_dev.shape[1]==prior_lockbox.shape[1]==originals.shape[1]):
            raise ValueError("old source splits must have identical context width")
        exclusion=np.concatenate((originals,prior_dev,prior_lockbox))
        unique_fresh=validate_fresh(fresh,exclusion,previous_heldout,
                                    window=args.overlap_window)
    else:
        unique_fresh=validate_fresh(fresh,originals,previous_heldout,
                                    window=args.overlap_window)
    src_receipt=json.loads(args.source_repair_receipt.read_text())
    if src_receipt.get("kind")!="native_c2_bounded_early_even_warmstart":
        raise ValueError("requires bounded source repair, not untrained candidate")
    if sha256_file(args.original_train_ids)!=src_receipt.get("source_train_sha256"):
        raise ValueError("fit source SHA does not match repaired original")
    if sha256_file(args.original_heldout_ids)!=src_receipt.get("source_heldout_sha256"):
        raise ValueError("original heldout SHA does not match repaired source")
    if sha256_file(args.candidate)!=src_receipt.get("output_sha256"):
        raise ValueError("original repaired source checkpoint SHA mismatch")
    fit_indices=src_receipt.get("original_fit_indices")
    if not isinstance(fit_indices,list) or not fit_indices:
        raise ValueError("source original fit indices absent")
    fit=originals[fit_indices[:args.max_fit_rows]]
    if not len(fit):raise ValueError("no original source fit rows")
    # A source originally trained on these fit examples is okay, but fresh
    # *validation* must not overlap either viewed source.
    _,source=load_trained(args.candidate,args.teacher,args.config)
    if source.cfg.moe_route_geometry!="c2_invariant":
        raise ValueError("expected repaired C2 source checkpoint")
    if source.cfg.moe_late_history_cap!=64.:
        raise ValueError("expected source R64 repaired history")
    if source.cfg.d_model%3:
        raise ValueError("cannot build same-width C3 triadic router")
    geometries=("c2_invariant","c2c3_wave","punctured9_wave_matched") if (
        args.include_c3_square) else ("c2_invariant","c2c3_wave")
    if args.include_c3_square and source.cfg.d_model%24:
        raise ValueError("original source needs 24D-aligned Fourier sheets")
    results=[]
    for seed in range(args.seed,args.seed+args.seeds):
        for geometry in geometries:
            model=clone_with_geometry(source,geometry)
            result=train_arm(model,fit,fresh,args=args,
                             geometry=geometry,seed=seed)
            if args.models_dir is not None:
                args.models_dir.mkdir(parents=True,exist_ok=True)
                checkpoint=args.models_dir/f"router_{geometry}_seed{seed}.pt"
                if checkpoint.exists():raise FileExistsError(checkpoint)
                torch.save({
                    "model_state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},
                    "config":dataclasses.asdict(model.cfg),
                    "screen_provenance":{
                        "kind":"fresh_bounded_c2_c2c3_router_only_candidate",
                        "original_source_sha256":sha256_file(args.candidate),
                        "geometry":geometry,"seed":seed,
                        "fit_steps":args.steps,
                        "best_step":result["best_step"],
                        "fresh_validation_sha256":sha256_file(args.fresh_validation_ids),
                        "release_admitted":False,
                    }
                },checkpoint)
                result["candidate_checkpoint"]=str(checkpoint)
                result["candidate_checkpoint_sha256"]=sha256_file(checkpoint)
            results.append(result)
            del model
            if args.device.startswith("cuda"):torch.cuda.empty_cache()
    by_seed={}
    for record in results:
        by_seed.setdefault(record["seed"],{})[record["geometry"]]=record
    paired=[]
    for seed,pair in by_seed.items():
        left=pair["c2_invariant"]
        right=pair["c2c3_wave"]
        if left["trained_parameter_count"]!=right["trained_parameter_count"]:
            raise AssertionError("geometry arms have unequal trained capacity")
        record={"seed":seed,
            "c2_best_ce":left["best_fresh_ce"],
            "c2c3_best_ce":right["best_fresh_ce"],
            "c2c3_minus_c2":right["best_fresh_ce"]-left["best_fresh_ce"]}
        if args.include_c3_square:
            third=pair["punctured9_wave_matched"]
            if third["trained_parameter_count"]!=left["trained_parameter_count"]:
                raise AssertionError("F6 3-arm router trainable parameter mismatch")
            record["c3square_best_ce"]=third["best_fresh_ce"]
            record["c3square_minus_c2"]=third["best_fresh_ce"]-left["best_fresh_ce"]
            record["c3square_minus_c3"]=third["best_fresh_ce"]-right["best_fresh_ce"]
        paired.append(record)
    receipt={"schema":1,"kind":("original_bounded_c2_c3_c3square_three_arm_development_screen"
                                   if args.include_c3_square else
                                   "original_bounded_c2_vs_c2c3_router_matched_screen"),
        "source_repaired_checkpoint_sha256":sha256_file(args.candidate),
        "source_repaired_receipt_sha256":sha256_file(args.source_repair_receipt),
        "fit_sha256":sha256_file(args.original_train_ids),
        "previous_heldout_sha256":sha256_file(args.original_heldout_ids),
        "fresh_validation_sha256":sha256_file(args.fresh_validation_ids),
        "validation_rows":len(fresh),"unique_validation_ngrams":unique_fresh,
        "validation_overlap_window":args.overlap_window,
        "fit_rows":len(fit),"seed":args.seed,"num_seeds":args.seeds,
        "fit_steps_per_arm":args.steps,"same_router_parameter_count":True,
        "source_cap_R64_persisted":True,
        "independent_prior_heldout_scored":False,
        "f6_three_arm_source_identical_initial_router_weights":bool(args.include_c3_square),
        "f6_fixed_nonlinear_feature_lift_not_effective_rank_equality":bool(args.include_c3_square),
        "f6_prior_f4_development_sha256":(sha256_file(args.prior_f4_development_ids) if args.include_c3_square else None),
        "f6_prior_f5_lockbox_sha256":(sha256_file(args.prior_f5_lockbox_ids) if args.include_c3_square else None),
        "f6_prior_f4_screen_receipt_sha256":(sha256_file(args.prior_f4_screen_receipt) if args.include_c3_square else None),
        "f6_prior_f5_final_receipt_sha256":(sha256_file(args.prior_f5_final_receipt) if args.include_c3_square else None),
        "no_semantic_c3_or_D4_triality_claim":True,
        "arms":results,"paired":paired,
        "release_admitted":False,
        "note":"Fresh validation selects best steps: not independent final heldout. "
               "Comparative result needs a further locked test for promotion."}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(receipt,sort_keys=True,indent=2,
                                      allow_nan=False)+"\n")
    print(json.dumps({"paired":paired,"fresh_rows":len(fresh),
                      "source_cap_R64_persisted":True},indent=2))
    return receipt

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("candidate","teacher","original-train-ids",
                 "original-heldout-ids","fresh-validation-ids",
                 "source-repair-receipt","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--config",type=Path)
    p.add_argument("--include-c3-square",action="store_true",
        help="F6 separate three-arm development; never reuse F4/F5 data")
    p.add_argument("--prior-f4-development-ids",type=Path)
    p.add_argument("--prior-f5-lockbox-ids",type=Path)
    p.add_argument("--prior-f4-screen-receipt",type=Path)
    p.add_argument("--prior-f5-final-receipt",type=Path)
    p.add_argument("--models-dir",type=Path,
        help="Optional empty directory to write best-step non-promoted arm weights")
    p.add_argument("--device",default="cuda")
    p.add_argument("--seed",type=int,default=41)
    p.add_argument("--seeds",type=int,default=2)
    p.add_argument("--steps",type=int,default=24)
    p.add_argument("--max-fit-rows",type=int,default=55)
    p.add_argument("--batch",type=int,default=1)
    p.add_argument("--eval-every",type=int,default=6)
    p.add_argument("--lr",type=float,default=1e-4)
    p.add_argument("--grad-clip",type=float,default=1.)
    p.add_argument("--overlap-window",type=int,default=16)
    a=p.parse_args()
    if not 2<=a.overlap_window<=18:
        raise ValueError("overlap window must be within 2..18 for source safety")
    run(a)
if __name__=="__main__":main()
