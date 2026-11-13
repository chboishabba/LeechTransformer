"""Quality-first progressive native-GPU teacher-to-signed-MoE curriculum.

Train actual guarded student weights, with dense FFNs retained as control in
not-yet-replaced layers; no changed teacher checkpoint. Stage selection and
best-epoch rollback use train-derived calibration ONLY. Independent heldout is
read and scored exactly once after all stages. This is not model admission.
Original Leech-LILA: Anatolii Kornienko; DASHI quality experiment adapter.
"""
from __future__ import annotations
import argparse
import copy
import dataclasses
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from scripts.diagnose_moe_quality_collapse import load_pair,token_cross_entropy
from scripts.train_guarded_moe_continuation import (
    build_guarded_hybrid,student_rollout_cross_entropy,sha256_file)
from scripts.native_training_progress import TrainingProgress

def parse_stages(value,depth):
    if not isinstance(value,str) or not value.strip():
        raise ValueError("nonempty semicolon-separated cumulative stages required")
    stages=[]
    previous=set()
    for raw in value.split(";"):
        words=raw.split(",")
        if any(not w.strip() or not w.strip().isdecimal() for w in words):
            raise ValueError("stage indices must be nonnegative decimal integers")
        stage=tuple(int(w.strip()) for w in words)
        current=set(stage)
        if (len(stage)!=len(current) or not current or
            any(i>=depth for i in current) or not previous.issubset(current) or
            (stages and current==previous)):
            raise ValueError("stages must grow strictly, have unique valid layers")
        stages.append(tuple(sorted(current)))
        previous=current
    return tuple(stages)

def split_train_calibration(train,heldout,seed,fraction):
    if (train.ndim!=2 or heldout.ndim!=2 or
        train.shape[1]!=heldout.shape[1] or train.shape[1]<2 or
        len(train)<8 or len(heldout)<1 or not .05<=fraction<=.4):
        raise ValueError("invalid original train/heldout layout or fraction")
    if (len({row.tobytes() for row in train})!=len(train) or
        len({row.tobytes() for row in heldout})!=len(heldout) or
        {row.tobytes() for row in train}&{row.tobytes() for row in heldout}):
        raise ValueError("repeated/leaked token windows across split")
    order=np.random.default_rng(seed).permutation(len(train))
    n=max(2,int(len(train)*fraction))
    if len(train)-n<2:raise ValueError("insufficient independent fit rows")
    return train[order[n:]],train[order[:n]],order[n:].tolist(),order[:n].tolist()

def train(args):
    if args.device not in ("cuda","cuda:0"):
        raise ValueError("native HIP GPU required; CPU fallback forbidden")
    if not torch.cuda.is_available():
        raise RuntimeError("native ROCm/HIP device unavailable")
    if (args.epochs<1 or args.batch<1 or args.lr<=0 or args.max_rows<8 or
        args.quality_budget<0 or
        args.progress_every<1 or args.student_ce_weight<=0 or
        args.teacher_kl_weight<0 or args.local_mse_weight<0):
        raise ValueError("invalid bounded training parameters")
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError("refuse overwrite of prior quality receipt")
    teacher,student=load_pair(args.teacher,args.student,args.config)
    if student.cfg.moe_experts < 2:
        raise ValueError("complete signed-MoE student checkpoint required")
    stages=parse_stages(args.stages,teacher.cfg.n_layers)
    train_ids=np.load(args.train_ids,allow_pickle=False)
    heldout=np.load(args.heldout_ids,allow_pickle=False)
    if not (np.issubdtype(train_ids.dtype,np.integer) and
            np.issubdtype(heldout.dtype,np.integer)):
        raise ValueError("token IDs require integer dtype")
    if (train_ids.shape[1]>teacher.cfg.block_size or
        heldout.shape[1]>teacher.cfg.block_size or
        np.any(train_ids<0) or np.any(heldout<0) or
        np.any(train_ids>=teacher.cfg.vocab_size) or
        np.any(heldout>=teacher.cfg.vocab_size)):
        raise ValueError("token IDs outside checkpoint bounds")
    train_ids=np.asarray(train_ids[:args.max_rows])
    fit,dev,fit_indices,dev_indices=split_train_calibration(
        train_ids,heldout,args.seed,args.calibration_fraction)
    torch.manual_seed(args.seed)
    device=torch.device("cuda:0")
    teacher=teacher.to(device).eval()
    model=build_guarded_hybrid(teacher,student).to(device).eval()
    with torch.no_grad():
        for block in model.blocks:
            block.quality_gate.zero_()
    with torch.no_grad():
        teacher_dev=token_cross_entropy(teacher,dev,batch=args.batch,device=device)["ce_nats"]
        teacher_heldout=None  # Don't inspect heldout before the final model.
    records=[]
    start=time.perf_counter()
    for stage_number,active in enumerate(stages,1):
        # Optimise the cumulative active student set. Dense layers outside
        # active are frozen and remain source teacher FFNs at rollout.
        active_set=set(active)
        trainable=[]
        for i,block in enumerate(model.blocks):
            for p in block.moe.parameters():
                p.requires_grad_(i in active_set and p is not block.moe.orientation.bias)
                if p.requires_grad:trainable.append(p)
            for p in block.even_student.parameters():
                p.requires_grad_(i in active_set)
                if p.requires_grad:trainable.append(p)
            with torch.no_grad():
                block.quality_gate.zero_()
        optimiser=torch.optim.AdamW(trainable,lr=args.lr)
        best_ce=float("inf")
        best_state=None
        stage_log=[]
        meter=TrainingProgress(
            label="curriculum-stage-%d"%stage_number,device=device,
            total_steps=args.epochs*((len(fit)+args.batch-1)//args.batch),
            progress_every=args.progress_every,
            jsonl=args.output.with_name(args.output.stem+".stage%d.progress.jsonl"%stage_number))
        for epoch in range(args.epochs):
            model.train()
            for module in model.modules():
                if isinstance(module,torch.nn.Dropout):module.eval()
            permutation=np.random.default_rng(args.seed+1009*stage_number+epoch).permutation(len(fit))
            for offset in range(0,len(fit),args.batch):
                ids=torch.as_tensor(fit[permutation[offset:offset+args.batch]],
                                    device=device,dtype=torch.long)
                optimiser.zero_grad(set_to_none=True)
                with torch.no_grad():
                    teacher_logits=teacher(ids)[0].detach()
                # Local source teacher-input MSE from existing guarded hybrid.
                model(ids)
                local=torch.stack([
                    model.blocks[i].last_local_distillation_loss for i in active]).mean()
                # Real end-to-end next-token gradient through active signed experts.
                logits=model(ids,student_only=True,student_only_layers=active)[0]
                student_ce=F.cross_entropy(
                    logits[:,:-1,:].reshape(-1,logits.shape[-1]),
                    ids[:,1:].reshape(-1))
                t_probs=F.softmax(teacher_logits[:,:-1,:],dim=-1)
                kl=F.kl_div(
                    F.log_softmax(logits[:,:-1,:],dim=-1),
                    t_probs,reduction="batchmean")/max(1,ids.shape[1]-1)
                loss=(args.student_ce_weight*student_ce+
                      args.teacher_kl_weight*kl+
                      args.local_mse_weight*local)
                if not bool(torch.isfinite(loss)):
                    raise ValueError("nonfinite native student objective")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable,1.)
                optimiser.step()
                meter.step(rows=ids.shape[0],sequence_length=ids.shape[1],
                           epoch=epoch+1,loss=float(loss.detach()))
            model.eval()
            with torch.no_grad():
                dev_ce=student_rollout_cross_entropy(
                    model,dev,batch=args.batch,device=device,student_only_layers=active)
            if dev_ce is None or not np.isfinite(dev_ce):
                raise ValueError("nonfinite curriculum development CE")
            if dev_ce<best_ce:
                best_ce=dev_ce
                # Snapshot only the mutable student tensors, never teacher.
                best_state={k:v.detach().cpu().clone()
                    for k,v in model.state_dict().items()
                    if ".moe." in k or ".even_student." in k}
                best_epoch=epoch+1
            stage_log.append({"epoch":epoch+1,"focused_dev_ce":dev_ce,
                              "teacher_dev_ce":teacher_dev})
            print(json.dumps({"stage":stage_number,"active_layers":active,
                "epoch":epoch+1,"focused_development_ce":dev_ce,
                "best_development_ce":best_ce}),flush=True)
        # Restore the epoch that best predicts the active focused branch.
        full_state=model.state_dict()
        for key,value in best_state.items():
            full_state[key]=value.to(full_state[key].device)
        model.load_state_dict(full_state,strict=True)
        records.append({"stage":stage_number,"active_layers":active,
                        "best_epoch":best_epoch,"best_development_ce":best_ce,
                        "epoch_scores":stage_log,"performance":meter.summary()})
    model.eval()
    # The independent heldout is inspected only once, after all training and
    # all stage-level selection. Never use this value for epoch/stage selection.
    with torch.no_grad():
        heldout_teacher=token_cross_entropy(
            teacher,heldout,batch=args.batch,device=device)["ce_nats"]
        heldout_focused=student_rollout_cross_entropy(
            model,heldout,batch=args.batch,device=device,
            student_only_layers=stages[-1])
        heldout_full=student_rollout_cross_entropy(
            model,heldout,batch=args.batch,device=device)
    if heldout_focused is None or heldout_full is None:
        raise ValueError("nonfinite heldout student evaluation")
    receipt={
        "schema":1,"kind":"native_gpu_cumulative_student_quality_curriculum",
        "source_teacher_sha256":sha256_file(args.teacher),
        "initial_student_sha256":sha256_file(args.student),
        "train_ids_sha256":sha256_file(args.train_ids),
        "heldout_ids_sha256":sha256_file(args.heldout_ids),
        "source_heldout_scored_once_after_stage_selection":True,
        "train_fit_indices":fit_indices,"train_development_indices":dev_indices,
        "fit_rows":len(fit),"development_rows":len(dev),"heldout_rows":len(heldout),
        "stages":records,"all_layers_active":len(stages[-1])==teacher.cfg.n_layers,
        "teacher_development_ce":teacher_dev,
        "teacher_heldout_ce":heldout_teacher,
        "focused_student_heldout_ce":heldout_focused,
        "full_student_only_heldout_ce":heldout_full,
        "full_student_score_computed_without_dense_ffn_calls":True,
        "focused_score_retains_dense_ffns_outside_active_layers":True,
        "full_student_quality_pass":bool(
            len(stages[-1])==teacher.cfg.n_layers and
            heldout_full<=heldout_teacher+args.quality_budget),
        "native_gpu":True,"gpu":torch.cuda.get_device_name(device),
        "hip":torch.version.hip,"elapsed_seconds":time.perf_counter()-start,
        "retained_dense_teacher_ffns_in_export":True,
        "release_admitted":False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"model_state_dict":model.cpu().state_dict(),
                "config":dataclasses.asdict(model.cfg),
                "quality_curriculum_receipt":receipt},args.output)
    args.output.with_suffix(".json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps({k:v for k,v in receipt.items() if k!="stages"},indent=2))
    return receipt

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher",type=Path,required=True)
    p.add_argument("--student",type=Path,required=True)
    p.add_argument("--config",type=Path)
    p.add_argument("--train-ids",type=Path,required=True)
    p.add_argument("--heldout-ids",type=Path,required=True)
    p.add_argument("--stages",required=True,
        help="strictly cumulative groups e.g. '1;0,1;0,1,2'")
    p.add_argument("--epochs",type=int,default=3)
    p.add_argument("--batch",type=int,default=1)
    p.add_argument("--max-rows",type=int,default=4096)
    p.add_argument("--seed",type=int,default=41)
    p.add_argument("--calibration-fraction",type=float,default=.15)
    p.add_argument("--lr",type=float,default=3e-4)
    p.add_argument("--student-ce-weight",type=float,default=1.)
    p.add_argument("--teacher-kl-weight",type=float,default=1.)
    p.add_argument("--local-mse-weight",type=float,default=1.)
    p.add_argument("--quality-budget",type=float,default=.02)
    p.add_argument("--progress-every",type=int,default=5)
    p.add_argument("--device",choices=("cuda","cuda:0"),default="cuda")
    p.add_argument("--output",type=Path,required=True)
    train(p.parse_args())
if __name__=="__main__":
    main()
