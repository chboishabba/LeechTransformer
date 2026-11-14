"""Q2.3/Q2.4 real C2 even-readout forward-only Dust vs SPSA vs AdamW.

ALL arms start from identical trained 4.763 student state, change ONLY one
source nn.Linear even readout, train on exactly the same fit windows/order,
and select using train-derived development rows. No independent heldout scan.
This is restricted Dust activation credit, not full temporal Dust attention.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import time
import math

import numpy as np
import torch
import torch.nn.functional as F
from scripts.reproduce_c2_quality_and_layers import load_trained
from scripts.verify_trained_federated_moe import checkpoint_parts
from scripts.train_guarded_moe_continuation import student_rollout_cross_entropy,sha256_file

def target_readout(model,layer):
    if not 0<=layer<len(model.blocks):
        raise ValueError("invalid source layer")
    head=model.blocks[layer].even_student
    if not isinstance(head,torch.nn.Sequential) or not isinstance(head[-1],torch.nn.Linear):
        raise ValueError("original C2 even projection not a final Linear")
    return head[-1]

def batch_loss(model,ids):
    logits=model(ids,student_only=True)[0]
    logits=logits[:,:-1,:]
    labels=ids[:,1:]
    per_token=F.cross_entropy(logits.transpose(1,2),labels,reduction="none")
    return per_token

def probe_source_future(centered,gamma):
    """Loss change at s can affect perturbation at t<=s; geometric future credit."""
    if not 0<=gamma<=1:
        raise ValueError("discount outside [0,1]")
    future=torch.zeros_like(centered)
    running=torch.zeros_like(centered[...,0])
    for t in range(centered.shape[-1]-1,-1,-1):
        running=centered[...,t]+gamma*running
        future[...,t]=running
    return future

@torch.no_grad()
def dust_step(model,linear,ids,*,lr,sigma,k,gamma,antithetic,generator):
    if k<2 or sigma<=0 or lr<=0 or (antithetic and k%2):
        raise ValueError("invalid source activation population")
    captured=[]
    perturb=[None]
    def before(mod,inputs):
        if not captured:captured.append(inputs[0].detach().clone())
    def after(mod,inputs,value):
        noise=perturb[0]
        return value if noise is None else value+sigma*noise
    hook_in=linear.register_forward_pre_hook(before)
    hook_out=linear.register_forward_hook(after)
    calls=0
    try:
        baseline=batch_loss(model,ids)
        calls+=1
        x=captured[0]
        b,t,output_width=ids.shape[0],ids.shape[1],linear.out_features
        if x.shape!=(b,t,linear.in_features):
            raise AssertionError("upstream even projection activation is not source-shaped")
        if antithetic:
            initial=torch.randn((k//2,b,t,output_width),
                generator=generator,device=ids.device,dtype=x.dtype)
            probes=torch.cat((initial,-initial),dim=0)
        else:
            probes=torch.randn((k,b,t,output_width),generator=generator,
                               device=ids.device,dtype=x.dtype)
        noisy=[]
        for j in range(k):
            perturb[0]=probes[j]
            noisy.append(batch_loss(model,ids))
            calls+=1
        perturb[0]=None
        losses=torch.stack(noisy,dim=0)
        centered=losses.mean(dim=0,keepdim=True)-losses
        future=probe_source_future(centered,gamma)
        # Valid source output times have next-token labels; final time has none.
        gradient=(-future.unsqueeze(-1)*probes[:,:,:-1,:]).mean(dim=0)/sigma
        x_valid=x[:,:-1]
        dweight=torch.einsum("btd,bth->dh",gradient,x_valid)/(b*(t-1))
        dbias=gradient.mean(dim=(0,1))
        if not bool(torch.isfinite(dweight).all() and torch.isfinite(dbias).all()):
            raise ValueError("nonfinite actual forward-only credit")
        linear.weight.add_(dweight,alpha=-lr)
        if linear.bias is not None:linear.bias.add_(dbias,alpha=-lr)
        return {"train_ce_before":float(baseline.mean()),
                "gradient_norm":float(torch.linalg.vector_norm(dweight)),
                "forward_calls":calls,"backward_calls":0,
                "probe_scheme":"antithetic" if antithetic else "independent",
                "credit_scope":"population_centered_immediate_and_discounted_future"}
    finally:
        hook_in.remove();hook_out.remove()

@torch.no_grad()
def spsa_step(model,linear,ids,*,lr,sigma,k,generator):
    if k<2 or k%2 or sigma<=0:
        raise ValueError("SPSA needs positive even forward-probe budget")
    calls=0
    w=linear.weight.detach()
    bias=linear.bias
    grad=torch.zeros_like(w)
    gbias=torch.zeros_like(bias) if bias is not None else None
    for _ in range(k//2):
        dw=torch.randint(0,2,w.shape,device=w.device,
                         generator=generator,dtype=torch.int64).to(w.dtype)*2-1
        db=(torch.randint(0,2,bias.shape,device=bias.device,
                          generator=generator,dtype=torch.int64).to(bias.dtype)*2-1
            if bias is not None else None)
        w.add_(dw,alpha=sigma)
        if db is not None:bias.add_(db,alpha=sigma)
        plus=float(batch_loss(model,ids).mean());calls+=1
        w.add_(dw,alpha=-2*sigma)
        if db is not None:bias.add_(db,alpha=-2*sigma)
        minus=float(batch_loss(model,ids).mean());calls+=1
        w.add_(dw,alpha=sigma)
        if db is not None:bias.add_(db,alpha=sigma)
        scalar=(plus-minus)/(2*sigma)
        grad+=scalar*dw/(k//2)
        if db is not None:gbias+=scalar*db/(k//2)
    if not bool(torch.isfinite(grad).all()):
        raise ValueError("nonfinite SPSA update")
    w.add_(grad,alpha=-lr)
    if gbias is not None:bias.add_(gbias,alpha=-lr)
    return {"gradient_norm":float(torch.linalg.vector_norm(grad)),
            "forward_calls":calls,"backward_calls":0,
            "probe_scheme":"weight_spsa_central"}

def adam_step(model,linear,ids,opt):
    opt.zero_grad(set_to_none=True)
    loss=batch_loss(model,ids).mean()
    if not bool(torch.isfinite(loss)):raise ValueError("nonfinite AdamW loss")
    loss.backward()
    g=linear.weight.grad
    if g is None:raise AssertionError("source even readout did not receive gradient")
    norm=float(torch.linalg.vector_norm(g.detach()))
    opt.step()
    return {"train_ce_before":float(loss.detach()),"gradient_norm":norm,
            "forward_calls":1,"backward_calls":1,
            "probe_scheme":"reverse_autodiff"}

def run(a):
    if a.device not in ("cuda","cuda:0") or not torch.cuda.is_available():
        raise ValueError("native GPU required; no CPU training fallback")
    if (a.steps<1 or a.batch<1 or a.probes<2 or a.probes%2 or
        a.sigma<=0 or a.lr<=0 or a.max_rows<2):
        raise ValueError("invalid matched training budget")
    if a.output.exists():raise FileExistsError("refuse overwrite comparison")
    receipt=json.loads(a.curriculum_receipt.read_text())
    for file,key in ((a.teacher,"source_teacher_sha256"),
                     (a.train_ids,"train_ids_sha256"),
                     (a.heldout_ids,"heldout_ids_sha256")):
        if sha256_file(file)!=receipt.get(key):
            raise ValueError("original training source SHA mismatch: "+key)
    _,base=load_trained(a.candidate,a.teacher,a.config)
    fit_ids=np.load(a.train_ids,allow_pickle=False)
    split=receipt["train_fit_indices"]
    if any(i<0 or i>=len(fit_ids) for i in split) or not split:
        raise ValueError("invalid original fit index receipt")
    fit=fit_ids[split][:a.max_rows]
    dev_ids=receipt["train_development_indices"]
    dev=fit_ids[dev_ids]
    if {r.tobytes() for r in fit}&{r.tobytes() for r in dev}:
        raise ValueError("fit and development token leakage")
    if len(fit)<2 or len(dev)<1:raise ValueError("insufficient original training data")
    target_readout(base,a.layer)
    original_state={key:value.detach().cpu().clone()
                    for key,value in base.state_dict().items()}
    outcomes=[]
    # All arms use identical source fit batches in identical order, to avoid
    # comparing unrelated compute or a changed initial model.
    order=np.random.default_rng(a.seed).integers(
        low=0,high=len(fit),size=a.steps*a.batch)
    for method in ("adamw","dust","dust_antithetic","spsa"):
        base.load_state_dict(original_state,strict=True)
        model=base.to(a.device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
            p.grad=None  # Do not carry AdamW gradient buffers into forward-only VRAM accounting.
        linear=target_readout(model,a.layer)
        if method=="adamw":
            for p in linear.parameters():p.requires_grad_(True)
        optimiser=(torch.optim.AdamW(linear.parameters(),lr=a.lr)
                   if method=="adamw" else None)
        generator=torch.Generator(device=a.device);generator.manual_seed(a.seed+17)
        torch.cuda.synchronize(a.device)
        torch.cuda.reset_peak_memory_stats(a.device)
        start=time.perf_counter()
        baseline=student_rollout_cross_entropy(
            model,dev,batch=a.batch,device=a.device)
        trace=[]
        calls=0;backs=0
        for step in range(a.steps):
            ids=torch.as_tensor(
                fit[order[step*a.batch:(step+1)*a.batch]],
                device=a.device,dtype=torch.long)
            if method=="adamw":
                result=adam_step(model,linear,ids,optimiser)
            elif method=="spsa":
                result=spsa_step(model,linear,ids,lr=a.lr,
                     sigma=a.sigma,k=a.probes,generator=generator)
            else:
                result=dust_step(model,linear,ids,lr=a.lr,
                    sigma=a.sigma,k=a.probes,gamma=a.gamma,
                    antithetic=(method=="dust_antithetic"),generator=generator)
            calls+=result["forward_calls"];backs+=result["backward_calls"]
            torch.cuda.synchronize(a.device)
            trace.append({"step":step+1,**result,
                "elapsed_seconds":time.perf_counter()-start})
        final=student_rollout_cross_entropy(
            model,dev,batch=a.batch,device=a.device)
        torch.cuda.synchronize(a.device)
        seconds=time.perf_counter()-start
        outcomes.append({"method":method,"initial_development_ce":baseline,
            "final_development_ce":final,
            "development_ce_improvement":baseline-final,
            "steps":a.steps,"train_forward_calls":calls,
            "train_backward_calls":backs,
            "train_input_token_evaluations":calls*a.batch*fit.shape[1],
            "train_target_token_evaluations":calls*a.batch*(fit.shape[1]-1),
            "total_elapsed_including_dev_seconds":seconds,
            "developer_note":"Uses full-model forward per activation probe; caches not yet implemented",
            "pytorch_gpu_peak_allocated_bytes":torch.cuda.max_memory_allocated(a.device),
            "trace":trace})
        model=model.to("cpu")
    report={"kind":"original_c2_even_readout_forward_only_competition",
            "schema":1,"source_checkpoint_sha256":sha256_file(a.candidate),
            "original_train_sha256":sha256_file(a.train_ids),
            "original_heldout_sha256":sha256_file(a.heldout_ids),
            "layer":a.layer,"steps_per_arm":a.steps,
            "source_fit_rows":len(fit),"source_dev_rows":len(dev),
            "probes":a.probes,"sigma":a.sigma,"gamma":a.gamma,
            "learning_rate":a.lr,"all_arms_same_initial_checkpoint":True,
            "all_arms_same_fit_batches":True,"heldout_scored":False,
            "trained_parameter_scope":"even_student_final_linear_weight_and_bias_only",
            "source_dust_kind":"restricted_activation_credit_not_full_attention_or_temporal_dust",
            "outcomes":outcomes,"release_admitted":False}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({**{k:v for k,v in report.items() if k!="outcomes"},
       "results":[{k:v for k,v in o.items() if k!="trace"} for o in outcomes]},
       indent=2,allow_nan=False))
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path)
    p.add_argument("--candidate",required=True,type=Path)
    p.add_argument("--teacher",required=True,type=Path)
    p.add_argument("--train-ids",required=True,type=Path)
    p.add_argument("--heldout-ids",required=True,type=Path)
    p.add_argument("--curriculum-receipt",required=True,type=Path)
    p.add_argument("--layer",type=int,required=True)
    p.add_argument("--steps",type=int,default=8)
    p.add_argument("--batch",type=int,default=1)
    p.add_argument("--max-rows",type=int,default=256)
    p.add_argument("--probes",type=int,default=4)
    p.add_argument("--sigma",type=float,default=.01)
    p.add_argument("--gamma",type=float,default=0.)
    p.add_argument("--lr",type=float,default=1e-4)
    p.add_argument("--seed",type=int,default=41)
    p.add_argument("--device",default="cuda")
    p.add_argument("--output",required=True,type=Path)
    run(p.parse_args())
if __name__=="__main__":main()
