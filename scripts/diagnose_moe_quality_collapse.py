"""Localize genuine signed-MoE quality collapse before retraining.

Use the SAME held-out token IDs and the original strictly loaded
teacher/student checkpoints. No new weights, synthetic corpus, or
optimizer changes. Measures isolated dense-FFN->student substitutions,
cumulative depth replacement, same-input FFN MSE/scale, and routed
history amplification/entropy for each layer.

  python -m scripts.diagnose_moe_quality_collapse \
    --teacher data/best_model.pt \
    --student /tmp/real_text_moe_jq1_smoke/punctured9_wave/student_model.pt \
    --heldout /tmp/real_text_moe_jq1_smoke/dataset/heldout_ids.npy \
    --config /tmp/leech_dense_config.json \
    --output /tmp/quality_diagnosis_p9.json

This reports causes to investigate, NOT a certificate of semantic
quality or a new geometric identity. Only the specified experimental
layers are substituted. The reference teacher stays unmodified on
exit (including on exceptions).
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from models.model import LeechGPT
from models.dense_to_signed_experts import canonical_even_input
from scripts.verify_trained_federated_moe import checkpoint_parts,infer_config


def load_pair(teacher_path:Path,student_path:Path,config_path:Path|None):
    teacher_state,teacher_info=checkpoint_parts(teacher_path)
    student_state,student_info=checkpoint_parts(student_path)
    teacher_cfg=infer_config(teacher_state,teacher_info,config_path)
    student_cfg=infer_config(student_state,student_info,None)
    if teacher_cfg.moe_experts or student_cfg.moe_experts<2:
        raise ValueError("dense teacher and actual MoE student required")
    for key in ("vocab_size","d_model","n_layers","n_heads","block_size"):
        if getattr(teacher_cfg,key)!=getattr(student_cfg,key):
            raise ValueError("checkpoint architecture mismatch: "+key)
    teacher=LeechGPT(teacher_cfg).eval()
    teacher.load_state_dict(teacher_state,strict=True)
    student=LeechGPT(student_cfg).eval()
    student.load_state_dict(student_state,strict=True)
    if any(block.moe is None or block.even_student is None
           for block in student.blocks):
        raise ValueError("complete even+odd parity student required")
    return teacher,student


@contextmanager
def substituted_ffns(teacher,student,layers):
    """Exact architecture-level intervention; never mutate trained weights."""
    originals={}
    try:
        for layer in sorted(set(layers)):
            dense_block=teacher.blocks[layer]
            originals[layer]=(dense_block.moe,dense_block.even_student)
            dense_block.moe=student.blocks[layer].moe
            dense_block.even_student=student.blocks[layer].even_student
        yield teacher
    finally:
        for layer,(moe,even) in originals.items():
            teacher.blocks[layer].moe=moe
            teacher.blocks[layer].even_student=even


@torch.no_grad()
def token_cross_entropy(model,rows,*,batch=2,device="cpu"):
    model.to(device).eval()
    nll=0.
    tokens_count=0
    finite=True
    for start in range(0,len(rows),batch):
        t=torch.as_tensor(rows[start:start+batch],
                          device=device,dtype=torch.long)
        logits=model(t)[0]
        finite=finite and bool(torch.isfinite(logits).all())
        if not finite:
            return {"ce_nats":None,"all_logits_finite":False}
        labels=t[:,1:].reshape(-1)
        pred=logits[:,:-1,:].reshape(-1,logits.shape[-1])
        nll+=float(F.cross_entropy(pred,labels,reduction="sum"))
        tokens_count+=labels.numel()
    return {"ce_nats":nll/tokens_count if tokens_count else None,
            "all_logits_finite":finite,"target_tokens":tokens_count}


@torch.no_grad()
def teacher_forced_layer_diagnostics(teacher,student,tokens,*,device="cpu"):
    """Compare each candidate FFN on IDENTICAL teacher normalized inputs."""
    teacher.eval().to(device)
    student.eval().to(device)
    normalized={}
    hooks=[]
    for i,block in enumerate(teacher.blocks):
        hooks.append(block.ln2.register_forward_hook(
            lambda module,args,output,layer=i:
                normalized.__setitem__(layer,output.detach().clone())))
    try:
        teacher(tokens.to(device))
    finally:
        for h in hooks:h.remove()
    rows=[]
    for i,block in enumerate(teacher.blocks):
        x=normalized[i]
        reference=block.ffn(x)
        student_block=student.blocks[i]
        odd,history,ids,signs,weights=student_block.moe(x)
        even=student_block.even_student(canonical_even_input(x))
        actual=odd+even
        diff=actual-reference
        teacher_mse=float(reference.float().square().mean())
        error_mse=float(diff.float().square().mean())
        counts=torch.bincount(ids.reshape(-1),minlength=student.cfg.moe_experts)
        proportion=counts.float()/max(int(counts.sum()),1)
        entropy=-torch.where(proportion>0,
            proportion*proportion.clamp_min(1e-12).log(),
            torch.zeros_like(proportion)).sum()
        # History trace, rather than only final memory: important for a
        # recurrent signed MoE whose gates consume quadratic features.
        h=x.new_zeros(x.shape[0],student.cfg.moe_experts,x.shape[-1])
        max_history_rms=0.
        max_history_abs=0.
        max_route_weight=0.
        for time in range(x.shape[1]):
            _,h,_,_,w=student_block.moe.step(x[:,time],h)
            max_history_rms=max(max_history_rms,
                float(h.float().square().mean().sqrt()))
            max_history_abs=max(max_history_abs,float(h.float().abs().max()))
            max_route_weight=max(max_route_weight,float(w.max()))
        rows.append({
            "layer":i,
            "teacher_ffn_rms":teacher_mse**.5,
            "student_ffn_rms":float(actual.float().square().mean().sqrt()),
            "same_input_ffn_rmse":error_mse**.5,
            "same_input_relative_mse":error_mse/max(teacher_mse,1e-12),
            "route_counts":counts.tolist(),
            "route_entropy_nats":float(entropy),
            "route_entropy_max_nats":math.log(student.cfg.moe_experts),
            "final_history_rms":float(history.float().square().mean().sqrt()),
            "maximum_trace_history_rms":max_history_rms,
            "maximum_trace_history_abs":max_history_abs,
            "maximum_route_weight":max_route_weight,
            "sign_positive_fraction":float((signs>0).float().mean()),
            "sign_zero_fraction":float((signs==0).float().mean())})
    return rows


@torch.no_grad()
def diagnose(teacher,student,rows,*,batch=2,device="cpu",
             teacher_forced_rows=2,mode="both"):
    if rows.ndim!=2 or rows.shape[0]<1 or rows.shape[1]<2:
        raise ValueError("heldout input must be [N>=1,T>=2] token IDs")
    if mode not in ("isolated","prefix","both"):
        raise ValueError("unknown diagnostic mode")
    n=len(teacher.blocks)
    baseline=token_cross_entropy(teacher,rows,batch=batch,device=device)
    student_result=token_cross_entropy(student,rows,batch=batch,device=device)
    isolated=[]
    prefix=[]
    if mode in ("isolated","both"):
        for i in range(n):
            with substituted_ffns(teacher,student,[i]):
                entry=token_cross_entropy(teacher,rows,batch=batch,device=device)
            isolated.append({"layer":i,**entry,
                "delta_ce":(entry["ce_nats"]-baseline["ce_nats"]
                    if entry["ce_nats"] is not None else None)})
    if mode in ("prefix","both"):
        for last in range(n):
            with substituted_ffns(teacher,student,range(last+1)):
                entry=token_cross_entropy(teacher,rows,batch=batch,device=device)
            prefix.append({"substituted_layers":last+1,**entry,
                "delta_ce":(entry["ce_nats"]-baseline["ce_nats"]
                    if entry["ce_nats"] is not None else None)})
    sample=torch.as_tensor(rows[:teacher_forced_rows],dtype=torch.long)
    forced=teacher_forced_layer_diagnostics(teacher,student,sample,device=device)
    return {
        "dense_teacher":baseline,"full_moe_student":student_result,
        "isolated_layer_substitutions":isolated,
        "prefix_layer_substitutions":prefix,
        "same_teacher_input_layer_diagnostics":forced,
        "uniform_ce_nats":math.log(teacher.cfg.vocab_size),
        "configuration":{
            "layers":n,"d_model":teacher.cfg.d_model,
            "route_geometry":student.cfg.moe_route_geometry,
            "experts":student.cfg.moe_experts,
            "top_k":student.cfg.moe_top_k},
        "interpretation":"Isolated CE locates fragile layers; prefix CE locates compounding; teacher-input MSE/history traces distinguish fit and history instability. No geometry/semantic advantage is inferred."}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--teacher",required=True,type=Path)
    ap.add_argument("--student",required=True,type=Path)
    ap.add_argument("--config",type=Path)
    ap.add_argument("--heldout",required=True,type=Path)
    ap.add_argument("--device",default="cpu")
    ap.add_argument("--batch",type=int,default=2)
    ap.add_argument("--teacher-forced-rows",type=int,default=2)
    ap.add_argument("--mode",choices=("isolated","prefix","both"),default="both")
    ap.add_argument("--output",type=Path,default=Path("moe_quality_diagnosis.json"))
    a=ap.parse_args()
    if a.batch<1 or a.teacher_forced_rows<1:
        raise ValueError("invalid batch or diagnostic sample count")
    teacher,student=load_pair(a.teacher,a.student,a.config)
    heldout=np.load(a.heldout,allow_pickle=False)
    if (heldout.ndim!=2 or heldout.shape[0]<1 or
        heldout.shape[1]<2 or heldout.shape[1]>teacher.cfg.block_size or
        not np.issubdtype(heldout.dtype,np.integer) or
        np.any(heldout<0) or np.any(heldout>=teacher.cfg.vocab_size)):
        raise ValueError("invalid heldout token IDs")
    report=diagnose(teacher,student,heldout,
        batch=a.batch,device=a.device,
        teacher_forced_rows=a.teacher_forced_rows,mode=a.mode)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2))
    print(json.dumps({
        "report":str(a.output),
        "teacher_ce":report["dense_teacher"]["ce_nats"],
        "student_ce":report["full_moe_student"]["ce_nats"],
        "uniform_ce":report["uniform_ce_nats"],
        "worst_single_layer":max(
            report["isolated_layer_substitutions"],
            key=lambda row:row["delta_ce"] if row["delta_ce"] is not None else float("inf"))
            if report["isolated_layer_substitutions"] else None},indent=2))


if __name__=="__main__":
    main()
