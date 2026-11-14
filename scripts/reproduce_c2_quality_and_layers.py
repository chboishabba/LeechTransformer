"""Q2.1/Q2.2 source checkpoint: numerical CE, layer substitutions, routing/history.
Never train or mutate original checkpoint. Development selection is train-derived.
Independent heldout is reporting-only. Original LeechTransformer attribution kept.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from models.dense_to_signed_experts import canonical_even_input
from scripts.verify_trained_federated_moe import checkpoint_parts, infer_config
from models.model import LeechGPT
from scripts.diagnose_moe_quality_collapse import token_cross_entropy
from scripts.train_guarded_moe_continuation import student_rollout_cross_entropy,sha256_file

def load_trained(candidate,teacher):
    state,info=checkpoint_parts(candidate)
    cfg=infer_config(state,info,None)
    if not cfg.moe_guarded_ffn or cfg.moe_experts<2:
        raise ValueError("guarded full C2 curriculum candidate required")
    model=LeechGPT(cfg).eval()
    model.load_state_dict(state,strict=True)
    ts,ti=checkpoint_parts(teacher)
    tc=infer_config(ts,ti,None)
    if tc.moe_experts or (tc.d_model,tc.n_layers,tc.vocab_size)!=(
        cfg.d_model,cfg.n_layers,cfg.vocab_size):
        raise ValueError("original dense teacher architecture mismatch")
    dense=LeechGPT(tc).eval()
    dense.load_state_dict(ts,strict=True)
    return dense,model

def numeric_ce(model,rows,*,batch,device,mode):
    target_logprob_sum=0.
    targets=0
    hash_object=hashlib.sha256()
    with torch.no_grad():
        for start in range(0,len(rows),batch):
            ids=torch.as_tensor(np.asarray(rows[start:start+batch]),
                                device=device,dtype=torch.long)
            logits=model(ids,student_only=(mode=="student"))[0] if mode=="student" else model(ids)[0]
            lp=F.log_softmax(logits[:,:-1,:].double(),dim=-1)
            chosen=lp.gather(-1,ids[:,1:].unsqueeze(-1)).squeeze(-1)
            if not bool(torch.isfinite(chosen).all()):
                raise ValueError("nonfinite token log probability")
            target_logprob_sum-=float(chosen.sum())
            targets+=chosen.numel()
            hash_object.update(chosen.detach().cpu().contiguous().numpy().astype("<f8").tobytes())
    return {"ce_nats":target_logprob_sum/targets,
            "target_tokens":targets,
            "token_log_probability_sha256_fp64":hash_object.hexdigest(),
            "log_probability_accumulation":"float64_log_softmax_per_sample"}

@torch.no_grad()
def diagnose(model,teacher,rows,*,batch,device,stage_groups=None,
             route_margin_tolerance=1e-3):
    model.eval();teacher.eval()
    stage_groups=stage_groups or []
    source=token_cross_entropy(teacher,rows,batch=batch,device=device)["ce_nats"]
    single=[];cumulative=[]
    for i in range(len(model.blocks)):
        ce=student_rollout_cross_entropy(model,rows,batch=batch,device=device,
                                         student_only_layers=(i,))
        single.append({"layer":i,"focused_ce":ce,"delta_vs_teacher":ce-source})
    for active in stage_groups:
        if not active:continue
        active=tuple(sorted(set(active)))
        ce=student_rollout_cross_entropy(model,rows,batch=batch,device=device,
                                         student_only_layers=active)
        cumulative.append({"layers":list(active),"focused_ce":ce,
                           "delta_vs_teacher":ce-source})
    # Teacher inputs are intentionally held fixed to distinguish local FFN
    # approximation from recursively accumulated student activation drift.
    local=[]
    for layer in range(len(model.blocks)):
        moe=model.blocks[layer].moe
        sums={"teacher_sq":0.,"student_sq":0.,"err_sq":0.,
              "odd_sq":0.,"even_sq":0.,"cross":0.,
              "history_sq":0.,"history_max":0.,"margin_sum":0.,
              "near_ties":0,"routes":0,"n":0}
        counts=torch.zeros(moe.num_experts,dtype=torch.int64)
        for start in range(0,len(rows),batch):
            ids=torch.as_tensor(np.asarray(rows[start:start+batch]),
                                device=device,dtype=torch.long)
            capture=[]
            hook=teacher.blocks[layer].ln2.register_forward_hook(
                lambda module,inp,out:capture.append(out.detach()))
            try:teacher(ids)
            finally:hook.remove()
            x=capture[-1]
            reference=teacher.blocks[layer].ffn(x)
            recorded=[]
            def router_hook(module,inp,out):
                recorded.append(out.detach())
            rh=moe.router.register_forward_hook(router_hook)
            try:odd,final_h,routed,signs,weights=moe(x)
            finally:rh.remove()
            even=model.blocks[layer].even_student(canonical_even_input(x))
            full=odd+even
            delta=full-reference
            sums["teacher_sq"]+=float(reference.double().square().sum())
            sums["student_sq"]+=float(full.double().square().sum())
            sums["err_sq"]+=float(delta.double().square().sum())
            sums["odd_sq"]+=float(odd.double().square().sum())
            sums["even_sq"]+=float(even.double().square().sum())
            sums["cross"]+=float((odd.double()*even.double()).sum())
            sums["history_sq"]+=float(final_h.double().square().sum())
            sums["history_max"]=max(sums["history_max"],float(final_h.abs().max()))
            sums["n"]+=reference.numel()
            counts+=torch.bincount(routed.detach().cpu().reshape(-1),minlength=moe.num_experts)
            for score in recorded:
                score=score.squeeze(-1)
                if score.ndim!=2 or score.shape[1]!=moe.num_experts:
                    raise AssertionError("source router scores shape differs")
                sorted_scores=torch.sort(score,dim=-1,descending=True).values
                margin=sorted_scores[:,moe.top_k-1]-sorted_scores[:,moe.top_k] if moe.top_k<moe.num_experts else torch.full_like(sorted_scores[:,0],float("inf"))
                finite=margin[torch.isfinite(margin)]
                sums["margin_sum"]+=float(finite.sum()) if finite.numel() else 0.
                sums["near_ties"]+=int((finite<=route_margin_tolerance).sum())
                sums["routes"]+=int(finite.numel())
        total=int(counts.sum())
        probabilities=(counts.double()/total) if total else torch.zeros_like(counts,dtype=torch.float64)
        entropy=float(-(probabilities[probabilities>0]*probabilities[probabilities>0].log()).sum())
        local.append({"layer":layer,"teacher_input_ffn_rmse":math.sqrt(sums["err_sq"]/sums["n"]),
           "teacher_input_relative_mse":sums["err_sq"]/max(sums["teacher_sq"],1e-12),
           "odd_rms":math.sqrt(sums["odd_sq"]/sums["n"]),
           "even_rms":math.sqrt(sums["even_sq"]/sums["n"]),
           "odd_even_cross_mean":sums["cross"]/sums["n"],
           "student_rms":math.sqrt(sums["student_sq"]/sums["n"]),
           "route_counts":counts.tolist(),"route_entropy_nats":entropy,
           "topk_boundary_mean_margin":sums["margin_sum"]/sums["routes"] if sums["routes"] else None,
           "topk_near_tie_fraction":sums["near_ties"]/sums["routes"] if sums["routes"] else None,
           "history_final_rms":math.sqrt(sums["history_sq"]/max(1,len(rows)*moe.num_experts*moe.d_model)),
           "history_final_max_abs":sums["history_max"]})
    return {"teacher_development_ce":source,"single_layer":single,
            "cumulative":cumulative,"local_teacher_input":local,
            "source_note":"Teacher-forced local MSE is not a causal model-layer attribution; routed final history is separately observed."}

def run(a):
    if a.output.exists():raise FileExistsError("refuse overwrite diagnostic")
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("native CUDA/HIP device unavailable")
    receipt=json.loads(Path(a.curriculum_receipt).read_text())
    if receipt.get("kind")!="native_gpu_cumulative_student_quality_curriculum":
        raise ValueError("wrong curriculum receipt")
    for path,key in ((a.candidate,None),(a.teacher,"source_teacher_sha256"),
                     (a.train_ids,"train_ids_sha256"),
                     (a.heldout_ids,"heldout_ids_sha256")):
        if key and sha256_file(path)!=receipt.get(key):
            raise ValueError("source SHA mismatch: "+key)
    train=np.load(a.train_ids,allow_pickle=False)
    held=np.load(a.heldout_ids,allow_pickle=False)
    dev_indices=receipt["train_development_indices"]
    if not dev_indices or any(i<0 or i>=len(train) for i in dev_indices):
        raise ValueError("invalid source development indices")
    dev=train[dev_indices]
    teacher,model=load_trained(a.candidate,a.teacher)
    if any(block.even_student is None for block in model.blocks):
        raise ValueError("even C2 channel missing")
    if np.any(dev<0) or np.any(dev>=model.cfg.vocab_size):
        raise ValueError("invalid development tokens")
    device=torch.device(a.device)
    teacher=teacher.to(device).eval()
    model=model.to(device).eval()
    stage_groups=[s["active_layers"] for s in receipt["stages"]]
    started=time.perf_counter()
    diagnostic=diagnose(model,teacher,dev,batch=a.batch,device=device,
         stage_groups=stage_groups,route_margin_tolerance=a.margin_tolerance)
    # Numeric CE recomputation on independent heldout is reporting ONLY;
    # no optimiser or checkpoint selection occurs in this script.
    held_teacher=numeric_ce(teacher,held,batch=a.batch,device=device,mode="teacher")
    held_student=numeric_ce(model,held,batch=a.batch,device=device,mode="student")
    report={"kind":"original_c2_source_quality_reproduction","schema":1,
        "checkpoint_sha256":sha256_file(a.candidate),
        "teacher_sha256":sha256_file(a.teacher),"heldout_sha256":sha256_file(a.heldout_ids),
        "dev_rows":len(dev),"heldout_rows":len(held),
        "numeric_teacher_heldout":held_teacher,"numeric_student_heldout":held_student,
        "original_teacher_reported_ce":receipt["teacher_heldout_ce"],
        "original_student_reported_ce":receipt["full_student_only_heldout_ce"],
        "teacher_ce_abs_delta_vs_original_receipt":abs(held_teacher["ce_nats"]-receipt["teacher_heldout_ce"]),
        "student_ce_abs_delta_vs_original_receipt":abs(held_student["ce_nats"]-receipt["full_student_only_heldout_ce"]),
        "reproduced_at_tolerance":(
          abs(held_teacher["ce_nats"]-receipt["teacher_heldout_ce"])<=a.ce_tolerance and
          abs(held_student["ce_nats"]-receipt["full_student_only_heldout_ce"])<=a.ce_tolerance),
        "development_diagnostics":diagnostic,"gpu":str(device),
        "analysis_seconds":time.perf_counter()-started,
        "no_retraining":True,"heldout_used_for_selection":False,
        "release_admitted":False}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    if not report["reproduced_at_tolerance"]:
        raise AssertionError("source heldout CE discrepancy: receipt saved; inspect original evaluation precision")
    print(json.dumps({k:v for k,v in report.items() if k!="development_diagnostics"},indent=2))
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate",type=Path,required=True)
    p.add_argument("--teacher",type=Path,required=True)
    p.add_argument("--train-ids",type=Path,required=True)
    p.add_argument("--heldout-ids",type=Path,required=True)
    p.add_argument("--curriculum-receipt",type=Path,required=True)
    p.add_argument("--batch",type=int,default=1)
    p.add_argument("--margin-tolerance",type=float,default=.001)
    p.add_argument("--ce-tolerance",type=float,default=.005)
    p.add_argument("--device",default="cuda")
    p.add_argument("--output",required=True,type=Path)
    run(p.parse_args())
if __name__=="__main__":main()
