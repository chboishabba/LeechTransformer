"""Quality-preserving, progressive teacher->signed-MoE continuation.

The original dense FFN is retained at every layer as a residual teacher.
At alpha=0 the hybrid is *exactly the dense network* (within floating
execution parity); the student is still trained by local FFN loss.
A calibrated alpha is admitted only if its CE on a separate calibration
slice stays within a declared tolerance of the dense baseline.

Train/calibration split comes ONLY from train_ids.npy. The independent
heldout_ids.npy is checked for initial dense parity and final quality,
never used to select gates. A separate student-only rollout opens an
end-to-end CE/KL gradient path into ALL trainable MoE layers even when
admission gates are still at zero. Accepted gates also train through
the calibrated hybrid path. Zero admitted gates means NO compression or
quality improvement; the original teacher is still present and must
not be reported as a distilled standalone model.

Example:
 python -m scripts.train_guarded_moe_continuation \
  --teacher data/best_model.pt --student student_model.pt \
  --config /tmp/leech_dense_config.json \
  --train-ids dataset/train_ids.npy \
  --heldout-ids dataset/heldout_ids.npy \
  --epochs 3 --calibration-budget-nats .02 \
  --output guarded_student.pt
"""
from __future__ import annotations
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model import LeechGPT
from scripts.diagnose_moe_quality_collapse import load_pair,token_cross_entropy


def sha256_file(path:Path):
    h=hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda:file.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()


def build_guarded_hybrid(teacher:LeechGPT,student:LeechGPT):
    if teacher.cfg.moe_experts or student.cfg.moe_experts<2:
        raise ValueError("guarded continuation requires dense teacher and MoE student")
    config=dataclasses.replace(student.cfg,moe_guarded_ffn=True)
    hybrid=LeechGPT(config)
    target=hybrid.state_dict()
    source=teacher.state_dict()
    candidate=student.state_dict()
    for key in (set(source)&set(candidate)):
        # Model-specific student weights are allowed to differ, but
        # every shared pretrained attention/embedding/norm weight MUST
        # remain the exact original, or comparisons are confounded.
        if ".moe." in key or ".even_student." in key or ".ffn." in key:
            continue
        if not torch.equal(source[key],candidate[key]):
            raise ValueError("student modified pretrained shared tensor: "+key)
    for key in target:
        if key.endswith(".quality_gate"):
            target[key]=torch.zeros_like(target[key])
        elif key in candidate:
            target[key]=candidate[key]
        elif key in source:
            target[key]=source[key]
        else:
            raise ValueError("uninitialized hybrid parameter "+key)
    hybrid.load_state_dict(target,strict=True)
    for parameter in hybrid.parameters():
        parameter.requires_grad_(False)
    for layer in hybrid.blocks:
        for p in layer.moe.parameters():p.requires_grad_(True)
        for p in layer.even_student.parameters():p.requires_grad_(True)
        layer.quality_gate.requires_grad_(False)
    return hybrid


def gates(model):
    return [float(block.quality_gate.detach()) for block in model.blocks]


@torch.no_grad()
def calibrate_gates(model,calibration,*,teacher_ce:float,budget:float,
                    batch:int,device:str,candidates=(0.,.02,.05,.1,.25,.5,1.),
                    focus_layers:tuple[int,...]|None=None):
    """Greedy one-layer-at-a-time gate opening, always fail-closed."""
    if budget<0:raise ValueError("negative CE budget")
    model.eval()
    # Previously accepted gates may become unsafe after an optimizer step.
    # Revoke them before testing the NEW student weight revision.
    previous_gates=gates(model)
    for block in model.blocks:
        block.quality_gate.fill_(0.)
    accepted=[]
    upper=teacher_ce+budget
    zero_ce=token_cross_entropy(model,calibration,batch=batch,
                                device=device)["ce_nats"]
    if zero_ce is None or not np.isfinite(zero_ce) or zero_ce>upper+1e-6:
        raise AssertionError("zero-gate teacher fallback failed calibration")
    for i,block in enumerate(model.blocks):
        if focus_layers is not None and i not in focus_layers:
            accepted.append({"layer":i,"previous":0.,"admitted":0.,
                             "attempted":[],"reason":"outside focused training"})
            continue
        baseline=0.
        admitted=baseline
        trial_scores=[]
        # Gate values only increase. If the new layer hurts, retain
        # previously accepted smaller gate; never promote on heldout.
        for alpha in candidates:
            if alpha<=baseline+1e-8:continue
            block.quality_gate.fill_(alpha)
            score=token_cross_entropy(model,calibration,batch=batch,
                                      device=device)["ce_nats"]
            trial_scores.append({"alpha":alpha,"ce":score})
            if score is not None and np.isfinite(score) and score<=upper:
                admitted=alpha
        block.quality_gate.fill_(admitted)
        accepted.append({"layer":i,"previous":baseline,
                         "admitted":admitted,"attempted":trial_scores})
    final_ce=token_cross_entropy(model,calibration,batch=batch,
                                 device=device)["ce_nats"]
    if final_ce is None or final_ce>upper+1e-6:
        raise AssertionError("calibrated gates violated declared CE budget")
    return {"dense_calibration_ce":teacher_ce,"allowed_ce":upper,
            "previous_revision_gates_revoked":previous_gates,
            "zero_gate_calibration_ce":zero_ce,
            "accepted":accepted,"actual_calibration_ce":final_ce}


@torch.no_grad()
def student_rollout_cross_entropy(hybrid,rows,*,batch=2,device="cpu",
                                  student_only_layers=None):
    """Measure hypothetical strict teacher-erased quality without opening gates."""
    hybrid.eval().to(device)
    nll=0.
    count=0
    for start in range(0,len(rows),batch):
        ids=torch.as_tensor(rows[start:start+batch],device=device,dtype=torch.long)
        logits=hybrid(ids,student_only=True,
                      student_only_layers=student_only_layers)[0]
        if not bool(torch.isfinite(logits).all()):
            return None
        nll+=float(F.cross_entropy(logits[:,:-1].reshape(-1,logits.shape[-1]),
                                   ids[:,1:].reshape(-1),reduction="sum"))
        count+=ids[:,1:].numel()
    return nll/count if count else None


def train_continuation(teacher,student,train,heldout,*,epochs:int,
                       seed:int=41,batch:int=2,lr:float=3e-4,
                       calibration_fraction:float=.125,
                       calibration_budget_nats:float=.02,
                       mse_weight:float=1.,
                       kl_weight:float=1.,
                       ce_weight:float=.25,
                       student_ce_weight:float=1.,
                       student_kl_weight:float=1.,
                       focus_layers:tuple[int,...]|None=None,
                       matched_control_objective:bool=False,
                       temperature:float=1.,
                       device:str="cpu", progress=None):
    if (epochs<1 or batch<1 or lr<=0 or temperature<=0 or
        mse_weight<=0 or kl_weight<0 or ce_weight<0 or
        student_ce_weight<0 or student_kl_weight<0 or
        student_ce_weight+student_kl_weight<=0 or
        not 0<calibration_fraction<.5):
        raise ValueError("invalid guarded continuation arguments")
    if (train.ndim!=2 or len(train)<8 or train.shape[1]<2 or
        heldout.ndim!=2 or len(heldout)<1 or heldout.shape[1]<2 or
        train.shape[1]>teacher.cfg.block_size or
        heldout.shape[1]>teacher.cfg.block_size):
        raise ValueError("invalid train/heldout sequences")
    if focus_layers is not None:
        if (not focus_layers or len(set(focus_layers))!=len(focus_layers)
                or any(not isinstance(i,int) or
                       not 0<=i<teacher.cfg.n_layers for i in focus_layers)):
            raise ValueError("focus_layers requires unique valid block indices")
    if matched_control_objective and (
            focus_layers is None or len(focus_layers)!=1 or
            temperature!=1. or mse_weight!=1. or student_ce_weight!=1.
            or student_kl_weight!=1.):
        raise ValueError(
            "matched_control_objective requires one focused layer, "
            "temperature=1 and unit local MSE, student CE/KL weights")
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    order=rng.permutation(len(train))
    calibration_count=max(2,int(len(train)*calibration_fraction))
    calibration_indices=order[:calibration_count]
    optimizer_fit_indices=order[calibration_count:]
    cal=train[calibration_indices]
    fit=train[optimizer_fit_indices]
    # Direct token-ID callers must not bypass split disjointness.
    if {row.tobytes() for row in train}.intersection(
            row.tobytes() for row in heldout):
        raise ValueError("train/heldout token windows overlap")
    hybrid=build_guarded_hybrid(teacher,student).to(device)
    if focus_layers is not None:
        for index,block in enumerate(hybrid.blocks):
            if index not in focus_layers:
                for param in block.moe.parameters():
                    param.requires_grad_(False)
                for param in block.even_student.parameters():
                    param.requires_grad_(False)
    teacher.to(device).eval()
    hybrid.eval()
    # Since the teacher includes Dropout modules, prohibit stochastic
    # mismatch while students train and while gates are evaluated.
    for module in hybrid.modules():
        if isinstance(module,nn.Dropout):
            module.eval()
    if progress:
        progress("measuring dense and zero-gate baselines")
    with torch.no_grad():
        dense_cal_ce=token_cross_entropy(teacher,cal,batch=batch,device=device)["ce_nats"]
        dense_held_ce=token_cross_entropy(teacher,heldout,batch=batch,device=device)["ce_nats"]
        zero_ce=token_cross_entropy(hybrid,heldout,batch=batch,device=device)["ce_nats"]
    if abs(zero_ce-dense_held_ce)>1e-5:
        raise AssertionError("zero-quality-gate hybrid does not reproduce frozen teacher")
    params=[p for p in hybrid.parameters() if p.requires_grad]
    optim=torch.optim.AdamW(params,lr=lr)
    history=[]
    for epoch in range(epochs):
        if matched_control_objective:
            # Equivalent teacher-input local FFN target on every update.
            # Previously calibrated gates are not allowed to contaminate
            # this matched one-layer training objective.
            with torch.no_grad():
                for block in hybrid.blocks:
                    block.quality_gate.fill_(0.)
        if progress:
            progress("epoch %d/%d: fitting %d rows" % (epoch + 1, epochs, len(fit)))
        hybrid.train()
        for module in hybrid.modules():
            if isinstance(module,nn.Dropout):module.eval()
        # Quality gate remains a manually calibrated scalar. This is
        # necessary to enforce the CE admission contract.
        row_order=np.random.default_rng(seed+epoch).permutation(len(fit))
        stats=[]
        for start in range(0,len(fit),batch):
            if progress and (start == 0 or start + batch >= len(fit) or
                             (start // batch) % 10 == 0):
                progress("epoch %d/%d: batch %d/%d" % (
                    epoch + 1, epochs, min(start // batch + 1, (len(fit) + batch - 1) // batch),
                    (len(fit) + batch - 1) // batch))
            ids=torch.as_tensor(fit[row_order[start:start+batch]],
                                device=device,dtype=torch.long)
            optim.zero_grad(set_to_none=True)
            with torch.no_grad():
                old=teacher(ids)[0]
            logits=hybrid(ids)[0]
            if matched_control_objective:
                local=hybrid.blocks[focus_layers[0]].last_local_distillation_loss
            else:
                local=sum(block.last_local_distillation_loss
                          for block in hybrid.blocks)/len(hybrid.blocks)
            # The zero-gate hybrid cannot carry CE gradients through the
            # student. A separate full student rollout fixes that deadlock
            # without opening or mutating ANY deployed quality gate.
            student_logits=hybrid(
                ids,student_only=True,student_only_layers=focus_layers)[0]
            shifted=logits[:,:-1].reshape(-1,logits.shape[-1])
            targets=ids[:,1:].reshape(-1)
            cross_entropy=F.cross_entropy(shifted,targets)
            old_probs=F.softmax(old.detach()/temperature,dim=-1)
            log_probs=F.log_softmax(logits/temperature,dim=-1)
            kl=F.kl_div(log_probs,old_probs,reduction="none").sum(-1).mean()*temperature**2
            student_ce=F.cross_entropy(
                student_logits[:,:-1].reshape(-1,student_logits.shape[-1]),targets)
            student_kl=F.kl_div(
                F.log_softmax(student_logits/temperature,dim=-1),
                old_probs,reduction="none").sum(-1).mean()*temperature**2
            if matched_control_objective:
                # Same terms and unit weights as train_early_layer_dense_control:
                # teacher-input local MSE + one actual focused-student CE/KL.
                # The separate guarded hybrid has its own calibration path.
                loss=(local+student_ce+student_kl)
            else:
                loss=(mse_weight*local+kl_weight*kl+
                      ce_weight*cross_entropy+student_ce_weight*student_ce+
                      student_kl_weight*student_kl)
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite train loss: rejecting geometric student")
            loss.backward()
            nn.utils.clip_grad_norm_(params,max_norm=1.)
            optim.step()
            stats.append({"local_mse":float(local.detach()),
                          "kl":float(kl.detach()),
                          "ce":float(cross_entropy.detach()),
                          "student_ce":float(student_ce.detach()),
                          "student_kl":float(student_kl.detach())})
        cal_receipt=calibrate_gates(
            hybrid,cal,teacher_ce=dense_cal_ce,
            budget=calibration_budget_nats,batch=batch,device=device,
            focus_layers=focus_layers)
        history.append({"epoch":epoch+1,
            "train_local_mse":sum(x["local_mse"] for x in stats)/len(stats),
            "train_kl":sum(x["kl"] for x in stats)/len(stats),
            "train_ce":sum(x["ce"] for x in stats)/len(stats),
            "train_student_only_ce":sum(x["student_ce"] for x in stats)/len(stats),
            "train_student_only_kl":sum(x["student_kl"] for x in stats)/len(stats),
            "gates":gates(hybrid),"calibration":cal_receipt,
            "student_only_calibration_ce":student_rollout_cross_entropy(
                hybrid,cal,batch=batch,device=device),
            "focused_student_calibration_ce":student_rollout_cross_entropy(
                hybrid,cal,batch=batch,device=device,
                student_only_layers=focus_layers)})
        if progress:
            progress("epoch %d/%d: calibration and development scoring complete" %
                     (epoch + 1, epochs))
    hybrid.eval()
    final=token_cross_entropy(hybrid,heldout,batch=batch,device=device)
    student_only_ce=student_rollout_cross_entropy(
        hybrid,heldout,batch=batch,device=device)
    focused_student_ce=student_rollout_cross_entropy(
        hybrid,heldout,batch=batch,device=device,
        student_only_layers=focus_layers)
    accepted=(final["ce_nats"] is not None and
              final["ce_nats"]<=dense_held_ce+calibration_budget_nats)
    student_pass=(student_only_ce is not None and np.isfinite(student_only_ce) and
                  student_only_ce<=dense_held_ce+calibration_budget_nats)
    return hybrid,{
        "teacher_calibration_ce":dense_cal_ce,
        "teacher_heldout_ce":dense_held_ce,
        "zero_gate_heldout_ce":zero_ce,
        "final_heldout_ce":final["ce_nats"],
        "student_only_heldout_ce":student_only_ce,
        "focused_student_heldout_ce":focused_student_ce,
        "focus_layers":list(focus_layers) if focus_layers is not None else None,
        "focus_is_diagnostic_only":focus_layers is not None,
        "matched_control_objective":matched_control_objective,
        "optimizer_fit_indices":optimizer_fit_indices.tolist(),
        "calibration_indices":calibration_indices.tolist(),
        "optimizer_fit_rows":len(fit),
        "optimizer_fit_window_bytes_sha256":hashlib.sha256(
            fit.tobytes()).hexdigest(),
        "student_only_heldout_pass":student_pass,
        "heldout_pass":accepted,
        "nonzero_quality_gates":sum(g>0 for g in gates(hybrid)),
        "all_quality_gates":gates(hybrid),
        "full_student_replacement_pass":focus_layers is None and accepted and
            student_pass and all(g==1.0 for g in gates(hybrid)),
        "retained_dense_teacher_ffns":True,
        "epochs":history,
        "validation_provenance":"train-only calibration reselects gates after each optimizer revision; independent heldout verifies initial dense parity and final quality only",
        "gradient_path":"student_only=True trains all student blocks end-to-end even at gate zero; admission gates change only through train-calibration",
        "student_loss_weights":{"global_ce":student_ce_weight,"global_kl":student_kl_weight,
                                "local_mse":mse_weight,
                                "hybrid_ce":0. if matched_control_objective else ce_weight,
                                "hybrid_kl":0. if matched_control_objective else kl_weight},
        "objective_comparison_scope":(
            "matched terms, optimizer training rows and updates only; "
            "different model initialization, topology, sparse routing and "
            "compute can still confound causal comparison"
            if matched_control_objective else
            "original hybrid-plus-student continuation objective"),
        "scope":"A hybrid gate=0 reproduces teacher but provides zero compression. A gated hybrid remains heavier than either standalone model until a validated full replacement allows teacher removal."}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--teacher",required=True,type=Path)
    p.add_argument("--student",required=True,type=Path)
    p.add_argument("--config",type=Path)
    p.add_argument("--train-ids",required=True,type=Path)
    p.add_argument("--heldout-ids",required=True,type=Path)
    p.add_argument("--epochs",type=int,default=3)
    p.add_argument("--batch",type=int,default=2)
    p.add_argument("--seed",type=int,default=41)
    p.add_argument("--device",default="cpu")
    p.add_argument("--learning-rate",type=float,default=3e-4)
    p.add_argument("--calibration-budget-nats",type=float,default=.02)
    p.add_argument("--student-ce-weight",type=float,default=1.)
    p.add_argument("--student-kl-weight",type=float,default=1.)
    p.add_argument("--focus-layers",type=int,nargs="+",
                   help="train only these layer indices; no teacher-free promotion in focused mode")
    p.add_argument("--matched-control-objective",action="store_true",
                   help="single-focus only: MSE + focused CE/KL with teacher-input FFN and zero training gates")
    p.add_argument("--export-optimizer-fit-ids",type=Path,
                   help="write exact optimiser rows after internal calibration holdback for a dense control")
    p.add_argument("--output",type=Path,default=Path("guarded_moe_candidate.pt"))
    args=p.parse_args()
    teacher,student=load_pair(args.teacher,args.student,args.config)
    train=np.load(args.train_ids,allow_pickle=False)
    heldout=np.load(args.heldout_ids,allow_pickle=False)
    for a in (train,heldout):
        if not np.issubdtype(a.dtype,np.integer):
            raise ValueError("token IDs must be integers")
        if a.min()<0 or a.max()>=teacher.cfg.vocab_size:
            raise ValueError("token IDs outside checkpoint vocabulary")
    hybrid,receipt=train_continuation(
        teacher,student,train,heldout,epochs=args.epochs,batch=args.batch,
        seed=args.seed,device=args.device,lr=args.learning_rate,
        calibration_budget_nats=args.calibration_budget_nats,
        student_ce_weight=args.student_ce_weight,
        student_kl_weight=args.student_kl_weight,
        focus_layers=tuple(args.focus_layers) if args.focus_layers is not None else None,
        matched_control_objective=args.matched_control_objective,
        progress=lambda message:print("[guarded-train] " + message, flush=True))
    receipt["source_checkpoint_sha256"]=sha256_file(args.teacher)
    receipt["student_candidate_sha256"]=sha256_file(args.student)
    receipt["candidate_only"]=not receipt["full_student_replacement_pass"]
    if args.export_optimizer_fit_ids is not None:
        dest=args.export_optimizer_fit_ids
        if dest.resolve() in (args.train_ids.resolve(),args.heldout_ids.resolve()):
            raise ValueError("refuse overwrite of original train/development token arrays")
        actual_fit=train[receipt["optimizer_fit_indices"]]
        if hashlib.sha256(actual_fit.tobytes()).hexdigest()!=receipt[
                "optimizer_fit_window_bytes_sha256"]:
            raise AssertionError("optimizer-fit export digest mismatch")
        dest.parent.mkdir(parents=True,exist_ok=True)
        np.save(dest,actual_fit)
        receipt["exported_optimizer_fit_path"]=str(dest.resolve())
        receipt["exported_optimizer_fit_file_sha256"]=sha256_file(dest)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"model_state_dict":hybrid.cpu().state_dict(),
                "config":dataclasses.asdict(hybrid.cfg),
                "guarded_quality_receipt":receipt},args.output)
    args.output.with_suffix(".json").write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2))


if __name__=="__main__":
    main()
