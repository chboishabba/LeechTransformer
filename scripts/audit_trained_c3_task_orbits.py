"""F2/F3 original frozen student: TASK-DEFINED C3 orbit hidden probe.

Requires an external explicit source-task action manifest and train/eval
orbit arrays [rows,3,context]. No canonical C3 action is assumed for
arbitrary natural-language tokens. Model parameters never train: ONLY a
two-coordinate supervised phase readout fits source-labelled orbits.
This tests decodability of a label/phase representation, NOT spontaneous
neural computation of modular addition or causality. Composition defects
are descriptive necessary conditions, not sufficient mechanism proofs.
"""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scripts.reproduce_c2_quality_and_layers import load_trained
from scripts.screen_bounded_c2_vs_c2c3_router_fresh import (
    rows_from_file,validate_fresh)
from scripts.train_guarded_moe_continuation import sha256_file

def validate_task_orbits(fit,eval_orbits,source_train,old_heldout,
                         development,reserved_final,manifest,window=16):
    if any(x.ndim!=3 or x.shape[1]!=3 or not np.issubdtype(x.dtype,np.integer)
           for x in (fit,eval_orbits)):
        raise ValueError("typed integer [rows,three C3 actions,tokens] required")
    if fit.shape[0]<3 or eval_orbits.shape[0]<3:
        raise ValueError("at least three independent source-orbit groups per split")
    width=source_train.shape[1]
    if any(x.shape[-1]!=width for x in (fit,eval_orbits)) or (
            any(a.shape[1]!=width for a in
                (old_heldout,development,reserved_final))):
        raise ValueError("original tokenizer/context width mismatch")
    ids_fit=manifest.get("fit_document_ids")
    ids_eval=manifest.get("eval_document_ids")
    if not isinstance(ids_fit,list) or not isinstance(ids_eval,list) or (
            len(ids_fit)!=len(fit) or len(ids_eval)!=len(eval_orbits)):
        raise ValueError("document-level orbit provenance absent")
    if any(not isinstance(s,str) or not s for s in ids_fit+ids_eval) or (
            len(set(ids_fit+ids_eval))!=len(ids_fit)+len(ids_eval)):
        raise ValueError("duplicate or missing orbit document identifiers")
    if manifest.get("kind")!="task_defined_c3_orbits" or (
            not manifest.get("input_action_rule")) or (
            not manifest.get("output_action_rule")) or (
            not manifest.get("tokenizer_sha256")):
        raise ValueError("real external source C3 input/label action manifest required")
    previous=np.concatenate((source_train,old_heldout,development,reserved_final))
    fit_rows=fit.reshape(-1,width)
    eval_rows=eval_orbits.reshape(-1,width)
    validate_fresh(fit_rows,previous,previous,window=window)
    validate_fresh(eval_rows,np.concatenate((previous,fit_rows)),previous,
                   window=window)
    return {"fit_orbits":len(fit),"eval_orbits":len(eval_orbits),
            "fit_documents":len(ids_fit),"eval_documents":len(ids_eval),
            "source_action_independently_semantically_verified":False}

def phase_diagnostics(representations):
    """Representations [rows,three action powers,2], already normalized."""
    p=np.asarray(representations,dtype=np.float64)
    if p.ndim!=3 or p.shape[1:]!=(3,2) or len(p)<3 or not np.isfinite(p).all():
        raise ValueError("finite [rows,3,2] heldout phase representations required")
    norms=np.linalg.norm(p,axis=-1)
    if np.min(norms)<1e-7:
        raise ValueError("collapsed learned output has no phase")
    p=p/norms[...,None]
    c=-.5;s=math.sqrt(3)/2
    rotated=np.stack((c*p[...,0]-s*p[...,1],s*p[...,0]+c*p[...,1]),-1)
    eq=np.linalg.norm(np.roll(p,-1,axis=1)-rotated,axis=-1)
    defects=[]
    for a in range(3):
        for b in range(3):
            ua=p[:,a];ub=p[:,b];target=p[:,(a+b)%3]
            product=np.stack((ua[:,0]*ub[:,0]-ua[:,1]*ub[:,1],
                              ua[:,0]*ub[:,1]+ua[:,1]*ub[:,0]),-1)
            defects.append(np.linalg.norm(product-target,axis=-1))
    comp=np.stack(defects,-1)
    return {"heldout_equivariance_mean_defect":float(eq.mean()),
            "heldout_equivariance_max_defect":float(eq.max()),
            "heldout_phase_composition_mean_defect":float(comp.mean()),
            "heldout_phase_composition_max_defect":float(comp.max()),
            "heldout_phase_variance":float(np.var(p,axis=(0,1)).sum()),
            "heldout_orbit_groups":len(p),
            "trained_original_model_c3_circuit_proved":False}

@torch.no_grad()
def capture_hidden(model,orbit_ids,*,layer,device):
    output=[]
    hook=model.blocks[layer].register_forward_hook(
        lambda module,args,result: output.append(result[0][:,-1,:].detach().cpu()))
    try:
        model.eval()
        for group in orbit_ids:
            tokens=torch.as_tensor(group,dtype=torch.long,device=device)
            model(tokens,student_only=True)
    finally:
        hook.remove()
    if len(output)!=len(orbit_ids):
        raise ValueError("selected original hidden site not captured")
    return torch.stack(output,dim=0).float()

def fit_readout(fit,eval_vectors,*,steps=400,seed=41,lr=1e-2):
    torch.manual_seed(seed)
    if steps<1 or not 0<lr<1 or fit.ndim!=3 or eval_vectors.ndim!=3:
        raise ValueError("invalid training or extracted source activations")
    head=torch.nn.Linear(fit.shape[-1],2)
    # Model latents are pre-extracted from frozen source; train readout only.
    for p in head.parameters():p.requires_grad_(True)
    opt=torch.optim.AdamW(head.parameters(),lr=lr,weight_decay=1e-3)
    angle=torch.arange(3,dtype=torch.float32)*2*math.pi/3
    target=torch.stack((angle.cos(),angle.sin()),-1)[None,:,:].expand(fit.shape[0],-1,-1)
    for _ in range(steps):
        opt.zero_grad()
        pred=head(fit)
        loss=F.mse_loss(pred,target)
        if not torch.isfinite(loss):raise ValueError("nonfinite learned phase readout loss")
        loss.backward()
        opt.step()
    head.eval()
    with torch.no_grad():
        pred=head(eval_vectors)
        if pred.norm(dim=-1).min()<1e-7:
            raise ValueError("learned probe collapsed to undefined phase")
        normalized=pred/pred.norm(dim=-1,keepdim=True)
    return normalized.numpy(),float(loss.detach())

def run(args):
    if args.output.exists():raise FileExistsError("refuse overwrite task orbit probe")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested physical HIP device unavailable")
    fit=np.load(args.fit_orbit_ids,allow_pickle=False)
    ev=np.load(args.eval_orbit_ids,allow_pickle=False)
    manifest=json.loads(args.task_action_manifest.read_text())
    source_receipt=json.loads(args.source_repair_receipt.read_text())
    if source_receipt.get("kind")!="native_c2_bounded_early_even_warmstart":
        raise ValueError("not an original repaired bounded C2 source")
    if sha256_file(args.candidate)!=source_receipt.get("output_sha256"):
        raise ValueError("source repaired checkpoint SHA mismatch")
    if sha256_file(args.tokenizer_model)!=manifest.get("tokenizer_sha256"):
        raise ValueError("task orbit tokenizer must match original supplied tokenizer digest")
    old=rows_from_file(args.original_train_ids)
    if sha256_file(args.original_train_ids)!=source_receipt.get("source_train_sha256") or (
            sha256_file(args.original_heldout_ids)!=source_receipt.get("source_heldout_sha256")):
        raise ValueError("source train and prior heldout SHA mismatch")
    heldout=rows_from_file(args.original_heldout_ids)
    dev=rows_from_file(args.development_ids)
    reserved=rows_from_file(args.reserved_final_ids)
    contract=validate_task_orbits(fit,ev,old,heldout,dev,reserved,manifest)
    _,model=load_trained(args.candidate,args.teacher,args.config)
    if model.cfg.moe_route_geometry!="c2_invariant":
        raise ValueError("source is not original repaired C2 geometry")
    if not 0<=args.layer<len(model.blocks) or model.cfg.moe_late_history_cap!=64.:
        raise ValueError("actual original bounded source checkpoint and layer required")
    if fit.max()>=model.cfg.vocab_size or ev.max()>=model.cfg.vocab_size or (
            fit.min()<0 or ev.min()<0):
        raise ValueError("task input not in original tokenizer vocabulary")
    model=model.to(args.device).eval()
    trained_hidden=capture_hidden(model,fit,layer=args.layer,device=args.device)
    eval_hidden=capture_hidden(model,ev,layer=args.layer,device=args.device)
    predictions,fit_loss=fit_readout(trained_hidden,eval_hidden,
                                     steps=args.steps,seed=args.seed,lr=args.lr)
    stats=phase_diagnostics(predictions)
    output={"schema":1,"kind":"original_frozen_hidden_task_defined_c3_orbit_probe",
        "original_checkpoint_sha256":sha256_file(args.candidate),
        "original_teacher_sha256":sha256_file(args.teacher),
        "task_manifest_sha256":sha256_file(args.task_action_manifest),
        "source_repair_receipt_sha256":sha256_file(args.source_repair_receipt),
        "tokenizer_model_sha256":sha256_file(args.tokenizer_model),
        "fit_orbits_sha256":sha256_file(args.fit_orbit_ids),
        "evaluation_orbits_sha256":sha256_file(args.eval_orbit_ids),
        "reserved_final_sha256":sha256_file(args.reserved_final_ids),
        "hidden_layer":args.layer,"readout_seed":args.seed,
        "readout_steps":args.steps,"readout_fit_mse":fit_loss,
        "document_identity_contract":contract,"diagnostics":stats,
        "source_task_action_semantics_independently_certified":False,
        "phase_readout_supervised_on_labels_not_discovered_automatically":True,
        "original_model_parameters_modified":False,
        "semantic_task_c3_intertwiner_proved":False,
        "release_admitted":False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2,sort_keys=True,
                                      allow_nan=False)+"\n")
    print(json.dumps({"diagnostics":stats,"fit_mse":fit_loss},indent=2))
    return output

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("candidate","teacher","fit-orbit-ids","eval-orbit-ids",
                 "task-action-manifest","original-train-ids",
                 "original-heldout-ids","development-ids","reserved-final-ids","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--tokenizer-model",type=Path,required=True)
    p.add_argument("--source-repair-receipt",type=Path,required=True)
    p.add_argument("--config",type=Path)
    p.add_argument("--device",default="cuda")
    p.add_argument("--layer",type=int,default=1)
    p.add_argument("--steps",type=int,default=400)
    p.add_argument("--lr",type=float,default=1e-2)
    p.add_argument("--seed",type=int,default=41)
    run(p.parse_args())
if __name__=="__main__":main()
