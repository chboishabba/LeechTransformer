"""F3 causal SOURCE-router intervention CE on unseen original-tokenizer rows.

Frozen F4 c2c3_wave checkpoint only; no training, architecture selection,
or final F5 lockbox evaluation. We intervene only at ROUTER INPUT,
preserving expert execution activation and retained history semantics.
Diagonal C3 and C2 are symmetry controls; relative C3 and signed coordinate
permutation are comparison actions. Signed permutation preserves norm but
NOT the displacement size; therefore these alone do not prove a phase
circuit. A task-defined C3 label action and learned u(a) remain OPEN.
"""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scripts.screen_bounded_c2_vs_c2c3_router_fresh import (
    rows_from_file,validate_fresh)
from scripts.score_frozen_c2c3_lockbox import (
    arm_checkpoints,load_frozen_arm,select_on_development)
from scripts.train_guarded_moe_continuation import sha256_file

ACTIONS=("none","c3_diagonal_shift","c3_relative_shift",
         "c2_diagonal_negation","signed_axis_permutation_x")

def paired_cluster_interval(baseline,other,*,seed=41,bootstraps=1999,alpha=.05):
    a=np.asarray(baseline,dtype=np.float64)
    b=np.asarray(other,dtype=np.float64)
    if a.ndim!=1 or b.shape!=a.shape or len(a)<3 or (
            not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError("at least 3 finite matched source-document rows needed")
    if bootstraps<19 or not 0<alpha<1:
        raise ValueError("invalid cluster-bootstrap contract")
    delta=b-a
    rng=np.random.default_rng(seed)
    draws=rng.integers(len(delta),size=(bootstraps,len(delta)))
    interval=np.quantile(delta[draws].mean(axis=1),[alpha/2,1-alpha/2])
    return {"mean_delta_ce":float(delta.mean()),
            "cluster_bootstrap_interval":interval.tolist(),
            "rows":len(delta)}

@torch.no_grad()
def row_ce(model,rows,device):
    output=[]
    model.eval()
    for row in rows:
        tokens=torch.as_tensor(row,device=device,dtype=torch.long)[None,:]
        logits=model(tokens,student_only=True)[0]
        lp=F.log_softmax(logits[:,:-1,:].double(),dim=-1)
        chosen=lp.gather(-1,tokens[:,1:].unsqueeze(-1)).squeeze(-1)
        ce=float(-chosen.mean())
        if not math.isfinite(ce):raise ValueError("nonfinite original-source CE")
        output.append(ce)
    return output

def run(args):
    if args.output.exists():raise FileExistsError("refuse to overwrite causal probe")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested physical HIP/GPU unavailable")
    screen=json.loads(args.screen_receipt.read_text())
    if screen.get("kind")!="original_bounded_c2_vs_c2c3_router_matched_screen":
        raise ValueError("requires original 2-arm F4 development receipt")
    selected,dev_means,seeds=select_on_development(screen)
    if args.seed not in seeds:raise ValueError("selected seed not in F4 screen")
    candidate=arm_checkpoints(screen,seeds)[(args.seed,"c2c3_wave")]
    for key,path in (
        ("source_repaired_checkpoint_sha256",args.source_candidate),
        ("fit_sha256",args.original_train_ids),
        ("previous_heldout_sha256",args.original_heldout_ids),
        ("fresh_validation_sha256",args.selection_validation_ids)):
        if sha256_file(path)!=screen.get(key):
            raise ValueError("input source provenance does not match F4: "+key)
    original=rows_from_file(args.original_train_ids)
    viewed=rows_from_file(args.original_heldout_ids)
    dev=rows_from_file(args.selection_validation_ids)
    probe=rows_from_file(args.probe_ids)
    if dev.shape[1]!=original.shape[1]:
        raise ValueError("F4 development context width mismatch")
    previous=np.concatenate((original,dev),axis=0)
    if args.reserved_lockbox_ids is not None:
        locked=rows_from_file(args.reserved_lockbox_ids)
        if locked.shape[1]!=previous.shape[1]:
            raise ValueError("reserved lockbox context width mismatch")
        previous=np.concatenate((previous,locked),axis=0)
    validate_fresh(probe,previous,viewed,
                   window=screen.get("validation_overlap_window",16))
    model=load_frozen_arm(candidate,seed=args.seed,geometry="c2c3_wave",
        source_sha=screen["source_repaired_checkpoint_sha256"],
        validation_sha=screen["fresh_validation_sha256"]).to(args.device).eval()
    results={}
    for action in ACTIONS:
        for layer in model.blocks:
            if layer.moe is None:
                raise ValueError("source has missing original MoE layer")
            layer.moe.route_probe_action=None if action=="none" else action
            layer.moe.route_probe_seed=args.seed
        results[action]=row_ce(model,probe,args.device)
    for layer in model.blocks:
        layer.moe.route_probe_action=None
    baseline=results["none"]
    paired={action:paired_cluster_interval(baseline,results[action],
                seed=args.seed+index,bootstraps=args.bootstraps)
            for index,action in enumerate(ACTIONS) if action!="none"}
    result={
        "schema":1,"kind":"original_trained_c3_router_input_site_causal_probe",
        "screen_receipt_sha256":sha256_file(args.screen_receipt),
        "candidate_checkpoint_sha256":candidate["candidate_checkpoint_sha256"],
        "original_source_sha256":sha256_file(args.source_candidate),
        "selection_validation_sha256":sha256_file(args.selection_validation_ids),
        "probe_sha256":sha256_file(args.probe_ids),
        "reserved_lockbox_sha256":(sha256_file(args.reserved_lockbox_ids)
                                  if args.reserved_lockbox_ids else None),
        "seed":args.seed,"number_of_probe_rows":len(probe),
        "original_f4_selected_geometry_from_development_only":selected,
        "measured_model_geometry":"c2c3_wave",
        "measured_intervention_site":"routing features only; no FFN input replacement",
        "baseline_mean_ce":float(np.mean(baseline)),
        "action_mean_ce":{k:float(np.mean(v)) for k,v in results.items()},
        "paired_action_vs_baseline":paired,
        "reserved_lockbox_excluded_from_probe":True,
        "source_document_identity_independently_verified":False,
        "actual_task_c3_input_label_action_welded":False,
        "learned_phase_composition_proved":False,
        "phase_causal_specificity_proved":False,
        "this_probe_is_not_the_f5_final_lockbox":True,
        "release_admitted":False,
        "note":"Diagonal symmetries should leave engineered features unchanged. "
               "Random signed permutation is equal-norm, not equal displacement. "
               "Source action/causal CE measured only if input provenance passes."}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True,
                                      allow_nan=False)+"\n")
    print(json.dumps({"baseline_mean_ce":result["baseline_mean_ce"],
        "paired_action_vs_baseline":paired},indent=2))
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("screen-receipt","source-candidate","original-train-ids",
        "original-heldout-ids","selection-validation-ids","probe-ids","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--reserved-lockbox-ids",type=Path,required=True)
    p.add_argument("--seed",type=int,default=41)
    p.add_argument("--device",default="cuda")
    p.add_argument("--bootstraps",type=int,default=1999)
    run(p.parse_args())
if __name__=="__main__":main()
