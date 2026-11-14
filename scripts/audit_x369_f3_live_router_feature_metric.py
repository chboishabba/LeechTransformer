"""X369 F3 source-owned live router feature metric on frozen original arm.

Reuses original F4 original-source hash checks, frozen nonselected C3 arm,
already-viewed F3 probe rows, and the same real HistoryOrbitMoE router
method. Observe [x,history_e] as it is actually consumed by every
selected source layer, without modifying model parameters or routing.
Collect bounded deterministic reservoir samples, then evaluate true
router-feature and centered-logit displacement for the relative-C3 and
activation-norm/displacement-matched random comparator.

Probe is exploratory: source F3 data may have been viewed before and
does not count as independent F5 evidence. Never load F5 token arrays,
even though an optional reserved SHA is reported.
"""
from __future__ import annotations
import argparse,contextlib,hashlib,json
from pathlib import Path
import numpy as np
import torch
from scripts.audit_x369_f3_f4_router_feature_metric import (
    source_f3_f4_router_geometry_audit)
from scripts.screen_bounded_c2_vs_c2c3_router_fresh import (
    rows_from_file,validate_fresh)
from scripts.score_frozen_c2c3_lockbox import (
    arm_checkpoints,load_frozen_arm,select_on_development)
from scripts.audit_c3_causal_router_ce import row_ce
from scripts.train_guarded_moe_continuation import sha256_file


@contextlib.contextmanager
def bounded_router_state_capture(model,*,seed=41,max_states_per_layer=32):
    if (type(max_states_per_layer) is not int or
            not 4<=max_states_per_layer<=512):
        raise ValueError("bounded router capture budget 4..512 required")
    rngs=[np.random.default_rng(seed+i*101) for i in range(len(model.blocks))]
    seen=[0]*len(model.blocks)
    states=[[] for _ in model.blocks]
    originals=[]
    try:
        for i,block in enumerate(model.blocks):
            moe=block.moe
            if moe is None or moe.route_geometry!="c2c3_wave":
                raise ValueError("real frozen C3 source geometry required")
            original=moe.select_source_experts
            originals.append((moe,original))
            def intercept(x,history,*,i=i,original=original):
                if x.ndim!=2 or x.shape[1]!=model.cfg.d_model or (
                        history.shape!=(len(x),model.cfg.moe_experts,
                                        model.cfg.d_model)):
                    raise ValueError("actual original route state changed shape")
                for j in range(len(x)):
                    seen[i]+=1
                    k=seen[i]
                    if len(states[i])<max_states_per_layer:
                        states[i].append((x[j].detach().cpu().clone(),
                                          history[j].detach().cpu().clone()))
                    else:
                        choice=int(rngs[i].integers(0,k))
                        if choice<max_states_per_layer:
                            states[i][choice]=(x[j].detach().cpu().clone(),
                                               history[j].detach().cpu().clone())
                return original(x,history)
            moe.select_source_experts=intercept
        yield states,seen
    finally:
        for moe,original in originals:
            moe.select_source_experts=original


def run(args):
    if args.output.exists():
        raise FileExistsError("refuse overwrite live source router metric")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("native RX 580 requested but unavailable")
    screen=json.loads(args.screen_receipt.read_text())
    if screen.get("kind")!="original_bounded_c2_vs_c2c3_router_matched_screen":
        raise ValueError("original 2-arm F4 screen required")
    selected,means,seeds=select_on_development(screen)
    if args.seed not in seeds:
        raise ValueError("probe seed not present in original F4 screen")
    candidate=arm_checkpoints(screen,seeds)[(args.seed,"c2c3_wave")]
    for key,path in (
        ("source_repaired_checkpoint_sha256",args.source_candidate),
        ("fit_sha256",args.original_train_ids),
        ("previous_heldout_sha256",args.original_heldout_ids),
        ("fresh_validation_sha256",args.selection_validation_ids)):
        if screen.get(key)!=sha256_file(path):
            raise ValueError("original F4 source hash mismatch: "+key)
    originals=rows_from_file(args.original_train_ids)
    viewed=rows_from_file(args.original_heldout_ids)
    development=rows_from_file(args.selection_validation_ids)
    probe=rows_from_file(args.probe_ids)
    if args.reserved_lockbox_sha256 is not None and (
            len(args.reserved_lockbox_sha256)!=64 or
            any(c not in "0123456789abcdef"
                for c in args.reserved_lockbox_sha256)):
        raise ValueError("reserved lockbox SHA256 must be 64 lowercase hex chars")
    validate_fresh(probe,np.concatenate((originals,development)),
                   viewed,window=screen.get("validation_overlap_window",16))
    model=load_frozen_arm(
        candidate,seed=args.seed,geometry="c2c3_wave",
        source_sha=screen["source_repaired_checkpoint_sha256"],
        validation_sha=screen["fresh_validation_sha256"]).to(args.device).eval()
    for block in model.blocks:
        block.moe.route_probe_action=None
    with bounded_router_state_capture(
            model,seed=args.seed,
            max_states_per_layer=args.max_states_per_layer) as (states,seen):
        baseline=row_ce(model,probe,args.device)
    layers=[]
    for i,(saved,amount) in enumerate(zip(states,seen)):
        if not saved or amount<2:
            raise ValueError("insufficient live original source router states")
        x=torch.stack([item[0] for item in saved]).double()
        h=torch.stack([item[1] for item in saved]).double()
        weight=model.blocks[i].moe.router.weight.detach().cpu().double()
        d=source_f3_f4_router_geometry_audit(x,h,weight,seed=args.seed+i)
        d.update({"layer":i,"original_states_seen":amount,
                  "reservoir_states_sampled":len(saved),
                  "probe_rows_are_exploratory_reuse":True})
        layers.append(d)
    result={"schema":1,
            "kind":"x369_f3_f4_original_checkpoint_router_feature_displacement",
            "source_f4_screen_sha256":sha256_file(args.screen_receipt),
            "actual_frozen_candidate_checkpoint_sha256":
                candidate["candidate_checkpoint_sha256"],
            "candidate_best_development_step":candidate["best_step"],
            "geometry_won_development_selection":selected=="c2c3_wave",
            "frozen_probe_baseline_mean_ce":float(np.mean(baseline)),
            "frozen_probe_sha256":sha256_file(args.probe_ids),
            "original_fit_sha256":sha256_file(args.original_train_ids),
            "original_previous_heldout_sha256":sha256_file(args.original_heldout_ids),
            "original_f4_development_sha256":sha256_file(args.selection_validation_ids),
            "reserved_F5_sha256_only_never_opened":args.reserved_lockbox_sha256,
            "original_source_document_identity_independently_confirmed":False,
            "source_activation_displacement_matched_not_feature_displacement":True,
            "old_arbitrary_random_is_not_router_feature_displacement_matched":True,
            "new_mixed_orientation_per_expert_feature_displacement_exact":True,
            "new_control_is_not_original_task_semantic_action":True,
            "actual_task_C3_phase_intertwiner_proved":False,
            "learned_C3_mechanism_proved":False,
            "F5_locked_set_unscored":True,
            "all_layers":layers}
    a=args.output
    a.parent.mkdir(parents=True,exist_ok=True)
    a.write_text(json.dumps(result,indent=2,sort_keys=True,
                            allow_nan=False)+"\n")
    print(json.dumps({"checkpoint_best_step":candidate["best_step"],
        "selected_geometry":selected,
        "layers":[{"layer":d["layer"],"feature_delta_rel_mean":
                      d["feature_delta_rel_mean"],"feature_delta_random_mean":
                      d["feature_delta_random_mean"],
                      "feature_delta_exact_mixed_c3_mean":
                      d["feature_delta_exact_mixed_c3_mean"],
                      "exact_mixed_max_feature_error":
                      d["exact_mixed_c3_per_expert_feature_norm_max_abs_error"],
                      "fraction_5pct_old_random_feature_matched":
                      d["fraction_original_source_rows_feature_displacement_matched_5pct"]}
                  for d in layers]},indent=2))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("screen-receipt","source-candidate","original-train-ids",
                 "original-heldout-ids","selection-validation-ids",
                 "probe-ids","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--reserved-lockbox-sha256",
                   help="SHA256 copy only; NEVER load F5 tokens")
    p.add_argument("--seed",type=int,default=41)
    p.add_argument("--device",default="cuda")
    p.add_argument("--max-states-per-layer",type=int,default=32)
    run(p.parse_args())


if __name__=="__main__":
    main()
