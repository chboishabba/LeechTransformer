"""F6 independent one-time frozen C2 / C3 / C3² three-arm lockbox.

Requires the separately preregistered F6 DEVELOPMENT screen and previously
completed F4/F5 official receipts. Locks architecture on F6 development means
before scoring ONE new independent final dataset. Rejects prior train,
historically exposed heldout, F4 development, F5 final, F6 development,
and optional other previously reviewed token sources (16-gram exact guard).
Document provenance, task C3² learning and compute/resource matching remain
separate proof/empirical obligations. This lockbox is NEVER an optimiser.
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
import torch
from scripts.score_frozen_c2c3_lockbox import load_frozen_arm
from scripts.screen_bounded_c2_vs_c2c3_router_fresh import (
    rows_from_file,validate_fresh,evaluate)
from scripts.reproduce_c2_quality_and_layers import load_trained
from scripts.train_guarded_moe_continuation import sha256_file

ARMS=("c2_invariant","c2c3_wave","punctured9_wave_matched")
KEYS=("c2_best_ce","c2c3_best_ce","c3square_best_ce")

def freeze_three_arm_winner(screen):
    if screen.get("kind")!="original_bounded_c2_c3_c3square_three_arm_development_screen":
        raise ValueError("requires official F6 three-arm development screen")
    pairs=screen.get("paired")
    if not isinstance(pairs,list) or not pairs:raise ValueError("missing F6 seeds")
    seeds=set();mean={g:[] for g in ARMS}
    for record in pairs:
        seed=record.get("seed")
        if not isinstance(seed,int) or seed in seeds:
            raise ValueError("missing or duplicate F6 paired seed")
        seeds.add(seed)
        for g,key in zip(ARMS,KEYS):
            value=record.get(key)
            if not isinstance(value,(float,int)) or not math.isfinite(value):
                raise ValueError("missing or nonfinite F6 development CE")
            mean[g].append(float(value))
    means={g:sum(v)/len(v) for g,v in mean.items()}
    # Precommitted tie break order: original C2, then triadic C3, then C3².
    chosen=min(ARMS,key=lambda g:(means[g],ARMS.index(g)))
    return chosen,means,seeds

def frozen_three_arm_checkpoints(screen,seeds):
    records=screen.get("arms")
    if not isinstance(records,list):raise ValueError("F6 arm checkpoints absent")
    mapping={}
    for r in records:
        k=(r.get("seed"),r.get("geometry"))
        if k in mapping or k[0] not in seeds or k[1] not in ARMS:
            raise ValueError("duplicate/unexpected F6 seed or geometry")
        if not r.get("candidate_checkpoint") or not r.get("candidate_checkpoint_sha256"):
            raise ValueError("F6 frozen checkpoint missing SHA/path")
        mapping[k]=r
    if set(mapping)!={(s,g) for s in seeds for g in ARMS}:
        raise ValueError("incomplete paired F6 model set")
    return mapping

def run(args):
    if args.output.exists():raise FileExistsError("official F6 final receipt exists")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("physical HIP GPU unavailable")
    if args.batch<1:raise ValueError("invalid batch size")
    screen=json.loads(args.f6_screen_receipt.read_text())
    chosen,development_means,seeds=freeze_three_arm_winner(screen)
    candidates=frozen_three_arm_checkpoints(screen,seeds)
    sources={
        "source_repaired_checkpoint_sha256":args.source_candidate,
        "fit_sha256":args.original_train_ids,
        "previous_heldout_sha256":args.original_heldout_ids,
        "fresh_validation_sha256":args.f6_development_ids,
        "f6_prior_f4_development_sha256":args.prior_f4_development_ids,
        "f6_prior_f5_lockbox_sha256":args.prior_f5_lockbox_ids,
        "f6_prior_f4_screen_receipt_sha256":args.prior_f4_screen_receipt,
        "f6_prior_f5_final_receipt_sha256":args.prior_f5_final_receipt,
    }
    for key,path in sources.items():
        if sha256_file(path)!=screen.get(key):
            raise ValueError("source/provenance mismatch in F6: "+key)
    old_f4=json.loads(args.prior_f4_screen_receipt.read_text())
    old_f5=json.loads(args.prior_f5_final_receipt.read_text())
    if old_f4.get("kind")!="original_bounded_c2_vs_c2c3_router_matched_screen" or (
            old_f5.get("kind")!="frozen_bounded_c2_c2c3_one_time_token_lockbox") or (
            old_f5.get("screen_receipt_sha256")!=sha256_file(args.prior_f4_screen_receipt)):
        raise ValueError("prior F4/F5 official receipt chain broken")
    old_train=rows_from_file(args.original_train_ids)
    old_heldout=rows_from_file(args.original_heldout_ids)
    f4_dev=rows_from_file(args.prior_f4_development_ids)
    f5_lock=rows_from_file(args.prior_f5_lockbox_ids)
    f6_dev=rows_from_file(args.f6_development_ids)
    locked=rows_from_file(args.f6_locked_test_ids)
    others=[rows_from_file(path) for path in args.other_reviewed_ids]
    old=np.concatenate([old_train,f4_dev,f5_lock,f6_dev]+others,axis=0)
    window=screen.get("validation_overlap_window")
    if not isinstance(window,int) or not 2<=window<=18:
        raise ValueError("invalid original overlap guard")
    unique=validate_fresh(locked,old,old_heldout,window=window)
    measured=[]
    for seed in sorted(seeds):
        for geometry in ARMS:
            r=candidates[(seed,geometry)]
            model=load_frozen_arm(r,seed=seed,geometry=geometry,
                source_sha=screen["source_repaired_checkpoint_sha256"],
                validation_sha=screen["fresh_validation_sha256"])
            model=model.to(args.device).eval()
            ce=evaluate(model,locked,args.batch,args.device)
            measured.append({"seed":seed,"geometry":geometry,
                "locked_ce":ce,"candidate_checkpoint_sha256":
                r["candidate_checkpoint_sha256"]})
            del model
            if args.device.startswith("cuda"):torch.cuda.empty_cache()
    _,original=load_trained(args.source_candidate,args.teacher,args.config)
    if original.cfg.moe_route_geometry!="c2_invariant" or (
            original.cfg.moe_late_history_cap!=64.):
        raise ValueError("not the original bounded C2 source")
    original=original.to(args.device).eval()
    source_ce=evaluate(original,locked,args.batch,args.device)
    del original
    means={g:sum(row["locked_ce"] for row in measured if row["geometry"]==g)/len(seeds)
           for g in ARMS}
    paired=[]
    for seed in sorted(seeds):
        scores={r["geometry"]:r["locked_ce"] for r in measured if r["seed"]==seed}
        paired.append({"seed":seed,"locked_ce_by_arm":scores,
                       "c3_minus_c2":scores[ARMS[1]]-scores[ARMS[0]],
                       "c3square_minus_c2":scores[ARMS[2]]-scores[ARMS[0]],
                       "c3square_minus_c3":scores[ARMS[2]]-scores[ARMS[1]]})
    result={"schema":1,"kind":"f6_frozen_three_arm_independent_one_time_lockbox",
        "f6_screen_receipt_sha256":sha256_file(args.f6_screen_receipt),
        "f6_locked_test_sha256":sha256_file(args.f6_locked_test_ids),
        "f6_locked_rows":len(locked),"unique_f6_lockbox_ngrams":unique,
        "original_source_sha256":sha256_file(args.source_candidate),
        "preselected_geometry_from_f6_development_only":chosen,
        "f6_development_mean_ce":development_means,
        "f6_same_lockbox_mean_ce":means,
        "original_repaired_c2_same_lockbox_ce":source_ce,
        "selected_minus_original_c2_same_lockbox_ce":means[chosen]-source_ce,
        "paired_seed_scores":paired,"frozen_models":measured,
        "previous_official_f4_and_f5_excluded":True,
        "previously_reviewed_extra_sources_excluded":
            [sha256_file(p) for p in args.other_reviewed_ids],
        "token_ngrams_verified_not_document_semantics":True,
        "source_document_identity_independently_verified":False,
        "nominal_router_parameters_matched_not_effective_rank_or_compute":True,
        "source_f3_squared_learned_phase_proved":False,
        "f6_final_test_not_used_for_selection":True,
        "release_admitted":False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True,
                                      allow_nan=False)+"\n")
    print(json.dumps({"selected":chosen,"locked_means":means,"source_ce":source_ce},
                     indent=2))
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("f6-screen-receipt","source-candidate","teacher",
        "original-train-ids","original-heldout-ids","prior-f4-development-ids",
        "prior-f5-lockbox-ids","prior-f4-screen-receipt","prior-f5-final-receipt",
        "f6-development-ids","f6-locked-test-ids","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--other-reviewed-ids",type=Path,action="append",default=[])
    p.add_argument("--config",type=Path)
    p.add_argument("--device",default="cuda")
    p.add_argument("--batch",type=int,default=1)
    run(p.parse_args())
if __name__=="__main__":main()
