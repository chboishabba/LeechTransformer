"""X369 F3-B: frozen-source C3 coherent-vs-mixed orientation ensemble CE.

The native source exact mixed +/- triad C3 control matches the coherent
120-degree action BOTH in original activation displacement and in the
NONLINEAR routed-feature displacement NORM for each expert/history. This
script repeats the exact control under several preselected orientation
masks, keeping the coherent C3, weights, and original F3 probe unchanged.

The already-viewed 16-row F3 probe is EXPLORATORY. Inference across the
held-out F5 lockbox is forbidden. Only the F5 SHA256 from the earlier
probe producer is used; this code NEVER opens raw reserved F5 IDs.
All group-specific comparisons are descriptive, not exact null p-values,
and training seeds are NOT independent original checkpoint samples.
"""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
import torch
from scripts.audit_c3_causal_router_ce import (
    row_ce,validate_f3_probe_manifest,paired_cluster_interval)
from scripts.screen_bounded_c2_vs_c2c3_router_fresh import (
    rows_from_file,validate_fresh)
from scripts.score_frozen_c2c3_lockbox import (
    select_on_development,arm_checkpoints,load_frozen_arm)
from scripts.train_guarded_moe_continuation import sha256_file

ACTION_MIXED="c3_feature_exact_mixed_orientation_x"
ACTION_COHERENT="c3_relative_shift"


def summarize_f3_mixed_ensemble(baseline,coherent,mixed,*,seed=41,
                                bootstraps=1999):
    """One original source document = ONE bootstrap cluster.

    mixed[draw,source_row] varies only the preselected mixed-orientation
    negative-control mask. The row bootstrap resamples source rows; it
    never pretends masks are independent documents or relies on a
    classical p-value from a post-hoc symmetry search.
    """
    b=np.asarray(baseline,dtype=np.float64)
    c=np.asarray(coherent,dtype=np.float64)
    m=np.asarray(mixed,dtype=np.float64)
    if (b.ndim!=1 or b.size<3 or c.shape!=b.shape or
            m.ndim!=2 or m.shape[1]!=b.size or
            not 4<=m.shape[0]<=64 or type(bootstraps) is not int or
            bootstraps<19 or not np.isfinite(b).all() or
            not np.isfinite(c).all() or not np.isfinite(m).all() or
            min(float(b.min()),float(c.min()),float(m.min()))<0):
        raise ValueError("finite nonnegative [>=4 masks,>=3 original rows] "
                         "paired CE and >=19 bootstrap draws required")
    mean_m=m.mean(axis=0)
    relative_to_mixed=paired_cluster_interval(
        mean_m,c,seed=seed,bootstraps=bootstraps,alpha=.05)
    # Mean coherent-versus-mixed per document; negative indicates
    # lower CE for coherent C3, on this EXPLORED F3 set only.
    draw_gaps=c.mean()-m.mean(axis=1)
    report={
        "schema":1,"kind":"x369_f3_coherent_vs_mixed_orientation_null_ensemble",
        "source_document_rows":int(b.size),
        "mixed_null_orientation_draw_count":int(m.shape[0]),
        "mean_baseline_ce":float(b.mean()),
        "mean_coherent_c3_ce":float(c.mean()),
        "mean_mixed_c3_ce":float(m.mean()),
        "coherent_minus_baseline_mean_delta_ce":float((c-b).mean()),
        "mixed_minus_baseline_mean_delta_ce":float((mean_m-b).mean()),
        "coherent_minus_mixed_mean_delta_ce":
            relative_to_mixed["mean_delta_ce"],
        "row_cluster_bootstrap_95pct_descriptive_interval":
            relative_to_mixed["cluster_bootstrap_interval"],
        "coherent_minus_each_mixed_draw_mean_ce":draw_gaps.tolist(),
        "fraction_mixed_orientation_draws_coherent_has_lower_mean_ce":
            float(np.mean(draw_gaps<0)),
        "mixed_draw_mean_ce_range":[float(m.mean(axis=1).min()),
                                    float(m.mean(axis=1).max())],
        "per_original_row_coherent_minus_mixed_mean_ce":
            (c-mean_m).tolist(),
        "orientation_masks_are_not_independent_document_clusters":True,
        "row_bootstrap_is_conditional_on_supplied_document_independence":True,
        "selected_null_after_inspecting_old_F3_is_exploratory":True,
        "this_is_not_an_exact_randomisation_test_p_value":True,
        "no_F5_token_rows_loaded_scored_or_fit":True,
        "original_semantic_input_output_C3_action_verified":False,
        "learned_phase_composition_proved":False,
        "release_admitted":False,
    }
    return report


def _set_probe(model,action,seed):
    for block in model.blocks:
        if block.moe is None or block.moe.route_geometry!="c2c3_wave":
            raise ValueError("original frozen C3 MoE missing")
        block.moe.route_probe_action=action
        block.moe.route_probe_seed=seed


def score_frozen_f3_ensemble(model,rows,device,*,source_seed,
                              null_seed_base=40100,null_draws=12,
                              trace_rows=False):
    if (type(null_draws) is not int or not 4<=null_draws<=64 or
            type(null_seed_base) is not int or null_seed_base<0 or
            null_seed_base+null_draws>=2**31 or
            type(source_seed) is not int):
        raise ValueError("preselected 4..64 null draws and safe seeds required")
    observations={}
    try:
        _set_probe(model,None,source_seed)
        observations["baseline"]=row_ce(model,rows,device)
        _set_probe(model,ACTION_COHERENT,source_seed)
        observations["coherent"]=row_ce(model,rows,device)
        mixed=[]
        for offset in range(null_draws):
            _set_probe(model,ACTION_MIXED,null_seed_base+offset)
            mixed.append(row_ce(model,rows,device))
        observations["mixed"]=mixed
    finally:
        _set_probe(model,None,source_seed)
    if trace_rows:
        observations["seed_schedule"]=[
            null_seed_base+i for i in range(null_draws)]
    return observations


def run(args):
    if args.output.exists():
        raise FileExistsError("refuse overwrite original F3 ensemble receipt")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("physical RX580 requested but native GPU unavailable")
    if type(args.seed) is not int or args.seed<0:
        raise ValueError("frozen source seed must be nonnegative")
    screen=json.loads(args.screen_receipt.read_text())
    if screen.get("kind")!="original_bounded_c2_vs_c2c3_router_matched_screen":
        raise ValueError("original source F4 2-arm receipt required")
    selection,means,seeds=select_on_development(screen)
    if args.seed not in seeds:
        raise ValueError("original C3 checkpoint seed not in F4 receipt")
    arm=arm_checkpoints(screen,seeds)[(args.seed,"c2c3_wave")]
    for key,path in (
        ("source_repaired_checkpoint_sha256",args.source_candidate),
        ("fit_sha256",args.original_train_ids),
        ("previous_heldout_sha256",args.original_heldout_ids),
        ("fresh_validation_sha256",args.selection_validation_ids)):
        if sha256_file(path)!=screen.get(key):
            raise ValueError("source mismatch with frozen F4: "+key)
    fit=rows_from_file(args.original_train_ids)
    old=rows_from_file(args.original_heldout_ids)
    dev=rows_from_file(args.selection_validation_ids)
    probe=rows_from_file(args.probe_ids)
    if dev.shape[1]!=fit.shape[1]:
        raise ValueError("F4 development width differs from original source")
    validate_fresh(probe,np.concatenate((fit,dev)),old,
                   window=screen.get("validation_overlap_window",16))
    manifest=json.loads(args.probe_provenance.read_text())
    validate_f3_probe_manifest(manifest,screen,
                               sha256_file(args.probe_ids),
                               args.reserved_lockbox_sha256)
    model=load_frozen_arm(arm,seed=args.seed,geometry="c2c3_wave",
        source_sha=screen["source_repaired_checkpoint_sha256"],
        validation_sha=screen["fresh_validation_sha256"]).to(args.device).eval()
    observations=score_frozen_f3_ensemble(
        model,probe,args.device,source_seed=args.seed,
        null_seed_base=args.null_seed_base,
        null_draws=args.null_draws,trace_rows=True)
    # Baseline/relative F3 replay has exact context/seed/weight hash.
    # Refuse to silently compare different source-checkpoint receipts.
    if args.original_f3_receipt is not None:
        old_receipt=json.loads(args.original_f3_receipt.read_text())
        if (old_receipt.get("kind")!=
              "original_trained_c3_router_input_site_causal_probe"
            or old_receipt.get("seed")!=args.seed
            or old_receipt.get("candidate_checkpoint_sha256")!=
                 arm["candidate_checkpoint_sha256"]
            or old_receipt.get("probe_sha256")!=
                 sha256_file(args.probe_ids)):
            raise ValueError("previous F3 receipt not the same frozen probe")
        old_actions=old_receipt.get("action_mean_ce")
        if (not isinstance(old_actions,dict) or
            abs(float(old_receipt["baseline_mean_ce"])-
                np.mean(observations["baseline"]))>1e-4 or
            abs(float(old_actions[ACTION_COHERENT])-
                np.mean(observations["coherent"]))>1e-4):
            raise ValueError("source original F3 CE replay drift")
    report=summarize_f3_mixed_ensemble(
        observations["baseline"],observations["coherent"],
        observations["mixed"],seed=args.seed,
        bootstraps=args.bootstraps)
    report.update({
        "original_F4_receipt_sha256":sha256_file(args.screen_receipt),
        "frozen_F4_checkpoint_sha256":arm["candidate_checkpoint_sha256"],
        "frozen_F4_checkpoint_best_development_step":arm["best_step"],
        "source_seed":args.seed,
        "development_winning_geometry":selection,
        "original_frozen_C3_geometry_was_F4_development_winner":
            selection=="c2c3_wave",
        "original_probe_sha256":sha256_file(args.probe_ids),
        "original_probe_provenance_sha256":
            sha256_file(args.probe_provenance),
        "reserved_F5_prior_array_sha256_only":
            args.reserved_lockbox_sha256,
        "original_F3_receipt_sha256_optional":
            sha256_file(args.original_f3_receipt)
            if args.original_f3_receipt else None,
        "null_seeds_preselected":[args.null_seed_base+i
                                   for i in range(args.null_draws)],
        "original_row_CE_baseline":observations["baseline"],
        "original_row_CE_coherent":observations["coherent"],
        "original_row_CE_mixed_by_seed":observations["mixed"],
    })
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(
        report,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({
        "source_seed":args.seed,
        "best_step":arm["best_step"],
        "mean_coherent_minus_mixed_ce":
            report["coherent_minus_mixed_mean_delta_ce"],
        "exploratory_doc_cluster_interval_95pct":
            report["row_cluster_bootstrap_95pct_descriptive_interval"],
        "fraction_null_draws_coherent_lower_mean_ce":
            report["fraction_mixed_orientation_draws_coherent_has_lower_mean_ce"],
        "reserved_F5_not_read":True
    },sort_keys=True,indent=2))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("screen-receipt","source-candidate","original-train-ids",
        "original-heldout-ids","selection-validation-ids",
        "probe-ids","probe-provenance","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--reserved-lockbox-sha256",required=True)
    p.add_argument("--original-f3-receipt",type=Path)
    p.add_argument("--seed",type=int,default=41)
    p.add_argument("--null-seed-base",type=int,default=40100)
    p.add_argument("--null-draws",type=int,default=12)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--bootstraps",type=int,default=1999)
    run(p.parse_args())


if __name__=="__main__":
    main()
