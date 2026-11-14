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
         "c2_diagonal_negation","signed_axis_permutation_x",
         "c3_matched_displacement_random_x",
         "c3_feature_exact_mixed_orientation_x")

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

def validate_f3_probe_manifest(manifest,screen,probe_sha,reserved_sha):
    """Hash-only F3/F5 separation, with no reserved file IO.

    This checks the recorded generator exclusion, not independently
    certified historical document identity or actual missing F5 bytes.
    """
    def valid_sha(t):
        return isinstance(t,str) and len(t)==64 and all(
            ch in "0123456789abcdef" for ch in t)
    if not valid_sha(reserved_sha) or not valid_sha(probe_sha):
        raise ValueError("actual F3/F5 digests must be 64 lowercase hex")
    prior=manifest.get("prior_token_array_sha256")
    produced=manifest.get("output_token_array_sha256")
    if (manifest.get("kind")!="x369_f4_f6_fresh_real_document_token_split"
            or not isinstance(prior,list) or reserved_sha not in prior
            or not isinstance(produced,dict)
            or produced.get("development")!=probe_sha
            or manifest.get(
                "selected_token_windows_disjoint_from_all_supplied_prior_arrays")
                is not True
            or manifest.get("original_token_overlap_window")!=
               screen.get("validation_overlap_window",16)):
        raise ValueError("F3 probe provenance fails frozen F5 prior-token "
                         "exclusion proof; do not open lockbox to repair")
    return {"reserve_digest_bound_in_original_F3_producer":True,
            "reserved_raw_tokens_not_opened_by_this_validation":True,
            "historical_document_independence_verified":False}


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
    candidate_best_step=candidate.get("best_step")
    if type(candidate_best_step) is not int or candidate_best_step<0:
        raise ValueError("F4 C3 candidate best_step missing or invalid")
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
    # F5 privacy: NEVER load or hash its raw token file in the F3 scorer.
    # The separately produced source fresh-split provenance must bind
    # this F3 probe to the frozen F5 reserved SHA through prior-file
    # overlap exclusion, or the F3 F5-disjoint admission fails closed.
    if args.reserved_lockbox_ids is not None:
        raise ValueError("F3 scorer must never load reserved F5 token arrays; "
                         "supply --reserved-lockbox-sha256 and --probe-provenance")
    if args.probe_provenance is None or args.reserved_lockbox_sha256 is None:
        raise ValueError("F3 requires provenance binding frozen F5 digest "
                         "without reading reserved F5 bytes")
    manifest=json.loads(args.probe_provenance.read_text())
    digest=args.reserved_lockbox_sha256
    validate_f3_probe_manifest(manifest,screen,
                               sha256_file(args.probe_ids),digest)
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
                seed=args.seed+index,bootstraps=args.bootstraps,
                alpha=.05/(len(ACTIONS)-1))
            for index,action in enumerate(ACTIONS) if action!="none"}
    # Directly compare the proposed C3 causal effect to a randomized
    # perturbation with MATCHED norm and displacement on each activation.
    # This still is not a semantic / task-action causality proof.
    # Exploratory TWO-comparator family.  The historical activation-
    # matched null is retained, but a new mixed ±C3 control matches the
    # actual nonlinear feature-displacement NORM exactly per expert.
    # Two-sided intervals are Bonferroni-adjusted for this PAIR of
    # exploratory specificity contrasts, not for all discovery questions.
    matched_specificity=paired_cluster_interval(
        results["c3_matched_displacement_random_x"],
        results["c3_relative_shift"],seed=args.seed+31,
        bootstraps=args.bootstraps,alpha=.025)
    exact_feature_specificity=paired_cluster_interval(
        results["c3_feature_exact_mixed_orientation_x"],
        results["c3_relative_shift"],seed=args.seed+32,
        bootstraps=args.bootstraps,alpha=.025)
    result={
        "schema":2,"kind":"original_trained_c3_router_input_site_causal_probe",
        "screen_receipt_sha256":sha256_file(args.screen_receipt),
        "candidate_checkpoint_sha256":candidate["candidate_checkpoint_sha256"],
        "original_source_sha256":sha256_file(args.source_candidate),
        "selection_validation_sha256":sha256_file(args.selection_validation_ids),
        "probe_sha256":sha256_file(args.probe_ids),
        "reserved_lockbox_sha256":digest,
        "probe_generation_provenance_sha256":sha256_file(args.probe_provenance),
        "seed":args.seed,"number_of_probe_rows":len(probe),
        "original_f4_selected_geometry_from_development_only":selected,
        "measured_model_geometry":"c2c3_wave",
        "measured_intervention_site":"routing features only; no FFN input replacement",
        "familywise_alpha":.05,
        "per_action_bonferroni_alpha":.05/(len(ACTIONS)-1),
        "baseline_mean_ce":float(np.mean(baseline)),
        "action_mean_ce":{k:float(np.mean(v)) for k,v in results.items()},
        "paired_action_vs_baseline":paired,
        "c3_relative_vs_equal_norm_equal_displacement_random":matched_specificity,
        "c3_relative_vs_exact_feature_displacement_mixed_orientation":
            exact_feature_specificity,
        "specificity_familywise_alpha":.05,
        "specificity_contrasts_in_family":2,
        "specificity_per_contrast_alpha":.025,
        "exact_feature_control_matched_for_each_original_expert":True,
        "exact_feature_control_breaks_coherent_global_c3_direction":True,
        "exact_feature_control_does_not_match_router_logit_displacement":True,
        "matched_random_control_preserves_source_router_input_displacement":True,
        "matched_random_control_preserves_hidden_x_norm_and_displacement":True,
        "matched_random_control_matches_nonlinear_router_feature_displacement":False,
        "matched_control_is_not_feature_metric_calibrated":True,
        "new_mixed_orientation_control_exact_feature_displacement_matched":True,
        "f4_development_selected_c3_geometry":selected=="c2c3_wave",
        "f3_c3_candidate_is_unselected_if_C2_development_won":
            selected=="c2_invariant",
        "c3_source_checkpoint_best_development_step":candidate_best_step,
        "f3_c3_checkpoint_had_no_selected_training_improvement":
            candidate_best_step==0,
        "f3_c3_intervention_demonstrates_trained_c3_circuit":False,
        "specificity_contrast_ci_is_unadjusted_for_multiple_exploratory_questions":False,
        "other_exploratory_question_family_not_included_in_this_adjustment":True,
        "reserved_lockbox_exclusion_recorded_by_prior_producer":True,
        "reserved_lockbox_exclusion_independently_verified_in_scorer":False,
        "repeat_original_F3_probe_is_posthoc_exploratory":True,
        "two_contrast_bonferroni_does_not_make_posthoc_probes_confirmatory":True,
        "joint_action_and_specificity_families_adjusted_together":False,
        "f5_token_file_bytes_not_opened_by_this_scorer":True,
        "f5_exclusion_bound_to_probe_prior_array_sha256":True,
        "source_document_identity_independently_verified":False,
        "actual_task_c3_input_label_action_welded":False,
        "learned_phase_composition_proved":False,
        "phase_causal_specificity_proved":False,
        "this_probe_is_not_the_f5_final_lockbox":True,
        "release_admitted":False,
        "note":"Diagonal symmetries should leave engineered features unchanged. "
               "The separate matched random action controls both norm and displacement; signed permutation does not. "
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
    p.add_argument("--reserved-lockbox-ids",type=Path,
        help="Forbidden compatibility arg: avoid loading reserved F5 tokens")
    p.add_argument("--reserved-lockbox-sha256",required=True)
    p.add_argument("--probe-provenance",type=Path,required=True)
    p.add_argument("--seed",type=int,default=41)
    p.add_argument("--device",default="cuda")
    p.add_argument("--bootstraps",type=int,default=1999)
    run(p.parse_args())
if __name__=="__main__":main()
