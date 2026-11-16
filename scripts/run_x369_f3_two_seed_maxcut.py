"""Original X369 F3-B physical two-checkpoint coherence maximum cut.

Runs BOTH immutable F4 C3 arms (seed41 best step0, seed42 best step24)
over the same exploratory F3 documents and the same predeclared twelve
mixed-orientation masks. Never opens/scorers F5 reserved token bytes.

Output includes per-seed paired document CE and an explicit *descriptive*
seed42-minus-seed41 interaction. Because seeds have different initial
weights, the contrast is not an identified causal training effect and
F3's 16 rows are exploratory, not a confirmatory heldout test.

One invocation guarantees common provenance/null schedule and refuses
partial existing output. For RX580 require --require-gfx803, which checks
native ROCm/HIP GPU architecture; otherwise CPU diagnostics are possible.
"""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from scripts.audit_x369_f3_mixed_orientation_ensemble_ce import run as run_one
from scripts.train_guarded_moe_continuation import sha256_file
from scripts.score_frozen_c2c3_lockbox import (
    arm_checkpoints, select_on_development)
from scripts.audit_c3_causal_router_ce import validate_f3_probe_manifest

PROVENANCE_KEYS=(
  "original_F4_receipt_sha256","original_probe_sha256",
  "original_probe_provenance_sha256","reserved_F5_prior_array_sha256_only")


def pair_seed_checkpoints(a,b,*,bootstraps=1999,bootstrap_seed=36927):
    """Cluster-bootstrap original SOURCE DOCUMENTS, not 24 fake mask samples."""
    if type(bootstraps) is not int or bootstraps<99:
        raise ValueError(">=99 document bootstrap samples required")
    if a.get("source_seed")!=41 or b.get("source_seed")!=42:
        raise ValueError("exact frozen seed41/42 pair required")
    if (a.get("frozen_F4_checkpoint_best_development_step")!=0 or
        b.get("frozen_F4_checkpoint_best_development_step")!=24):
        raise ValueError("original frozen F4 best steps drifted")
    for key in PROVENANCE_KEYS:
        if not a.get(key) or a.get(key)!=b.get(key):
            raise ValueError("F3 source mismatch across checkpoints: "+key)
    if a.get("null_seeds_preselected")!=b.get("null_seeds_preselected"):
        raise ValueError("orientation-mask schedule differs between checkpoints")
    if a.get("null_seeds_preselected")!=list(range(40100,40112)):
        raise ValueError("precommitted twelve mask null schedule changed")
    n=a.get("source_document_rows")
    if (type(n) is not int or n<3 or n!=b.get("source_document_rows")):
        raise ValueError("original document cluster count differs")
    def observations(r):
        coherent=np.asarray(r.get("original_row_CE_coherent"),dtype=np.float64)
        baseline=np.asarray(r.get("original_row_CE_baseline"),dtype=np.float64)
        mixed=np.asarray(r.get("original_row_CE_mixed_by_seed"),dtype=np.float64)
        if (coherent.shape!=(n,) or baseline.shape!=(n,) or
            mixed.shape!=(12,n) or
            not np.isfinite(coherent).all() or
            not np.isfinite(baseline).all() or
            not np.isfinite(mixed).all() or
            min(coherent.min(),baseline.min(),mixed.min())<0):
            raise ValueError("malformed 12-mask original per-document CE source")
        return baseline,coherent,mixed
    b41,c41,m41=observations(a)
    b42,c42,m42=observations(b)
    delta41=c41-m41.mean(axis=0)
    delta42=c42-m42.mean(axis=0)
    interaction=delta42-delta41
    rng=np.random.default_rng(bootstrap_seed)
    idx=rng.integers(0,n,size=(bootstraps,n))
    intervals={}
    for name,values in (("seed41",delta41),("seed42",delta42),
                        ("seed42_minus_seed41",interaction)):
        means=values[idx].mean(axis=1)
        intervals[name]=[float(x) for x in np.quantile(means,[.025,.975])]
    per_mask41=c41.mean()-m41.mean(axis=1)
    per_mask42=c42.mean()-m42.mean(axis=1)
    paired_mask_change=per_mask42-per_mask41
    return dict(schema=1,kind="x369_f3_two_frozen_c3_seed_interaction",
        document_count=n,orientation_draws_per_seed=12,
        original_F4_development_steps=dict(seed41=0,seed42=24),
        source_identity={key:a[key] for key in PROVENANCE_KEYS},
        source_seed41_checkpoint=a["frozen_F4_checkpoint_sha256"],
        source_seed42_checkpoint=b["frozen_F4_checkpoint_sha256"],
        coherent_minus_mixed_mean_ce_by_seed={
            "seed41":float(delta41.mean()),"seed42":float(delta42.mean())},
        checkpoint_interaction_seed42_minus_seed41_ce=float(interaction.mean()),
        bootstrap_conditional_document_95pct_intervals=intervals,
        seed41_fraction_masks_coherent_better=float(np.mean(per_mask41<0)),
        seed42_fraction_masks_coherent_better=float(np.mean(per_mask42<0)),
        share_mask_draws_coherence_better_at_both_seeds=float(
            np.mean((per_mask41<0)&(per_mask42<0))),
        paired_mask_interaction_CE=paired_mask_change.tolist(),
        per_document_checkpoint_interaction_CE=interaction.tolist(),
        seed41_mean_baseline_ce=float(b41.mean()),
        seed42_mean_baseline_ce=float(b42.mean()),
        mask_draws_are_not_independent_document_clusters=True,
        independent_source_seed_causal_training_effect_estimated=False,
        preselected_F3_exploratory_not_F5_confirmatory=True,
        F5_raw_token_bytes_never_opened=True,
        source_original_C3_action_sufficiency_empirically_proved=False,
        physical_action_prediction_advantage_admitted=False)


def preflight_frozen_original_f4(args):
    """CPU/read-only gate BOTH original seed checkpoints before touching GPU.

    No reserved F5 file is accessed. This is a genuine-source discovery
    gate, not a claim that the C3 hypothesis has been physically tested.
    """
    if (not isinstance(args.reserved_lockbox_sha256, str) or
            len(args.reserved_lockbox_sha256) != 64 or
            any(c not in "0123456789abcdef"
                for c in args.reserved_lockbox_sha256)):
        raise ValueError("original F5 digest receipt required; raw F5 file forbidden")
    paths = {
        "screen_receipt": args.screen_receipt,
        "source_candidate": args.source_candidate,
        "original_train_ids": args.original_train_ids,
        "original_heldout_ids": args.original_heldout_ids,
        "selection_validation_ids": args.selection_validation_ids,
        "probe_ids": args.probe_ids,
        "probe_provenance": args.probe_provenance,
    }
    for key, path in paths.items():
        if path is None or not Path(path).is_file():
            raise FileNotFoundError(
                "R2 original F4 source artifact missing: " + key +
                " at " + str(path))
    screen = json.loads(Path(args.screen_receipt).read_text(encoding="utf-8"))
    if screen.get("kind") != "original_bounded_c2_vs_c2c3_router_matched_screen":
        raise ValueError("R2 requires original two-arm F4 source receipt")
    selected, means, seeds = select_on_development(screen)
    if seeds != {41, 42}:
        raise ValueError("R2 requires exactly original frozen F4 seeds 41 and 42")
    pairs = arm_checkpoints(screen, seeds)
    expected_paths = {
        "source_repaired_checkpoint_sha256": args.source_candidate,
        "fit_sha256": args.original_train_ids,
        "previous_heldout_sha256": args.original_heldout_ids,
        "fresh_validation_sha256": args.selection_validation_ids,
    }
    for key, path in expected_paths.items():
        expected = screen.get(key)
        if not isinstance(expected,str) or sha256_file(path) != expected:
            raise ValueError("R2 original F4 input SHA mismatch: " + key)
    probe_hash = sha256_file(args.probe_ids)
    provenance = json.loads(Path(args.probe_provenance).read_text(encoding="utf-8"))
    validate_f3_probe_manifest(
        provenance, screen, probe_hash, args.reserved_lockbox_sha256)
    checkpoints = []
    # Preflight BOTH seeds before running even the first frozen model.
    for seed, best_step in ((41, 0), (42, 24)):
        record = pairs[(seed, "c2c3_wave")]
        path = Path(record["candidate_checkpoint"])
        expected = record.get("candidate_checkpoint_sha256")
        if (record.get("best_step") != best_step or
                not isinstance(expected,str) or len(expected)!=64 or
                any(c not in "0123456789abcdef" for c in expected)):
            raise ValueError(
                f"R2 original F4 C3 seed{seed} checkpoint step/hash mismatch")
        if not path.is_file():
            raise FileNotFoundError(
                f"R2 original F4 C3 seed{seed} frozen checkpoint missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"R2 original F4 C3 seed{seed} SHA mismatch at {path}")
        checkpoints.append({
            "seed": seed,
            "original_f4_selected_development_step": best_step,
            "checkpoint_path": str(path),
            "checkpoint_sha256": actual,
        })
    return {
        "schema": 1,
        "kind": "x369_r2_frozen_original_two_seed_preflight_only",
        "original_F4_screen_receipt_sha256": sha256_file(args.screen_receipt),
        "original_F4_development_winner": selected,
        "original_F4_means_by_geometry": means,
        "original_probe_sha256": probe_hash,
        "original_probe_provenance_sha256": sha256_file(args.probe_provenance),
        "checkpoint_sources": checkpoints,
        "source_C2_checkpoint_sha256": screen["source_repaired_checkpoint_sha256"],
        "predeclared_mixed_null_seeds": list(range(40100, 40112)),
        "reserved_F5_sha256_only": args.reserved_lockbox_sha256,
        "reserved_F5_raw_bytes_read_by_this_preflight": False,
        "frozen_C3_weight_tensors_loaded": False,
        "original_R2_physical_CE_measured": False,
    }


def run(args):
    # A read-only artifact diagnosis must work even on a CPU-only host,
    # without creating output dirs or asserting live RX580 availability.
    verified = preflight_frozen_original_f4(args)
    if getattr(args, "preflight_only", False):
        print(json.dumps(verified, sort_keys=True, indent=2))
        return verified
    if args.output_dir.exists() and list(args.output_dir.iterdir()):
        raise FileExistsError("refuse to reuse/overwrite original F3-B output dir")
    if args.require_gfx803:
        if args.device!="cuda:0":
            raise ValueError("physical RX580 commitment needs device cuda:0")
        from scripts.train_lila_dense_gpu_smoke import assert_gpu
        assert_gpu()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    args_common={k:getattr(args,k) for k in (
        "screen_receipt","source_candidate","original_train_ids",
        "original_heldout_ids","selection_validation_ids","probe_ids",
        "probe_provenance","reserved_lockbox_sha256","device","bootstraps")}
    receipts=[]
    for seed,expected_step in ((41,0),(42,24)):
        result=run_one(SimpleNamespace(**args_common,
            seed=seed,null_seed_base=40100,null_draws=12,
            original_f3_receipt=(args.original_f3_receipt_41
                if seed==41 else args.original_f3_receipt_42),
            output=args.output_dir/f"f3_seed_{seed}.json"))
        if result["frozen_F4_checkpoint_best_development_step"]!=expected_step:
            raise ValueError("original frozen checkpoint F4 best step differs")
        receipts.append(result)
    paired=pair_seed_checkpoints(
        receipts[0],receipts[1],bootstraps=args.bootstraps)
    paired["seed41_receipt_sha256"]=sha256_file(args.output_dir/"f3_seed_41.json")
    paired["seed42_receipt_sha256"]=sha256_file(args.output_dir/"f3_seed_42.json")
    paired["hardware_req_gfx803"]=bool(args.require_gfx803)
    dest=args.output_dir/"f3_two_seed_summary.json"
    dest.write_text(json.dumps(paired,sort_keys=True,indent=2,
                               allow_nan=False)+"\n")
    print(json.dumps({k:paired[k] for k in (
        "coherent_minus_mixed_mean_ce_by_seed",
        "checkpoint_interaction_seed42_minus_seed41_ce",
        "seed41_fraction_masks_coherent_better",
        "seed42_fraction_masks_coherent_better",
        "share_mask_draws_coherence_better_at_both_seeds")},indent=2))
    return paired


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for item in ("screen-receipt","source-candidate","original-train-ids",
                 "original-heldout-ids","selection-validation-ids","probe-ids",
                 "probe-provenance","output-dir"):
        p.add_argument("--"+item,required=True,type=Path)
    p.add_argument("--reserved-lockbox-sha256",required=True)
    p.add_argument("--original-f3-receipt-41",type=Path)
    p.add_argument("--original-f3-receipt-42",type=Path)
    p.add_argument("--bootstraps",type=int,default=1999)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--require-gfx803",action="store_true")
    p.add_argument("--preflight-only",action="store_true",
        help="verify both original F4 C3 checkpoint files and source SHA before any GPU or output I/O")
    args=p.parse_args()
    if (len(args.reserved_lockbox_sha256)!=64 or any(
        c not in "0123456789abcdef" for c in args.reserved_lockbox_sha256)):
        p.error("pre-existing F5 SHA256 receipt required; NEVER pass its raw tokens")
    run(args)


if __name__=="__main__":main()
