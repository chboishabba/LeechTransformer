"""One-time, frozen C2/C2xC3 lockbox scoring after the X369-F4 screen.

This is a SEPARATE process from step selection. The F4 screen selects the
geometry using fresh development CE; this program freezes that selection
before reading any test CE. It checks exact token-window separation from
fit, previously viewed heldout, and selection data, then scores both
frozen arms and the original repaired C2 source on the same test rows.

Token n-gram disjointness is not proof of document/source disjointness,
nor is a lower CE a learned C3 mechanism. The caller must additionally
audit document IDs, tokenizer and dataset revisions. Never repeatedly
use the lockbox for architecture or hyperparameter selection.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from models.model import LeechConfig, LeechGPT
from scripts.reproduce_c2_quality_and_layers import load_trained
from scripts.screen_bounded_c2_vs_c2c3_router_fresh import (
    evaluate, rows_from_file, validate_fresh)
from scripts.train_guarded_moe_continuation import sha256_file

GEOMETRIES = ("c2_invariant", "c2c3_wave")


def validate_lockbox(locked, train, previously_viewed, selection, *, window=16):
    """Require exact 16-gram disjointness across ALL previously used splits."""
    # Verify the selection source too, not only the new test rows.
    validate_fresh(selection, train, previously_viewed, window=window)
    old = np.concatenate((train, selection), axis=0)
    return validate_fresh(locked, old, previously_viewed, window=window)


def select_on_development(screen):
    """Precommit to the lowest across-seed mean of best F4 development CEs."""
    pairs = screen.get("paired")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("screen lacks paired development evidence")
    seeds = set()
    sums = {g: [] for g in GEOMETRIES}
    for pair in pairs:
        seed = pair.get("seed")
        if not isinstance(seed, int) or seed in seeds:
            raise ValueError("missing or duplicate matched seed")
        seeds.add(seed)
        for g, key in (("c2_invariant", "c2_best_ce"),
                       ("c2c3_wave", "c2c3_best_ce")):
            value = pair.get(key)
            if not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError("missing or nonfinite development CE")
            sums[g].append(float(value))
    means = {g: sum(values) / len(values) for g, values in sums.items()}
    # Conservative fixed tie-break: C2, irrespective of lockbox outcome.
    selected = min(GEOMETRIES, key=lambda g: (means[g], GEOMETRIES.index(g)))
    return selected, means, seeds


def arm_checkpoints(screen, seeds):
    records = screen.get("arms")
    if not isinstance(records, list):
        raise ValueError("screen lacks candidate arm checkpoints")
    lookup = {}
    for record in records:
        key = (record.get("seed"), record.get("geometry"))
        if key in lookup or key[0] not in seeds or key[1] not in GEOMETRIES:
            raise ValueError("duplicate, unexpected or unpaired arm")
        if not record.get("candidate_checkpoint") or not record.get(
                "candidate_checkpoint_sha256"):
            raise ValueError("F4 arm was not saved for frozen scoring")
        lookup[key] = record
    if set(lookup) != {(s, g) for s in seeds for g in GEOMETRIES}:
        raise ValueError("not all paired seed checkpoints are present")
    return lookup


def load_frozen_arm(record, *, seed, geometry, source_sha, validation_sha):
    checkpoint = Path(record["candidate_checkpoint"])
    if sha256_file(checkpoint) != record["candidate_checkpoint_sha256"]:
        raise ValueError("frozen arm checkpoint hash disagrees with F4 receipt")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    provenance = payload["screen_provenance"]
    expected = {
        "kind": "fresh_bounded_c2_c2c3_router_only_candidate",
        "original_source_sha256": source_sha,
        "fresh_validation_sha256": validation_sha,
        "geometry": geometry,
        "seed": seed,
        "release_admitted": False,
    }
    if any(provenance.get(key) != val for key, val in expected.items()):
        raise ValueError("frozen checkpoint identity/provenance mismatch")
    cfg = LeechConfig(**payload["config"])
    if cfg.moe_route_geometry != geometry or cfg.moe_late_history_cap != 64.:
        raise ValueError("wrong frozen geometry or original R64 bound")
    model = LeechGPT(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.eval()


def run(args):
    if args.output.exists():
        raise FileExistsError("lockbox receipt exists; do not rescore it")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested physical HIP/CUDA GPU unavailable")
    if args.batch < 1:
        raise ValueError("positive batch size required")
    screen = json.loads(args.screen_receipt.read_text())
    if screen.get("kind") != "original_bounded_c2_vs_c2c3_router_matched_screen":
        raise ValueError("requires the original bounded F4 screening receipt")
    selected, development_means, seeds = select_on_development(screen)
    checkpoints = arm_checkpoints(screen, seeds)
    paths = {
        "fit_sha256": args.original_train_ids,
        "previous_heldout_sha256": args.original_heldout_ids,
        "fresh_validation_sha256": args.selection_validation_ids,
        "source_repaired_checkpoint_sha256": args.source_candidate,
    }
    for key, path in paths.items():
        if sha256_file(path) != screen.get(key):
            raise ValueError(f"F4 identity mismatch: {key}")
    test_sha = sha256_file(args.locked_test_ids)
    if test_sha in [sha256_file(path) for path in paths.values()]:
        raise ValueError("locked test is a previously used source")
    original = rows_from_file(args.original_train_ids)
    viewed = rows_from_file(args.original_heldout_ids)
    selection = rows_from_file(args.selection_validation_ids)
    locked = rows_from_file(args.locked_test_ids)
    window = screen.get("validation_overlap_window")
    if not isinstance(window, int) or not 2 <= window <= 18:
        raise ValueError("untrusted F4 overlap width")
    unique_test = validate_lockbox(locked, original, viewed, selection,
                                   window=window)
    # Selection is frozen entirely from F4 before the first test score.
    tested = []
    for seed in sorted(seeds):
        for geometry in GEOMETRIES:
            record = checkpoints[(seed, geometry)]
            model = load_frozen_arm(
                record, seed=seed, geometry=geometry,
                source_sha=screen["source_repaired_checkpoint_sha256"],
                validation_sha=screen["fresh_validation_sha256"])
            model = model.to(args.device)
            test_ce = evaluate(model, locked, args.batch, args.device)
            tested.append({
                "seed": seed, "geometry": geometry,
                "frozen_checkpoint_sha256": record["candidate_checkpoint_sha256"],
                "locked_test_ce": test_ce,
            })
            del model
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
    _, control = load_trained(args.source_candidate, args.teacher, args.config)
    if control.cfg.moe_route_geometry != "c2_invariant" or (
            control.cfg.moe_late_history_cap != 64.):
        raise ValueError("original bounded C2 control differs from F4 source")
    control = control.to(args.device).eval()
    control_ce = evaluate(control, locked, args.batch, args.device)
    del control
    means = {g: sum(r["locked_test_ce"] for r in tested
                    if r["geometry"] == g) / len(seeds) for g in GEOMETRIES}
    paired = [{
        "seed": seed,
        "c2_locked_ce": next(r["locked_test_ce"] for r in tested if
                             r["seed"] == seed and r["geometry"] == GEOMETRIES[0]),
        "c2c3_locked_ce": next(r["locked_test_ce"] for r in tested if
                               r["seed"] == seed and r["geometry"] == GEOMETRIES[1]),
    } for seed in sorted(seeds)]
    for row in paired:
        row["c2c3_minus_c2"] = row["c2c3_locked_ce"] - row["c2_locked_ce"]
    result = {
        "schema": 1, "kind": "frozen_bounded_c2_c2c3_one_time_token_lockbox",
        "screen_receipt_sha256": sha256_file(args.screen_receipt),
        "locked_test_sha256": test_sha,
        "locked_test_rows": len(locked),
        "unique_locked_test_ngrams": unique_test,
        "window": window,
        "selected_on_development_only": selected,
        "development_mean_ce": development_means,
        "lockbox_mean_ce": means,
        "original_repaired_c2_control_locked_ce": control_ce,
        "selected_minus_same_lockbox_source_ce": means[selected] - control_ce,
        "c2c3_minus_c2_mean_locked_ce":
            means["c2c3_wave"] - means["c2_invariant"],
        "frozen_arms": tested,
        "paired": paired,
        "document_disjointness_independently_verified": False,
        "token_window_overlap_rejected": True,
        "prior_heldout_scored": False,
        "learned_c3_equivariance_or_phase_composition_proved": False,
        "protocol_requires_no_repeated_lockbox_peeking": True,
        "release_admitted": False,
        "note": "Frozen selection is based on F4 development CE, never lockbox CE. "
                "Same-data repaired control is scored; historical 4.5966 heldout "
                "is NOT a directly comparable lockbox CE. Document/source lineage, "
                "32+ state bound, resource and causal claims remain separate gates."
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True,
                                      allow_nan=False) + "\n")
    print(json.dumps({
        "selected_on_development_only": selected,
        "lockbox_mean_ce": means,
        "original_repaired_c2_control_locked_ce": control_ce,
        "c2c3_minus_c2_mean_locked_ce": result["c2c3_minus_c2_mean_locked_ce"],
    }, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("screen-receipt", "original-train-ids",
                 "original-heldout-ids", "selection-validation-ids",
                 "locked-test-ids", "source-candidate", "teacher", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=1)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
