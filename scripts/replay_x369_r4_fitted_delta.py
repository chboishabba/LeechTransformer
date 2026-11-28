"""Independent X369 R4c fitted-delta audit (source-bound; F5 prohibited).

Read the completed R4b development receipt, native R1 receipt and original
C2 source checkpoint, reconstruct each trained residual arm, and re-score
the EXACT development documents on the original recorded device.

This creates an independent consumer for model-delta artifacts; it is
still NOT a task-labelled C3 intervention or a confirmatory F5 evaluation.
Never train or perform new model selection in this script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from models.model import LeechGPT
from models.x369_fair_c3_residual import (
    R4_GEOMETRIES, restore_residual_delta, r4_geometry_source_fingerprint)
from scripts.audit_x369_slr_spacy_grammar_early_epoch import load_docs
from scripts.run_x369_r4_three_arm_source_maxcut import (
    digest, validated_r1_summary, validate_source_docs, per_document_ce)
from scripts.verify_trained_federated_moe import checkpoint_parts, infer_config


def run(args):
    if args.output.exists():
        raise FileExistsError("never overwrite R4 independent replay")
    r4 = json.loads(args.r4b_receipt.read_text())
    r1 = json.loads(args.r1_summary.read_text())
    if (r4.get("kind") != "x369_r4b_original_slr_three_capacity_matched_development" or
            r4.get("R4_scientific_release_admitted") is not False or
            r4.get("F5_raw_or_reserved_opened") is not False or
            r4.get("development_selection_only") is not True or
            r4.get("trained_delta_same_device_replay_required") is not True or
            r4.get("feature_semantics") != "x369_r4_q_gated_mixed_v2" or
            r4.get("geometry_source_sha256") != r4_geometry_source_fingerprint() or
            r4.get("fixed_mixed_C3_is_reparameterization_gauge") is not True or
            r4.get("phase_null_requires_sample_varying_q_mask") is not True or
            r4.get("nonfixed_mask_source_variation_empirically_verified") is not True or
            r4.get("device") != args.device):
        raise ValueError("requires complete development-only R4b receipt on the original device")
    source_sha = digest(args.repaired_c2_checkpoint)
    source_jsonl_sha = digest(args.documents_jsonl)
    if (source_sha != r4.get("repaired_c2_checkpoint_sha256") or
            digest(args.source_repair_receipt) != r4.get("source_repair_receipt_sha256") or
            digest(args.r1_summary) != r4.get("r1_summary_sha256")):
        raise ValueError("R4b original C2 source/R1 receipt identities differ")
    provenance = validated_r1_summary(
        r1, jsonl_sha=source_jsonl_sha, tokenizer_sha=digest(args.tokenizer))
    if provenance != r4.get("native_R1_provenance"):
        raise ValueError("native R1 source metadata changed since R4b")
    source_receipt = json.loads(args.source_repair_receipt.read_text())
    if (source_receipt.get("kind") != "native_c2_bounded_early_even_warmstart" or
            source_receipt.get("output_sha256") != source_sha):
        raise ValueError("wrong original repaired C2 source receipt")
    state, info = checkpoint_parts(args.repaired_c2_checkpoint)
    cfg = infer_config(state, info, args.config)
    if (cfg.moe_route_geometry != "c2_invariant" or
            not cfg.moe_guarded_ffn or cfg.moe_shared_e8_history or
            cfg.moe_late_history_cap != 64.):
        raise ValueError("not the original repaired R64 C2 student")
    with torch.random.fork_rng(devices=[]):
        source = LeechGPT(cfg).eval()
    source.load_state_dict(state, strict=True)
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    records = load_docs(args.documents_jsonl, sp, digest(args.tokenizer))
    _, development = validate_source_docs(records, cfg)
    if ([d["source_sha"] for d in development] !=
            r4.get("source_development_document_sha256")):
        raise ValueError("original R4 development documents no longer match")
    expected = {(seed, geom)
                for seed in r4.get("seeds", [])
                for geom in R4_GEOMETRIES}
    observations = r4.get("arms", [])
    if (not isinstance(observations, list) or not expected or
            len(observations) != len(expected) or
            len(r4.get("paired", [])) != len(r4["seeds"])):
        raise ValueError("missing R4b original paired arms")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("original HIP replay device not available")
    if args.delta_dir.name != r4.get("fitted_residual_delta_directory"):
        raise ValueError("fitted delta archive directory identity differs")
    seen = set()
    replays = []
    for arm in observations:
        seed, geom = arm.get("seed"), arm.get("geometry")
        key = (seed, geom)
        if key not in expected or key in seen:
            raise ValueError("duplicated or foreign fitted R4 arm")
        seen.add(key)
        if geom == "c2_c3_mixed_null_residual":
            diverse = arm.get("mixed_null_source_mask_diversity")
            if (not isinstance(diverse, dict) or
                    diverse.get("nonfixed_source_orientation_measured") is not True or
                    not 1 <= diverse.get("layers_with_distinct_sample_masks", 0)
                         <= cfg.n_layers):
                raise ValueError("R4 phase null did not vary on original source")
        if arm.get("saved_delta_replay_same_device_verified") is not True:
            raise ValueError("original R4b never verified physical trained delta")
        filename = arm.get("fitted_delta_artifact")
        if (not isinstance(filename, str) or
                filename != f"r4_seed_{seed}_{geom}.pt"):
            raise ValueError("unsafe or changed R4 fitted delta filename")
        checkpoint = args.delta_dir / filename
        if digest(checkpoint) != arm.get("fitted_delta_sha256"):
            raise ValueError("R4 artifact SHA does not match original receipt")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (saved.get("geometry") != geom or saved.get("seed") != seed or
                saved.get("selected_development_step") != arm.get("best_step")):
            raise ValueError("R4 checkpoint no longer encodes selected development state")
        model = restore_residual_delta(
            source, saved, source_sha256=source_sha,
            source_jsonl_sha256=source_jsonl_sha).to(args.device).eval()
        rows, ce = per_document_ce(model, development, args.device)
        if (abs(ce - arm["best_dev_ce_nats"]) > 1e-6 or
                len(rows) != len(arm.get("development_rows", [])) or
                any(row["document_id"] != old["document_id"] or
                    row["source_sha256"] != old["source_sha256"] or
                    abs(row["document_ce_nats"] - old["document_ce_nats"]) > 1e-6
                    for row, old in zip(rows, arm["development_rows"]))):
            raise AssertionError("independent original-device R4 delta CE replay drift")
        replays.append({
            "seed": seed, "geometry": geom, "fitted_delta_sha256": digest(checkpoint),
            "independent_replay_development_ce_nats": ce,
            "all_original_source_document_CE_replayed": True,
        })
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    if seen != expected:
        raise AssertionError("R4 experiment lacks a capacity-matched arm")
    result = {
        "schema": 1, "kind": "x369_r4c_independent_source_bound_fitted_delta_replay",
        "r4b_receipt_sha256": digest(args.r4b_receipt),
        "r1_summary_sha256": digest(args.r1_summary),
        "repaired_c2_checkpoint_sha256": source_sha,
        "native_slr_jsonl_sha256": source_jsonl_sha,
        "device": args.device,
        "independently_replayed_all_fitted_arms": True,
        "arm_replays": replays,
        "typed_C3_task_action_verified": False,
        "F5_raw_or_reserved_opened": False,
        "R4_scientific_release_admitted": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2)+"\n")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("r4b-receipt","r1-summary","repaired-c2-checkpoint",
                 "source-repair-receipt","documents-jsonl",
                 "tokenizer","delta-dir","output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--config", type=Path)
    p.add_argument("--device", choices=("cpu","cuda:0"), default="cuda:0")
    run(p.parse_args())


if __name__ == "__main__":
    main()
