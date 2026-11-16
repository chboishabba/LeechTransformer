"""X369 R4b: three capacity-matched C2/C3 ORIGINAL-source development experiment.

Requires a verified NATIVE SLR R1 orchestrator receipt, its exact exported
train/development JSONL, the original SentencePiece model, and the genuinely
repaired guarded C2 student checkpoint bound to its native repair receipt.

Arms (one GPU consumer at a time, equal TRAINABLE count):
   C2 residual = learn an extra score on |[x,h]| (capacity control)
   C3 coherent = learn an extra score on F(rotate3(x),h)
   C3 mixed    = learn an extra score on F(mixed_triads(x),h)
Every residual is initialized to zero. All source router, expert, history,
attention and output parameters remain FROZEN. This is a DEVELOPMENT
comparison, NOT a learnt task-C3-action theorem nor R5 final admission.

The R1 source receipt alone does not prove document independence from the
historical repaired C2 checkpoint's training corpus; this limitation is
explicit in all receipts. Never access reserved/F5 arrays in this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F

from models.model import LeechGPT
from models.trialectic_wave_prior import q_gated_mixed_c3_mask
from models.x369_fair_c3_residual import (
    R4_GEOMETRIES, assert_identical_initial_student_rollout,
    clone_c2_with_zero_c3_residual,
    make_residual_delta_payload, restore_residual_delta,
    r4_geometry_source_fingerprint,
)
from scripts.audit_x369_slr_spacy_grammar_early_epoch import load_docs
from scripts.verify_trained_federated_moe import checkpoint_parts, infer_config


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validated_r1_summary(r1, *, jsonl_sha: str, tokenizer_sha: str):
    if not isinstance(r1, dict):
        raise ValueError("R1 source summary must be an object")
    if (r1.get("parser_authority") != "slr_native_pg" or
            r1.get("F5_raw_source_opened") is not False):
        raise ValueError("need native SLR/PG receipt, never legacy/fixture/F5")
    kind = r1.get("kind")
    if kind == "x369_r1_native_pg_source_only_v1":
        exporter = r1.get("slr_exporter_original_receipt", {})
        inventory = r1.get("source_inventory", {})
        if (r1.get("source_export_R1_only") is not True or
                r1.get("native_PG_readonly_snapshot") is not True or
                exporter.get("kind") != "slr_native_pg_leech_source_export" or
                exporter.get("output_sha256") != jsonl_sha or
                exporter.get("F5_source_read") is not False or
                exporter.get("candidate_parser_promoted_to_truth") is not False or
                inventory.get("all_manifest_source_revisions_present") is not True or
                inventory.get("source_document_identity_unique") is not True or
                inventory.get("source_count") != exporter.get("source_count") or
                inventory.get("train_count") != exporter.get("train_count") or
                inventory.get("development_count") != exporter.get("development_count")):
            raise ValueError("standalone native R1 export receipt not source-complete")
    elif kind == "x369_three_repo_physical_maxcut":
        stages = r1.get("stage_execution_receipts", {})
        stage = stages.get("slr_native_or_legacy_pg_cpu", {})
        if (r1.get("all_stage_exit_codes_zero") is not True or
                stage.get("output_sha256") != jsonl_sha or
                stage.get("exit_code") != 0):
            raise ValueError("full DAG native R1 source stage incomplete")
    else:
        raise ValueError("unsupported native R1 receipt kind")
    if (r1.get("original_SLR_source_export_sha256") != jsonl_sha or
            r1.get("original_tokenizer_sha256") != tokenizer_sha):
        raise ValueError("R1 SLR output/tokenizer hash mismatch")
    verified = r1.get("verified_parser_source_manifest", {})
    if (verified.get("source_splits") != ["development", "train"] or
            verified.get("F5_reserved_source_read") is not False):
        raise ValueError("R1 original source split/lockbox manifest not verified")
    return {
        "parser_authority": r1["parser_authority"],
        "original_SLR_source_export_sha256": jsonl_sha,
        "original_tokenizer_sha256": tokenizer_sha,
        "source_manifest_rows": verified.get("source_manifest_rows"),
    }


def validate_source_docs(records, cfg):
    train = []
    development = []
    ids = set()
    digests = {}
    for record in records:
        key, source_sha = record["id"], record["source_sha"]
        if key in ids or source_sha in digests:
            raise ValueError("repeat source document or source bytes across source records")
        ids.add(key)
        digests[source_sha] = record["split"]
        if (record["alignment"].visibility != "full_context" or
                not 2 <= len(record["token_ids"]) <= cfg.block_size or
                min(record["token_ids"]) < 0 or
                max(record["token_ids"]) >= cfg.vocab_size):
            raise ValueError("original full-context source document exceeds model contract")
        if record["split"] == "train":
            train.append(record)
        elif record["split"] == "development":
            development.append(record)
        else:
            raise ValueError("reserved/F5 or unknown source split forbidden")
    if not train or not development:
        raise ValueError("both real train and source-disjoint development required")
    return train, development


@torch.no_grad()
def per_document_ce(model, records, device):
    model.eval()
    values = []
    for record in records:
        token = torch.as_tensor(record["token_ids"], dtype=torch.long, device=device)[None, :]
        logits = model(token, student_only=True)[0]
        ce = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.shape[-1]),
            token[:, 1:].reshape(-1))
        if not bool(torch.isfinite(ce)):
            raise ValueError("nonfinite original document CE")
        values.append({
            "document_id": record["id"],
            "source_sha256": record["source_sha"],
            "target_tokens": token.shape[1] - 1,
            "document_ce_nats": float(ce),
        })
    total = sum(x["target_tokens"] for x in values)
    mean = sum(x["document_ce_nats"] * x["target_tokens"] for x in values) / total
    return values, mean


@contextmanager
def source_mixed_mask_diversity_capture(model):
    """Observe actual per-token source q-gates, without copying activations.

    Only a first boolean mask and an accumulated difference bit are
    retained for each layer; no input F5 tokens or full hidden states.
    """
    patches=[]
    metrics=[]
    try:
        for block in model.blocks:
            moe=block.moe
            if (moe is None or
                    moe.route_geometry!="c2_c3_mixed_null_residual"):
                raise ValueError("q-gate diversity capture needs actual mixed R4 source")
            original=moe.select_source_experts
            stat={"first":None, "changed":None, "source_states":0}
            def observed(x,history,*,orig=original,record=stat,layer=moe):
                mask=q_gated_mixed_c3_mask(x,layer._c3_mixed_null_mask)
                if record["first"] is None:
                    record["first"]=mask[0].detach().clone()
                changed=(mask!=record["first"]).any()
                record["changed"]=(
                    changed if record["changed"] is None else
                    record["changed"] | changed)
                record["source_states"]+=mask.shape[0]
                return orig(x,history)
            moe.select_source_experts=observed
            patches.append((moe,original))
            metrics.append(stat)
        yield metrics
    finally:
        for moe,original in patches:
            moe.select_source_experts=original


def train_arm(source, geometry, train_docs, dev_docs, *, args, seed,
              delta_dir=None, source_sha256=None, source_jsonl_sha256=None):
    torch.manual_seed(seed)
    model = clone_c2_with_zero_c3_residual(source, geometry).to(args.device)
    model.eval()  # deterministic, no dropout; gradients still allowed
    residual = []
    for param in model.parameters():
        param.requires_grad_(False)
        param.grad = None
    for block in model.blocks:
        param = block.moe.c3_residual_weight
        param.requires_grad_(True)
        residual.append(param)
    if len(residual) != model.cfg.n_layers:
        raise AssertionError("not all original C2 layers have residual")
    parameter_count = sum(p.numel() for p in residual)
    optimizer = torch.optim.AdamW(residual, lr=args.lr, weight_decay=0.)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    start = time.perf_counter()
    mask_diversity = None
    if geometry == "c2_c3_mixed_null_residual":
        with source_mixed_mask_diversity_capture(model) as observations:
            initial_rows, initial_ce = per_document_ce(
                model, dev_docs, args.device)
        changed=[bool(s["changed"].item()) for s in observations
                 if s["changed"] is not None]
        seen=[s["source_states"] for s in observations]
        mask_diversity={
            "source_observations_per_layer":seen,
            "layers_with_distinct_sample_masks":sum(changed),
            "all_observed_source_layers_count":len(observations),
            "nonfixed_source_orientation_measured":any(changed),
        }
        if not any(changed):
            raise ValueError(
                "R4 q-mixed control collapsed to fixed gauge on original development states")
    else:
        initial_rows, initial_ce = per_document_ce(model, dev_docs, args.device)
    trace = [{"step": 0, "development_ce_nats": initial_ce}]
    best = initial_ce
    best_step = 0
    best_residual = [p.detach().cpu().clone() for p in residual]
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    for step in range(1, args.steps + 1):
        # Identical source-document sampling across all arms for this seed.
        idx = int(torch.randint(len(train_docs), (1,), generator=generator).item())
        record = train_docs[idx]
        ids = torch.as_tensor(record["token_ids"], device=args.device,
                              dtype=torch.long)[None, :]
        optimizer.zero_grad(set_to_none=True)
        logits = model(ids, student_only=True)[0]
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.shape[-1]),
            ids[:, 1:].reshape(-1))
        if not bool(torch.isfinite(loss)):
            raise ValueError("nonfinite original source training loss")
        loss.backward()
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all())
               for p in residual):
            raise ValueError("source residual lacks finite gradients")
        torch.nn.utils.clip_grad_norm_(residual, args.clip_norm)
        optimizer.step()
        if step % args.eval_every == 0 or step == args.steps:
            _, dev_ce = per_document_ce(model, dev_docs, args.device)
            trace.append({"step": step, "fit_ce_nats": float(loss.detach()),
                          "development_ce_nats": dev_ce})
            if dev_ce < best:
                best, best_step = dev_ce, step
                best_residual = [p.detach().cpu().clone() for p in residual]
    for p, saved in zip(residual, best_residual):
        p.data.copy_(saved.to(p.device))
    final_rows, final_ce = per_document_ce(model, dev_docs, args.device)
    if abs(final_ce - best) > 1e-6:
        raise AssertionError("development best-step rollback CE mismatch")
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    result = {
        "geometry": geometry, "seed": seed, "steps": args.steps,
        "initial_ce_nats": initial_ce, "best_dev_ce_nats": best,
        "best_step": best_step, "residual_trainable_parameter_count": parameter_count,
        "development_rows": final_rows, "step_trace": trace,
        "wall_seconds": time.perf_counter() - start,
        "peak_gpu_vram_bytes": (
            torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else None),
        "initial_development_rows": initial_rows,
        "mixed_null_source_mask_diversity": mask_diversity,
    }
    # Archive the best-step trainable delta, not the original frozen model.
    delta_path = None
    if delta_dir is not None:
        if source_sha256 is None or source_jsonl_sha256 is None:
            raise ValueError("R4 delta archive requires original source hashes")
        delta_path = Path(delta_dir) / f"r4_seed_{seed}_{geometry}.pt"
        if delta_path.exists():
            raise FileExistsError("refuse overwriting a fitted R4 residual")
        payload = make_residual_delta_payload(
            model, source_sha256=source_sha256,
            source_jsonl_sha256=source_jsonl_sha256,
            seed=seed, best_step=best_step)
        torch.save(payload, delta_path)
        result["fitted_delta_artifact"] = delta_path.name
        result["fitted_delta_sha256"] = digest(delta_path)
        result["fitted_delta_contains_full_source_weights"] = False
    del optimizer, model, residual
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    if delta_path is not None:
        # Never hold two original-source models on the RX580 simultaneously.
        loaded = torch.load(delta_path, map_location="cpu", weights_only=True)
        restored = restore_residual_delta(
            source, loaded, source_sha256=source_sha256,
            source_jsonl_sha256=source_jsonl_sha256).to(args.device).eval()
        replay_rows, replay_ce = per_document_ce(restored, dev_docs, args.device)
        if (abs(replay_ce - final_ce) > 1e-6 or
                len(replay_rows) != len(final_rows) or
                any(a["document_id"] != b["document_id"] or
                    a["source_sha256"] != b["source_sha256"] or
                    abs(a["document_ce_nats"] -
                        b["document_ce_nats"]) > 1e-6
                    for a, b in zip(replay_rows, final_rows))):
            raise AssertionError(
                "saved R4 delta cannot reproduce selected real-source CE")
        result["saved_delta_replay_same_device_verified"] = True
        result["reloaded_development_ce_nats"] = replay_ce
        del restored
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    return result



@torch.no_grad()
def physical_device_initial_parity(source, record, device):
    """Original data, SAME physical device, one complete model at a time.

    The RX580 cannot hold four 12-layer full models at once. The *same*
    source model is moved to GPU, sampled, and returned to CPU before each
    residual clone is materialized. Unlike the CPU preflight this gate
    actually exercises HIP routing, expert transitions and output logits.
    """
    token = torch.as_tensor(
        record["token_ids"], dtype=torch.long, device=device)[None, :]
    def snapshot(model):
        model = model.to(device).eval()
        out = model(token, student_only=True, return_expert_histories=True)
        snap = (
            out[0].detach().cpu().clone(),
            out[1].detach().cpu().clone(),
            [h.detach().cpu().clone() for h in out[4]])
        model.cpu()
        del out
        if device.startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        return snap
    base = snapshot(source)
    verified = []
    for geometry in R4_GEOMETRIES:
        model = clone_c2_with_zero_c3_residual(source, geometry)
        arm = snapshot(model)
        if (not torch.equal(base[0], arm[0]) or
                not torch.equal(base[1], arm[1]) or
                len(base[2]) != len(arm[2]) or
                any(not torch.equal(a, b) for a, b in zip(base[2], arm[2]))):
            raise AssertionError("R4 same-device original source recurrent function mismatch: " + geometry)
        verified.append(geometry)
        del model, arm
    return {
        "physical_device": device,
        "verified_original_document_id": record["id"],
        "verified_original_source_sha256": record["source_sha"],
        "bitwise_initial_logits_hidden_and_history_equal_for_arms": verified,
    }

def run(args):
    if args.output.exists():
        raise FileExistsError("R4b refuses overwriting a measured receipt")
    delta_dir = args.output.with_name(args.output.stem + "_deltas")
    if delta_dir.exists():
        raise FileExistsError("R4 fitted delta artifact directory already exists")
    if (args.steps < 1 or args.seeds < 1 or args.eval_every < 1 or
            args.lr <= 0 or args.clip_norm <= 0 or args.cpu_threads < 1):
        raise ValueError("R4b budget must be strictly positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("RX580/HIP CUDA runtime unavailable")
    torch.set_num_threads(args.cpu_threads)
    r1 = json.loads(args.r1_summary.read_text(encoding="utf-8"))
    provenance = validated_r1_summary(
        r1, jsonl_sha=digest(args.documents_jsonl),
        tokenizer_sha=digest(args.tokenizer))
    if digest(args.repaired_c2_checkpoint) != args.repaired_c2_sha256:
        raise ValueError("original repaired C2 checkpoint SHA mismatch")
    source_receipt = json.loads(args.source_repair_receipt.read_text())
    if (source_receipt.get("kind") != "native_c2_bounded_early_even_warmstart" or
            source_receipt.get("output_sha256") != args.repaired_c2_sha256):
        raise ValueError("original repaired C2 source/receipt not authenticated")
    state, info = checkpoint_parts(args.repaired_c2_checkpoint)
    cfg = infer_config(state, info, args.config)
    if (cfg.moe_route_geometry != "c2_invariant" or
            not cfg.moe_guarded_ffn or
            cfg.moe_shared_e8_history or
            cfg.moe_late_history_cap != 64.):
        raise ValueError("expected genuine bounded R64 guarded C2 architecture")
    with torch.random.fork_rng(devices=[]):
        source = LeechGPT(cfg).eval()
    source.load_state_dict(state, strict=True)
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    records = load_docs(args.documents_jsonl, sp, provenance["original_tokenizer_sha256"])
    train, development = validate_source_docs(records, cfg)
    probe = torch.tensor(development[0]["token_ids"], dtype=torch.long)[None, :]
    c2 = clone_c2_with_zero_c3_residual(source, "c2_c2_residual")
    coherent = clone_c2_with_zero_c3_residual(source, "c2_c3_residual")
    mixed = clone_c2_with_zero_c3_residual(source, "c2_c3_mixed_null_residual")
    exact = assert_identical_initial_student_rollout(
        source, coherent, mixed, probe, c2_control=c2)
    del c2, coherent, mixed
    physical_parity = physical_device_initial_parity(
        source, development[0], args.device)
    delta_dir.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": 1,
        "kind": "x369_r4b_original_slr_three_capacity_matched_development",
        "r1_summary_sha256": digest(args.r1_summary),
        "source_repair_receipt_sha256": digest(args.source_repair_receipt),
        "repaired_c2_checkpoint_sha256": args.repaired_c2_sha256,
        "native_R1_provenance": provenance,
        "source_train_document_count": len(train),
        "source_development_document_count": len(development),
        "source_train_document_sha256": [rec["source_sha"] for rec in train],
        "source_development_document_sha256": [rec["source_sha"] for rec in development],
        "source_C2_initial_parity": exact,
        "source_physical_device_initial_parity": physical_parity,
        "source_both_splits_distinct": True,
        "source_history_repaired_R64": True,
        "trainable_budget_equal_per_arm": True,
        "historical_C2_training_corpus_document_independence_independently_verified": False,
        "typed_C3_input_label_action_verified": False,
        "phase_consumer_intervention_verified": False,
        "development_selection_only": True,
        "R4_scientific_release_admitted": False,
        "F5_raw_or_reserved_opened": False,
        "device": args.device,
        "seeds": list(range(args.seed, args.seed + args.seeds)),
        "steps_per_arm": args.steps,
        "fitted_residual_delta_directory": delta_dir.name,
        "trained_delta_same_device_replay_required": True,
        "feature_semantics": "x369_r4_q_gated_mixed_v2",
        "geometry_source_sha256": r4_geometry_source_fingerprint(),
        "fixed_mixed_C3_is_reparameterization_gauge": True,
        "phase_null_requires_sample_varying_q_mask": True,
        "nonfixed_mask_source_variation_empirically_verified": False,
        "arms": [],
        "paired": [],
    }
    for seed in result["seeds"]:
        arms = {}
        for geometry in R4_GEOMETRIES:
            arm = train_arm(
                source, geometry, train, development, args=args,
                seed=seed, delta_dir=delta_dir,
                source_sha256=args.repaired_c2_sha256,
                source_jsonl_sha256=provenance["original_SLR_source_export_sha256"])
            result["arms"].append(arm)
            arms[geometry] = arm
        if len({arms[g]["residual_trainable_parameter_count"]
                for g in R4_GEOMETRIES}) != 1:
            raise AssertionError("C2, coherent C3 and mixed C3 capacity mismatch")
        left = arms["c2_c2_residual"]["best_dev_ce_nats"]
        right = arms["c2_c3_residual"]["best_dev_ce_nats"]
        null = arms["c2_c3_mixed_null_residual"]["best_dev_ce_nats"]
        result["paired"].append({
            "seed": seed, "c2_best_dev_ce_nats": left,
            "coherent_c3_best_dev_ce_nats": right,
            "mixed_null_best_dev_ce_nats": null,
            "coherent_minus_c2": right - left,
            "coherent_minus_mixed": right - null,
            "coherent_selective_development_advantage_over_both": right < min(left, null),
        })
    if not all(
            row["mixed_null_source_mask_diversity"] is not None and
            row["mixed_null_source_mask_diversity"][
                "nonfixed_source_orientation_measured"] is True
            for row in result["arms"]
            if row["geometry"] == "c2_c3_mixed_null_residual"):
        raise AssertionError("R4 source-varying phase null not proven in actual development")
    result["nonfixed_mask_source_variation_empirically_verified"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "kind": result["kind"], "development_only": True,
        "paired": result["paired"], "F5_raw_or_reserved_opened": False,
    }, sort_keys=True, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("r1-summary", "documents-jsonl", "tokenizer",
                 "repaired-c2-checkpoint", "source-repair-receipt", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--repaired-c2-sha256", required=True)
    p.add_argument("--config", type=Path)
    p.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--steps", type=int, default=2048)
    p.add_argument("--eval-every", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--clip-norm", type=float, default=1.)
    p.add_argument("--cpu-threads", type=int, default=2)
    run(p.parse_args())


if __name__ == "__main__":
    main()
