"""Fail-closed extraction of a teacher-free MoE from a fully admitted hybrid.

Zero/intermediate quality gates still execute dense FFNs. They cannot be
exported as a smaller standalone student. This command requires all gates
equal to one, a successful *independent* heldout-quality receipt, the pinned
dense teacher and an independent teacher-free re-evaluation. The resulting
state contains no dense FFN or quality gate tensors. It is a candidate,
NOT a claim of E8/Leech equivalence or actual WAN speed.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from models.model import LeechConfig, LeechGPT
from scripts.diagnose_moe_quality_collapse import token_cross_entropy
from scripts.verify_trained_federated_moe import checkpoint_parts, infer_config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_teacher_free(hybrid: LeechGPT) -> LeechGPT:
    """Remove teacher tensors only when EVERY layer has alpha exactly 1."""
    if not hybrid.cfg.moe_guarded_ffn or hybrid.cfg.moe_experts < 2:
        raise ValueError("input must be a quality-guarded MoE")
    if any(block.moe is None or block.even_student is None or
           block.ffn is None or block.quality_gate is None
           for block in hybrid.blocks):
        raise ValueError("guarded parity student must be complete")
    if any(not bool(torch.isfinite(block.quality_gate).all()) or
           float(block.quality_gate.detach()) != 1.
           for block in hybrid.blocks):
        raise ValueError("not all quality gates equal 1; dense teacher still required")

    cfg = dataclasses.replace(hybrid.cfg, moe_guarded_ffn=False)
    destination = LeechGPT(cfg)
    old = hybrid.state_dict()
    new = destination.state_dict()
    for name in new:
        if name not in old or old[name].shape != new[name].shape:
            raise ValueError("missing or mismatched standalone tensor: " + name)
        new[name] = old[name].detach().clone()
    destination.load_state_dict(new, strict=True)
    if any(".ffn." in name or name.endswith(".quality_gate")
           for name in destination.state_dict()):
        raise AssertionError("dense teacher leaked into standalone checkpoint")
    return destination.eval()


@torch.no_grad()
def evaluate_teacher_free(teacher: LeechGPT, hybrid: LeechGPT,
                          standalone: LeechGPT, heldout: np.ndarray, *,
                          batch: int = 2, device: str = "cpu",
                          ce_budget_nats: float = .02,
                          max_logit_delta: float = 1e-4) -> dict:
    """Final check cannot tune gates or update weights."""
    if (batch < 1 or ce_budget_nats < 0 or max_logit_delta < 0 or
        heldout.ndim != 2 or len(heldout) < 1 or
        heldout.shape[1] < 2 or heldout.shape[1] > teacher.cfg.block_size or
        not np.issubdtype(heldout.dtype, np.integer) or
        np.any(heldout < 0) or np.any(heldout >= teacher.cfg.vocab_size)):
        raise ValueError("invalid heldout/threshold arguments")
    for model in (teacher, hybrid, standalone):
        model.eval().to(device)

    dense_ce = token_cross_entropy(teacher, heldout, batch=batch,
                                   device=device)["ce_nats"]
    hybrid_ce = token_cross_entropy(hybrid, heldout, batch=batch,
                                    device=device)["ce_nats"]
    student_ce = token_cross_entropy(standalone, heldout, batch=batch,
                                     device=device)["ce_nats"]
    max_delta = 0.
    finite = all(v is not None and np.isfinite(v)
                 for v in (dense_ce, hybrid_ce, student_ce))
    for start in range(0, len(heldout), batch):
        tokens = torch.as_tensor(heldout[start:start+batch],
                                 dtype=torch.long, device=device)
        guarded_logits = hybrid(tokens)[0]
        student_logits = standalone(tokens)[0]
        finite = finite and bool(torch.isfinite(guarded_logits).all()
                                 and torch.isfinite(student_logits).all())
        if finite:
            max_delta = max(max_delta, float(
                (guarded_logits - student_logits).abs().max()))
    old_params = sum(t.numel() for t in hybrid.parameters())
    new_params = sum(t.numel() for t in standalone.parameters())
    accepted = bool(finite and max_delta <= max_logit_delta and
                    student_ce <= dense_ce + ce_budget_nats and
                    hybrid_ce <= dense_ce + ce_budget_nats)
    return {
        "accepted": accepted,
        "dense_heldout_ce_nats": dense_ce,
        "hybrid_heldout_ce_nats": hybrid_ce,
        "standalone_heldout_ce_nats": student_ce,
        "ce_budget_nats": ce_budget_nats,
        "max_guarded_to_standalone_logit_abs_delta": max_delta,
        "max_logit_delta_budget": max_logit_delta,
        "all_logits_and_ce_finite": bool(finite),
        "guarded_parameter_count": old_params,
        "teacher_free_parameter_count": new_params,
        "removed_parameter_count": old_params - new_params,
        "all_dense_ffn_tensors_removed": True,
        "source": "final untouched heldout recheck; no gate or weight search",
        "limits": "Quality/teacher-free parity only. No measured wall time, federated WAN, semantic intervention, Leech gluing, or full E8 expert equivariance."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--guarded", required=True, type=Path)
    parser.add_argument("--heldout-ids", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ce-budget-nats", type=float, default=.02)
    parser.add_argument("--max-logit-delta", type=float, default=1e-4)
    args = parser.parse_args()

    checkpoint = torch.load(args.guarded, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("config"), dict):
        raise ValueError("guarded checkpoint lacks explicit configuration")
    receipt = checkpoint.get("guarded_quality_receipt")
    if not isinstance(receipt, dict) or not receipt.get("full_student_replacement_pass"):
        raise ValueError("guarded training did not certify full student replacement")
    if receipt.get("source_checkpoint_sha256") != sha256_file(args.teacher):
        raise ValueError("dense source checksum differs from guarded training receipt")

    cfg = LeechConfig(**checkpoint["config"])
    if not cfg.moe_guarded_ffn:
        raise ValueError("guarded flag missing from checkpoint")
    hybrid = LeechGPT(cfg)
    hybrid.load_state_dict(checkpoint["model_state_dict"], strict=True)
    standalone = extract_teacher_free(hybrid)

    teacher_state, teacher_meta = checkpoint_parts(args.teacher)
    teacher_cfg = infer_config(teacher_state, teacher_meta, args.config)
    if teacher_cfg.moe_experts or any(
            getattr(teacher_cfg, attr) != getattr(cfg, attr)
            for attr in ("vocab_size", "d_model", "n_layers", "n_heads",
                         "block_size")):
        raise ValueError("teacher/checkpoint architecture mismatch")
    teacher = LeechGPT(teacher_cfg)
    teacher.load_state_dict(teacher_state, strict=True)
    ids = np.load(args.heldout_ids, allow_pickle=False)
    report = evaluate_teacher_free(
        teacher, hybrid, standalone, ids, batch=args.batch,
        device=args.device, ce_budget_nats=args.ce_budget_nats,
        max_logit_delta=args.max_logit_delta)
    report.update({
        "teacher_checkpoint_sha256": sha256_file(args.teacher),
        "guarded_checkpoint_sha256": sha256_file(args.guarded),
        "heldout_ids_sha256": sha256_file(args.heldout_ids),
        "quality_gate_values": [float(b.quality_gate.detach())
                                for b in hybrid.blocks],
        "source_guarded_receipt_full_replacement": True,
        "source_attribution": "Original Leech-LILA: Anatolii Kornienko, DOI 10.5281/zenodo.18784424. E8/Leech theorem roles remain separate."})
    if not report["accepted"]:
        raise ValueError("final teacher-free quality/parity check failed: " +
                         json.dumps(report, sort_keys=True))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": standalone.cpu().state_dict(),
                "config": dataclasses.asdict(standalone.cfg),
                "teacher_free_promotion_receipt": report}, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
