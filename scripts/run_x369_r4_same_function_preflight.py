"""R4 CPU-only exact-initial-function admission for an ORIGINAL C2 checkpoint.

Creates no synthetic training corpus. The deterministic synthetic token probe
is solely a numerical witness for bitwise equality at initialization; it
cannot be scored as original-data learning, F3 or the frozen F5 lockbox.
No secrets, PostgreSQL connection, model tuning or GPU are needed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import torch

from models.model import LeechGPT
from models.x369_fair_c3_residual import (
    assert_identical_initial_student_rollout,
    clone_c2_with_zero_c3_residual,
)
from scripts.verify_trained_federated_moe import checkpoint_parts, infer_config


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args):
    if args.output.exists():
        raise FileExistsError("refuse overwriting an existing original R4 receipt")
    if args.batch < 1 or args.seq_len < 2:
        raise ValueError("nonempty >=2-token source-C2 parity probe required")
    torch.set_num_threads(args.cpu_threads)
    pinned = file_sha256(args.checkpoint)
    if pinned != args.checkpoint_sha256:
        raise ValueError("source C2 checkpoint does not match pinned SHA256")
    weights, info = checkpoint_parts(args.checkpoint)
    cfg = infer_config(weights, info, args.config)
    if (cfg.moe_route_geometry != "c2_invariant" or
            not cfg.moe_guarded_ffn or cfg.moe_experts < 2):
        raise ValueError("R4 requires source guarded independent-history C2 model")
    if args.seq_len > cfg.block_size:
        raise ValueError("probe exceeds original checkpoint context window")
    with torch.random.fork_rng(devices=[]):
        source = LeechGPT(cfg).eval()
    source.load_state_dict(weights, strict=True)
    coherent = clone_c2_with_zero_c3_residual(source, "c2_c3_residual")
    mixed_null = clone_c2_with_zero_c3_residual(
        source, "c2_c3_mixed_null_residual")
    c2_control = clone_c2_with_zero_c3_residual(
        source, "c2_c2_residual")
    generator = torch.Generator(device="cpu").manual_seed(412026)
    tokens = torch.randint(
        0, cfg.vocab_size, (args.batch, args.seq_len),
        dtype=torch.long, generator=generator)
    receipt = assert_identical_initial_student_rollout(
        source, coherent, mixed_null, tokens, c2_control=c2_control)
    receipt.update({
        "schema": 1,
        "original_c2_checkpoint_sha256": pinned,
        "original_config": {
            "d_model": cfg.d_model,
            "n_layers": cfg.n_layers,
            "moe_experts": cfg.moe_experts,
            "moe_top_k": cfg.moe_top_k,
            "moe_route_geometry": cfg.moe_route_geometry,
            "moe_late_history_cap": cfg.moe_late_history_cap,
        },
        "comparison_geometries": [
            "c2_c2_residual", "c2_c3_residual", "c2_c3_mixed_null_residual"],
        "probe_seed": 412026,
        "probe_is_deterministic_synthetic_tokens_not_source_data": True,
        "original_data_training_done": False,
        "actual_task_label_action_established": False,
        "original_C3_mechanism_causal_weld_established": False,
        "r4_full_empirical_release_admitted": False,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps(receipt, sort_keys=True, allow_nan=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--checkpoint-sha256", required=True)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=4)
    p.add_argument("--cpu-threads", type=int, default=2)
    args = p.parse_args()
    if args.cpu_threads < 1:
        raise ValueError("positive CPU thread budget required")
    run(args)


if __name__ == "__main__":
    main()
