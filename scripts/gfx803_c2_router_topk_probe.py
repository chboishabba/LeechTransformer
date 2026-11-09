"""Fault-isolate C2 router ``topk`` on authentic checkpoint-derived logits.

This is deliberately narrower than the C2 model gate.  It records the router
scores produced on CPU for the actual normalized layer input and evolving C2
history, then tests exactly those score matrices on the selected HIP device.
It never trains, mutates a checkpoint, or treats a passing probe as admission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts.diagnose_moe_quality_collapse import load_pair
from scripts.gfx803_moe_gpu_gate import collect_teacher_ln2, comparison
from scripts.train_guarded_moe_continuation import build_guarded_hybrid


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_receipt(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def capture_cpu_router_logits(teacher, student, ids_cpu: torch.Tensor, layer: int):
    """Return pre-topk logits from every real router call in token order."""
    hybrid = build_guarded_hybrid(teacher, student).eval()
    moe = hybrid.blocks[layer].moe
    if moe is None:
        raise ValueError("selected layer has no C2 router")
    captured = []

    def hook(_module, _args, output):
        # ``HistoryOrbitMoE.step`` adds expert_bias immediately after this
        # Linear call; retain the exact tensor that reaches torch.topk.
        captured.append((output.detach().squeeze(-1) +
                         moe.expert_bias.detach()).clone())

    handle = moe.router.register_forward_hook(hook)
    try:
        _, normalized = collect_teacher_ln2(teacher.eval(), ids_cpu, layer)
        with torch.no_grad():
            moe(normalized)
    finally:
        handle.remove()
    if len(captured) != normalized.shape[1]:
        raise AssertionError("router capture count does not equal token count")
    expected = (ids_cpu.shape[0], moe.num_experts)
    if any(tuple(value.shape) != expected for value in captured):
        raise AssertionError("router score shape changed during sequential replay")
    return captured, moe.top_k, moe.route_geometry


def run(args) -> dict:
    if args.repeats < 2 or args.probe_batch < 1:
        raise ValueError("need at least two repeats and one authentic row")
    if torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError("HIP GPU unavailable; refusing CPU fallback")
    if args.receipt.exists():
        raise FileExistsError("refusing to overwrite existing evidence: " + str(args.receipt))
    rows = np.load(args.train_ids, allow_pickle=False)
    if rows.ndim != 2 or len(rows) < args.probe_batch:
        raise ValueError("probe batch exceeds authentic token rows")
    teacher, student = load_pair(args.teacher, args.student, args.config)
    ids_cpu = torch.as_tensor(rows[:args.probe_batch], dtype=torch.long)
    scores, top_k, geometry = capture_cpu_router_logits(
        teacher, student, ids_cpu, args.layer)
    receipt = {
        "scope": "authentic C2 router-score CPU-copy HIP topk microprobe; not model or training admission",
        "source_sha256": {"teacher": sha256(args.teacher), "student": sha256(args.student), "train": sha256(args.train_ids)},
        "gpu": torch.cuda.get_device_name(0),
        "torch_hip": torch.version.hip,
        "layer": args.layer,
        "route_geometry": geometry,
        "top_k": top_k,
        "score_shape": list(scores[0].shape),
        "score_frames": len(scores),
        "repeats": args.repeats,
        "frames": [],
        "passed": False,
        "stage": "cpu-router-capture-complete",
    }
    save_receipt(args.receipt, receipt)
    for index, score_cpu in enumerate(scores):
        values_cpu, ids_reference = torch.topk(score_cpu, top_k, dim=-1)
        frame = {"index": index, "score_sha256": hashlib.sha256(score_cpu.numpy().tobytes()).hexdigest(), "repeats": []}
        receipt["stage"] = "gpu-topk-start"
        receipt["frame_index"] = index
        save_receipt(args.receipt, receipt)
        score_gpu = score_cpu.to("cuda:0")
        for repeat in range(args.repeats):
            values_gpu, ids_gpu = torch.topk(score_gpu, top_k, dim=-1)
            torch.cuda.synchronize()
            value_check = comparison(values_cpu, values_gpu, atol=args.atol, rtol=args.rtol)
            ids_match = bool(torch.equal(ids_reference, ids_gpu.cpu()))
            frame["repeats"].append({"values": value_check, "ids_match": ids_match})
            if not (value_check["passed"] and ids_match):
                receipt["frames"].append(frame)
                receipt["stage"] = "gpu-topk-mismatch"
                save_receipt(args.receipt, receipt)
                return receipt
        receipt["frames"].append(frame)
        save_receipt(args.receipt, receipt)
    receipt["passed"] = True
    receipt["stage"] = "complete"
    save_receipt(args.receipt, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--train-ids", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--probe-batch", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as error:
        if not args.receipt.exists():
            save_receipt(args.receipt, {"passed": False, "stage": "python-exception", "error": repr(error)})
        raise
    print(json.dumps({"passed": result["passed"], "stage": result["stage"]}, sort_keys=True), flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
