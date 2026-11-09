"""Original dense Leech-LILA: independent CPU/HIP gradient and AdamW parity.

This is a NO-C2 diagnostic on authentic dense checkpoint and token windows.
It neither patches ROCm nor launches persistent model training. It records
all-parameter backward agreement and optimizer state evolution before a
separate, deliberate training deployment can be admitted.

Original Leech-LILA: Anatolii Kornienko, DOI 10.5281/zenodo.18784424.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import platform
import sys
import traceback

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from models.model import LeechGPT
from scripts.verify_trained_federated_moe import checkpoint_parts, infer_config
from scripts.gfx803_moe_gpu_gate import comparison, file_sha256, write_progress
from scripts.gfx803_runtime_attestation import attest_runtime


def load_dense_teacher(checkpoint: Path, config: Path | None):
    weights, info = checkpoint_parts(checkpoint)
    cfg = infer_config(weights, info, config)
    if cfg.moe_experts != 0 or cfg.moe_guarded_ffn:
        raise ValueError("expected original dense teacher, not an MoE checkpoint")
    model = LeechGPT(cfg)
    model.load_state_dict(weights, strict=True)
    return model.eval()


def validate_tokens(rows: np.ndarray, model: LeechGPT, batch: int) -> None:
    if (not isinstance(rows, np.ndarray) or rows.ndim != 2
            or not np.issubdtype(rows.dtype, np.integer)
            or batch < 1 or len(rows) < batch or rows.shape[1] < 2
            or rows.shape[1] > model.cfg.block_size
            or np.any(rows < 0) or np.any(rows >= model.cfg.vocab_size)):
        raise ValueError("requires original checkpoint-compatible consecutive token rows")


def dense_train_loss(model, ids):
    # Real autoregressive next-token loss, not synthetic activation loss.
    logits = model(ids)[0]
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("nonfinite dense logits")
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        ids[:, 1:].reshape(-1))
    if not bool(torch.isfinite(loss)):
        raise ValueError("nonfinite dense next-token loss")
    return loss, logits


def compared_gradients(cpu_model, gpu_model, *, atol=2e-3, rtol=2e-3):
    a = dict(cpu_model.named_parameters())
    b = dict(gpu_model.named_parameters())
    if list(a) != list(b):
        raise AssertionError("parameter names differ across devices")
    if not a:
        raise AssertionError("dense model has no parameters")
    failures = []
    max_difference = 0.
    counted = 0
    for name in a:
        p, q = a[name], b[name]
        if p.grad is None or q.grad is None:
            failures.append({"name": name, "reason": "missing gradient"})
            continue
        score = comparison(p.grad, q.grad, atol=atol, rtol=rtol)
        counted += 1
        max_difference = max(max_difference, score.get("max_abs_diff", 0.))
        if not score["passed"]:
            failures.append({"name": name, **score})
    return {"passed": not failures and counted == len(a),
            "checked": counted, "expected": len(a),
            "max_abs_diff": max_difference,
            "failures": failures[:20], "failure_count": len(failures)}


def compared_parameters(cpu_model, gpu_model, *, atol=2e-3, rtol=2e-3):
    a = dict(cpu_model.named_parameters())
    b = dict(gpu_model.named_parameters())
    if list(a) != list(b):
        raise AssertionError("parameter names differ")
    failed = []
    largest = 0.
    for name in a:
        c = comparison(a[name], b[name], atol=atol, rtol=rtol)
        largest = max(largest, c.get("max_abs_diff", 0.))
        if not c["passed"]:
            failed.append({"name": name, **c})
    return {"passed": not failed, "checked": len(a),
            "max_abs_diff": largest, "failure_count": len(failed),
            "failures": failed[:20]}


def dense_trajectory(teacher, rows, *, device="cuda:0", steps=20,
                     repeats=3, lr=3e-4, atol=1e-3, rtol=1e-3,
                     progress_path=None):
    if steps < 1 or repeats < 2 or lr <= 0:
        raise ValueError("need positive steps/learning rate and >=2 repeats")
    validate_tokens(rows, teacher, len(rows))
    # Deliberately NO C2 references, no routing or non-dense FFN substitution.
    cpu = teacher.cpu().eval()
    gpu = copy.deepcopy(cpu).to(device).eval()
    for model in (cpu, gpu):
        # Preserve deterministic gradient-bearing training of the original
        # dense teacher; dropout is disabled equally for both copies.
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.eval()
        for p in model.parameters():
            p.requires_grad_(True)
    cpu_ids = torch.as_tensor(rows, dtype=torch.long, device="cpu")
    gpu_ids = cpu_ids.to(device)
    cpu_params = list(cpu.parameters())
    gpu_params = list(gpu.parameters())
    opt_cpu = torch.optim.AdamW(cpu_params, lr=lr)
    opt_gpu = torch.optim.AdamW(gpu_params, lr=lr)
    report = {"admitted": False, "workload": "original dense LILA",
              "stage": "initialization", "steps_requested": steps,
              "steps_completed": 0, "repeats": repeats,
              "shape": list(cpu_ids.shape), "trajectory": [],
              "not_c2_training": True, "model_checkpoint_unchanged": True}
    for step in range(1, steps + 1):
        write_progress(progress_path, "dense-step-start", step=step)
        opt_cpu.zero_grad(set_to_none=True)
        opt_gpu.zero_grad(set_to_none=True)
        loss_cpu, logits_cpu = dense_train_loss(cpu, cpu_ids)
        loss_gpu, logits_gpu = dense_train_loss(gpu, gpu_ids)
        row = {"step": step}
        row["loss"] = comparison(loss_cpu, loss_gpu, atol=atol, rtol=rtol)
        row["forward"] = comparison(logits_cpu, logits_gpu,
                                    atol=atol, rtol=rtol)
        if not (row["loss"]["passed"] and row["forward"]["passed"]):
            report["trajectory"].append(row)
            report["stage"] = "forward/loss-step-" + str(step)
            return report
        # Any nondeterministic GPU forward is a fail even if its gradient
        # accidentally remains within tolerance.
        repeat_checks = []
        with torch.no_grad():
            for _ in range(1, repeats):
                repeated = gpu(gpu_ids)[0]
                repeat_checks.append(comparison(
                    logits_gpu, repeated, atol=1e-5, rtol=1e-5))
        row["repeats"] = {"passed": all(x["passed"] for x in repeat_checks),
                          "max_abs_diff": max(x.get("max_abs_diff", 0.)
                                              for x in repeat_checks)}
        if not row["repeats"]["passed"]:
            report["trajectory"].append(row)
            report["stage"] = "gpu-repeat-step-" + str(step)
            return report
        write_progress(progress_path, "dense-backward-start", step=step)
        loss_cpu.backward()
        loss_gpu.backward()
        row["gradients"] = compared_gradients(cpu, gpu)
        if not row["gradients"]["passed"]:
            report["trajectory"].append(row)
            report["stage"] = "backward-step-" + str(step)
            return report
        torch.nn.utils.clip_grad_norm_(cpu_params, 1.)
        torch.nn.utils.clip_grad_norm_(gpu_params, 1.)
        write_progress(progress_path, "dense-AdamW-start", step=step)
        opt_cpu.step()
        opt_gpu.step()
        row["parameters"] = compared_parameters(cpu, gpu)
        if not row["parameters"]["passed"]:
            report["trajectory"].append(row)
            report["stage"] = "AdamW-parameter-step-" + str(step)
            return report
        with torch.no_grad():
            after_cpu = cpu(cpu_ids)[0]
            after_gpu = gpu(gpu_ids)[0]
        row["post_update_logits"] = comparison(
            after_cpu, after_gpu, atol=atol, rtol=rtol)
        report["trajectory"].append(row)
        if not row["post_update_logits"]["passed"]:
            report["stage"] = "post-update-step-" + str(step)
            return report
        report["steps_completed"] = step
        print(f"[lila-dense] matched CPU/GPU step {step}/{steps}: passed",
              flush=True)
        write_progress(progress_path, "dense-step-complete", step=step)
    report["admitted"] = True
    report["stage"] = "bounded-dense-numerical-parity-complete"
    report["limits"] = ("Does not establish persistent training, full device "
                        "stability, final heldout quality, throughput, "
                        "or C2 correctness")
    return report


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--config", type=Path)
    p.add_argument("--train-ids", type=Path, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    p.add_argument("--progress-file", type=Path)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    a = p.parse_args()
    result = {"admitted": False, "scope": "dense-LILA only; no C2 or ROCm repair",
              "torch_version": torch.__version__, "torch_hip": torch.version.hip,
              "platform": platform.platform(), "python": sys.version.split()[0],
              "input_files": {}}
    try:
        result["input_files"] = {
            "teacher_sha256": file_sha256(a.teacher),
            "tokens_sha256": file_sha256(a.train_ids)}
        if a.config:
            result["input_files"]["config_sha256"] = file_sha256(a.config)
        if torch.version.hip is None or not torch.cuda.is_available():
            raise RuntimeError("No ROCm/HIP GPU; refusing CPU fallback")
        props = torch.cuda.get_device_properties(0)
        arch = getattr(props, "gcnArchName", None)
        if arch is not None and not str(arch).startswith("gfx803"):
            raise RuntimeError("selected GPU is not gfx803")
        result["gpu"] = {"name": torch.cuda.get_device_name(0),
                         "architecture": arch,
                         "total_memory_bytes": int(props.total_memory)}
        teacher = load_dense_teacher(a.teacher, a.config)
        token_rows = np.load(a.train_ids, allow_pickle=False)
        validate_tokens(token_rows, teacher, a.batch)
        result["probe_rows"] = list(range(a.batch))
        result["result"] = dense_trajectory(
            teacher, token_rows[:a.batch], steps=a.steps,
            repeats=a.repeats, lr=a.learning_rate,
            progress_path=a.progress_file)
        result["admitted"] = result["result"]["admitted"]
        result["effective_runtime_attestation"] = attest_runtime()
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback_tail"] = traceback.format_exc()[-1600:]
    finally:
        a.receipt.parent.mkdir(parents=True, exist_ok=True)
        a.receipt.write_text(json.dumps(result, indent=2) + "\n")
        write_progress(a.progress_file, "dense-finally",
                       admitted=result["admitted"],
                       result_stage=result.get("result", {}).get("stage"),
                       error=result.get("error"))
    print(json.dumps({"admitted": result["admitted"],
                      "receipt": str(a.receipt),
                      "stage": result.get("result", {}).get("stage"),
                      "error": result.get("error")}, indent=2), flush=True)
    return 0 if result["admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
