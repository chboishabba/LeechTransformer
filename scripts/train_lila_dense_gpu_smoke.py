"""GPU-native original dense Leech-LILA fixed-token training and true resume.

No C2 model, CPU tensor execution, ROCm patches or Nix packaging here.
Authentic checkpoint/tokens; 200-step bounded experiment with 25-step
atomically reloaded checkpoints and a one-time process-boundary resume.

Original Leech-LILA: Anatolii Kornienko, DOI 10.5281/zenodo.18784424.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.lila_dense_training_parity import load_dense_teacher, validate_tokens
from scripts.gfx803_moe_gpu_gate import file_sha256

SCHEMA = 1


def to_cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu").clone()
    if isinstance(value, dict):
        return {k: to_cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(to_cpu_tree(v) for v in value)
    return value


def same_tree(a, b):
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
                and a.dtype == b.dtype and a.shape == b.shape
                and torch.equal(a.cpu(), b.cpu()))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same_tree(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(
            same_tree(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    try:
        temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def atomic_torch(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    try:
        torch.save(obj, temp)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def deterministic_indices(nrows: int, batch: int, step: int, seed: int):
    """One without-replacement reshuffle each full epoch of batch draws.

    Every absolute step has the same rows whether trained continuously
    or after process restart. All training rows are indexed by their
    original positions; no streaming corpus or development split needed.
    """
    if nrows < batch or batch < 1 or step < 1:
        raise ValueError("invalid fixed-window training data or absolute step")
    batches_per_epoch = (nrows + batch - 1) // batch
    epoch, ordinal = divmod(step - 1, batches_per_epoch)
    order = np.random.default_rng(np.random.SeedSequence(
        [int(seed), int(epoch)])).permutation(nrows)
    first = ordinal * batch
    # Wrap within the SAME epoch permutation; no omitted rows.
    return [int(order[(first + j) % nrows]) for j in range(batch)]


def settings(args, teacher_sha, token_sha, config_sha):
    return {
        "schema": SCHEMA, "teacher_sha256": teacher_sha,
        "train_tokens_sha256": token_sha,
        "config_sha256": config_sha,
        "batch": int(args.batch),
        "lr": float(args.learning_rate),
        "seed": int(args.seed),
        "grad_clip": float(args.grad_clip),
        "save_every": int(args.save_every),
        "target_steps": int(args.steps),
        "train_mode": "train; dropout RNG checkpointed",
        "dtype": "float32", "optimizer": "AdamW",
        "objective": "next-token cross entropy",
        "sampling": "seed/absolute-step epoch permutation",
    }


def assert_gpu():
    if not (torch.cuda.is_available() and torch.version.hip is not None):
        raise RuntimeError("ROCm/HIP GPU required; no CPU fallback")
    props = torch.cuda.get_device_properties(0)
    arch = getattr(props, "gcnArchName", None)
    if arch is not None and not str(arch).startswith("gfx803"):
        raise RuntimeError("selected HIP device is not gfx803")
    return {"name": torch.cuda.get_device_name(0),
            "architecture": arch,
            "memory_bytes": int(props.total_memory),
            "torch": torch.__version__, "hip": torch.version.hip}


def _capture_rng():
    # torch.save(weights_only=True) accepts primitive tuple/list structures,
    # not pickled custom random-state classes.
    ns = np.random.get_state()
    return {"torch_cpu": torch.get_rng_state().cpu().clone(),
            "torch_gpu_all": [x.cpu().clone() for x in torch.cuda.get_rng_state_all()],
            "numpy": (ns[0], ns[1].tolist(), int(ns[2]),
                      int(ns[3]), float(ns[4])),
            "python": random.getstate()}


def _restore_rng(saved):
    torch.set_rng_state(saved["torch_cpu"].cpu())
    torch.cuda.set_rng_state_all([x.cpu() for x in saved["torch_gpu_all"]])
    n = saved["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32),
                         int(n[2]), int(n[3]), float(n[4])))
    random.setstate(saved["python"])


def _payload(model, optimizer, run_settings, step, trace):
    return {
        "schema_version": SCHEMA,
        "model_state_dict": to_cpu_tree(model.state_dict()),
        "optimizer_state_dict": to_cpu_tree(optimizer.state_dict()),
        "rng_state": _capture_rng(),
        "settings": run_settings,
        "step": int(step),
        "trace": list(trace),
        "artifact_role": "dense-LILA training smoke, not C2/quality promotion",
    }


def verify_saved_checkpoint(path, original):
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("model_state_dict", "optimizer_state_dict", "rng_state",
                "settings", "step", "trace", "schema_version"):
        if key not in loaded or not same_tree(loaded[key], original[key]):
            raise RuntimeError("save/reload mismatch at " + key)
    return loaded


def _optimizer_state_compatible(optimizer, state):
    optimizer.load_state_dict(state)
    # A cold-reconstructed optimizer can contain CPU tensors before/after
    # loading. PyTorch performs correct per-parameter device transfer here.
    for param, tensor_map in optimizer.state.items():
        for key, value in tensor_map.items():
            if (isinstance(value, torch.Tensor) and value.numel() > 1
                    and value.device != param.device):
                raise RuntimeError("resumed AdamW state device mismatch: " + key)


def train(model, rows, args, *, optimizer=None, previous=None, progress=None,
          device="cuda:0"):
    """One bounded invocation. Caller owns HIP attestation and checkpoint I/O."""
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    start_step = 0 if previous is None else int(previous["step"])
    stop = args.steps if args.stop_after is None else args.stop_after
    if not (0 <= start_step < stop <= args.steps):
        raise ValueError("resume start / stop / target step contract violated")
    if args.save_every < 1 or args.batch < 1 or args.steps < 1 or args.grad_clip <= 0:
        raise ValueError("invalid smoke training hyperparameters")
    if args.batch > len(rows):
        raise ValueError("batch exceeds original saved token rows")
    model.train()
    is_gpu = str(device).startswith("cuda")
    if is_gpu:
        torch.cuda.reset_peak_memory_stats()
    trace = [] if previous is None else list(previous["trace"])
    elapsed_train_s = 0.
    steps_this_run = 0
    paths = []
    for step in range(start_step + 1, stop + 1):
        indices = deterministic_indices(len(rows), args.batch, step, args.seed)
        ids = torch.as_tensor(rows[indices].copy(), dtype=torch.long, device=device)
        if is_gpu:
            torch.cuda.synchronize()
        tic = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits = model(ids)[0]
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("nonfinite GPU dense logits")
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                               ids[:, 1:].reshape(-1))
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite dense loss")
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        if not grads or any(g is None or not bool(torch.isfinite(g).all()) for g in grads):
            raise FloatingPointError("missing/nonfinite full-model gradient")
        grad_norm = nn.utils.clip_grad_norm_(
            model.parameters(), args.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        if is_gpu:
            torch.cuda.synchronize()
        dt = time.perf_counter() - tic
        elapsed_train_s += dt
        steps_this_run += 1
        tokens = len(indices) * (ids.shape[1] - 1)
        item = {"step": step, "train_loss_nats": float(loss.detach().cpu()),
                "grad_norm_preclip": float(grad_norm.detach().cpu()),
                "target_tokens": int(tokens),
                "train_step_seconds": dt,
                "train_tokens_per_second": tokens / max(dt, 1e-12),
                "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if is_gpu else None,
                "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()) if is_gpu else None,
                "row_indices": indices}
        trace.append(item)
        if progress:
            progress(item)
        if step % args.save_every == 0 or step == stop:
            payload = _payload(model, optimizer, args.run_settings, step, trace)
            filename = args.output_dir / f"dense_step_{step:06d}.pt"
            if filename.exists():
                raise FileExistsError("refuse to overwrite checkpoint: " + str(filename))
            atomic_torch(filename, payload)
            verify_saved_checkpoint(filename, payload)
            paths.append({"step": step, "path": str(filename),
                          "sha256": file_sha256(filename),
                          "reload_verified": True})
        del ids, logits, loss, grads, grad_norm
    output = {
        "schema_version": SCHEMA,
        "scope": "native gfx803 dense LILA smoke; not C2 or final quality",
        "admitted_for_scope": True,
        "start_step": start_step, "end_step": stop,
        "target_steps": args.steps,
        "steps_this_process": steps_this_run,
        "training_seconds_excluding_checkpoint": elapsed_train_s,
        "mean_training_tokens_per_second": (
            sum(v["target_tokens"] for v in trace[start_step:]) /
            max(elapsed_train_s, 1e-12)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if is_gpu else None,
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()) if is_gpu else None,
        "last_loss_nats": trace[-1]["train_loss_nats"],
        "checkpoints_written": paths,
        "resume_verified": previous is not None,
        "trace": trace,
        "settings": args.run_settings,
    }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--train-ids", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--grad-clip", type=float, default=1.)
    args = parser.parse_args()
    # Immutable per-process evidence: never overwrite a previous receipt,
    # including an interrupted/failing attempt.
    if args.receipt.exists():
        raise FileExistsError("refuse existing receipt: " + str(args.receipt))
    receipt = {"status": "not-admitted", "native_gpu_only": True,
               "no_c2": True, "checkpoint_resume_verified": False}
    try:
        if args.steps < 1 or args.save_every < 1 or args.batch < 1:
            raise ValueError("invalid step/save/batch settings")
        if args.learning_rate <= 0 or args.grad_clip <= 0:
            raise ValueError("invalid learning rate or clip")
        gpu = assert_gpu()
        # Validate the actual original checkpoint BEFORE touching any output.
        teacher_hash = file_sha256(args.teacher)
        tokens_hash = file_sha256(args.train_ids)
        cfg_hash = file_sha256(args.config) if args.config else None
        model = load_dense_teacher(args.teacher, args.config)
        rows = np.load(args.train_ids, allow_pickle=False)
        validate_tokens(rows, model, args.batch)
        args.run_settings = settings(args, teacher_hash, tokens_hash, cfg_hash)
        receipt.update({"gpu": gpu, "settings": args.run_settings,
                        "training_rows": len(rows),
                        "sequence_length": int(rows.shape[1])})
        args.output_dir.mkdir(parents=True, exist_ok=True)
        model.to("cuda:0")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        previous = None
        if args.resume is None:
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            np.random.seed(args.seed)
            random.seed(args.seed)
        else:
            previous = torch.load(args.resume, map_location="cpu",
                                  weights_only=True)
            if previous.get("schema_version") != SCHEMA:
                raise RuntimeError("incompatible resume checkpoint schema")
            if previous.get("settings") != args.run_settings:
                raise RuntimeError("resume settings or source identity mismatch")
            if previous.get("step", 0) < 1:
                raise RuntimeError("resume checkpoint has no completed step")
            model.load_state_dict(previous["model_state_dict"], strict=True)
            _optimizer_state_compatible(optimizer, previous["optimizer_state_dict"])
            _restore_rng(previous["rng_state"])
            receipt["resumed_checkpoint_sha256"] = file_sha256(args.resume)
            receipt["resumed_step"] = previous["step"]
            receipt["checkpoint_resume_verified"] = True
        result = train(model, rows, args, optimizer=optimizer, previous=previous,
                       progress=lambda item: print(
                           f"[lila-gpu] step {item['step']}/{args.steps} "
                           f"CE={item['train_loss_nats']:.5f} "
                           f"tok/s={item['train_tokens_per_second']:.1f} "
                           f"peakMiB={item['gpu_peak_allocated_bytes']/1048576:.1f}",
                           flush=True))
        receipt.update(result)
        receipt["status"] = "complete" if result["end_step"] == args.steps else "partial-restart-required"
        receipt["checkpoint_resume_verified"] = previous is not None
    except Exception as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        # This receipt is distinct from the on-disk saved training checkpoint;
        # GPU fault may kill Python before the finally block executes.
        if args.receipt.exists():
            raise FileExistsError("refuse overwrite of simultaneous receipt: " +
                                  str(args.receipt))
        atomic_json(args.receipt, receipt)
        print(json.dumps({"status": receipt["status"],
                          "receipt": str(args.receipt),
                          "error": receipt.get("error")}), flush=True)


if __name__ == "__main__":
    main()
