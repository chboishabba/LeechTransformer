"""Run the actual Leech block-0 QKV projection as a profiler-safe micro-repro.

This deliberately does not use nn.Module hooks: they changed the failing gfx803
execution.  The input is reconstructed using the model's real embedding,
position and ``block0.ln1`` modules, then the original ``block0.attn.qkv``
module is called inside one ROCTX range.  It is an attribution aid only; it
never admits GPU training or claims that a profiled dispatch is identical to
an unprofiled one.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from models.model import LeechGPT
from scripts.gfx803_runtime_attestation import attest_runtime
from scripts.verify_trained_federated_moe import checkpoint_parts, infer_config


def sha256_tensor(value: torch.Tensor) -> str:
    data = value.detach().contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def qkv_input(model: LeechGPT, ids: torch.Tensor) -> torch.Tensor:
    """The exact block-0 QKV input from LeechGPT.forward, without hooks."""
    if ids.ndim != 2 or ids.shape[1] > model.cfg.block_size:
        raise ValueError("token IDs must be [batch, T<=block_size]")
    width = ids.shape[1]
    x = model.tok_emb(ids) + model.pos_emb[:, :width, :]
    return model.blocks[0].ln1(x)


def qkv_contract(model: LeechGPT, ids: torch.Tensor) -> dict:
    linear = model.blocks[0].attn.qkv
    batch, tokens = (int(v) for v in ids.shape)
    return {
        "operation": "torch.nn.Linear / rocBLAS GEMM candidate",
        "logical_input_shape_btd": [batch, tokens, int(linear.in_features)],
        "logical_weight_shape_out_in": list(linear.weight.shape),
        "logical_output_shape_btd": [batch, tokens, int(linear.out_features)],
        "flattened_gemm_m_n_k": [batch * tokens, int(linear.out_features),
                                  int(linear.in_features)],
        "dtype": str(linear.weight.dtype).replace("torch.", ""),
        "bias": linear.bias is not None,
        "n_heads": int(model.blocks[0].attn.n_heads),
        "head_dim": int(model.blocks[0].attn.head_dim),
    }


class RoctxRange:
    """Best-effort ROCTX range; absence is reported rather than hidden."""
    def __init__(self, label: str):
        self.label = label.encode()
        self.lib = None
        self.available = False
        self.detail = "ROCTX library unavailable"
        candidates = [
            os.environ.get("ROCTX_LIBRARY", ""),
            "librocprofiler-sdk-roctx.so.1", "librocprofiler-sdk-roctx.so",
            "libroctx64.so", "libroctx64.so.1",
        ]
        for candidate in candidates:
            if not candidate:
                continue
            try:
                lib = ctypes.CDLL(candidate)
                if hasattr(lib, "roctxRangePushA") and hasattr(lib, "roctxRangePop"):
                    self.lib = lib
                    self.available = True
                    self.detail = candidate
                    break
            except OSError:
                continue

    def __enter__(self):
        if self.available:
            # selected-regions is driven by this pair when the SDK exposes it.
            if hasattr(self.lib, "roctxProfilerResume"):
                self.lib.roctxProfilerResume(ctypes.c_uint64(0))
            self.lib.roctxRangePushA(self.label)
            if hasattr(self.lib, "roctxMarkA"):
                self.lib.roctxMarkA(self.label)
        return self

    def __exit__(self, *_exc):
        if self.available:
            self.lib.roctxRangePop()
            if hasattr(self.lib, "roctxProfilerPause"):
                self.lib.roctxProfilerPause(ctypes.c_uint64(0))


def load_teacher(path: Path, config: Path | None) -> LeechGPT:
    state, info = checkpoint_parts(path)
    cfg = infer_config(state, info, config)
    if cfg.moe_experts:
        raise ValueError("QKV attribution requires the dense teacher checkpoint")
    model = LeechGPT(cfg).eval()
    model.load_state_dict(state, strict=True)
    return model


def run(args) -> dict:
    if torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError("ROCm HIP GPU unavailable; refusing CPU fallback")
    rows = np.load(args.token_ids, allow_pickle=False)
    if rows.ndim != 2 or len(rows) < args.batch:
        raise ValueError("token IDs need at least --batch two-dimensional rows")
    ids = torch.as_tensor(rows[:args.batch], dtype=torch.long)
    teacher = load_teacher(Path(args.teacher),
                           None if args.config is None else Path(args.config))
    cpu = teacher.cpu().eval()
    with torch.no_grad():
        cpu_x = qkv_input(cpu, ids)
        cpu_y = cpu.blocks[0].attn.qkv(cpu_x)
    gpu = teacher.cuda().eval()
    ids_gpu = ids.cuda()
    torch.cuda.synchronize()
    with torch.no_grad():
        gpu_x = qkv_input(gpu, ids_gpu)
        marker = RoctxRange("leech:block0:qkv")
        with marker:
            gpu_y = gpu.blocks[0].attn.qkv(gpu_x)
        torch.cuda.synchronize()
        repeats = []
        for _ in range(args.repeats - 1):
            next_y = gpu.blocks[0].attn.qkv(gpu_x)
            torch.cuda.synchronize()
            repeats.append(float((gpu_y - next_y).abs().max().cpu()))
    gpu_y_cpu = gpu_y.detach().float().cpu()
    cpu_y = cpu_y.detach().float().cpu()
    report = {
        "purpose": "QKV kernel attribution, not GPU admission",
        "phase": args.phase,
        "qkv_contract": qkv_contract(cpu, ids),
        "roctx": {"label": "leech:block0:qkv", "available": marker.available,
                  "library": marker.detail},
        "cpu_gpu_qkv_max_abs": float((cpu_y - gpu_y_cpu).abs().max()),
        "gpu_repeat_qkv_max_abs": repeats,
        "cpu_qkv_sha256": sha256_tensor(cpu_y),
        "gpu_qkv_sha256": sha256_tensor(gpu_y_cpu),
        "qkv_input_cpu_sha256": sha256_tensor(cpu_x),
        "qkv_input_gpu_sha256": sha256_tensor(gpu_x),
        "runtime": attest_runtime(),
        "caveat": ("Profiler/ROCTX can change timing or dispatch. Compare this "
                   "receipt with the unprofiled receipt before assigning a kernel."),
    }
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher", required=True)
    p.add_argument("--config")
    p.add_argument("--token-ids", required=True)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--phase", choices=("unprofiled", "profiled"), required=True)
    p.add_argument("--receipt", required=True)
    args = p.parse_args()
    if args.batch < 1 or args.repeats < 2:
        p.error("--batch must be >=1 and --repeats must be >=2")
    result = run(args)
    target = Path(args.receipt)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"receipt": str(target), "gemm": result["qkv_contract"]["flattened_gemm_m_n_k"],
                      "cpu_gpu_max_abs": result["cpu_gpu_qkv_max_abs"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
