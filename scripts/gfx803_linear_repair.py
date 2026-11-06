"""gfx803 ROCm 6.4 diagnostic: replace HIP FP32 nn.Linear GEMM with CPU GEMM.

This is a *hybrid, deliberately slow* correctness-isolation experiment, not
the Schaka WGM1 GPU patch and not a GPU training admission. Forward and backward
remain differentiable through the GPU<->CPU copies. No fake solution selection:
a plain PyTorch linear on gfx803 is known to dispatch faulty rocBLAS kernels.
"""
from __future__ import annotations

import os
from typing import Any

_installed = False
_calls = 0
_original = None


def install_cpu_fp32_linear_repair() -> None:
    """Patch only FP32 nn.Linear on HIP gfx803; fail closed on other GPU dtypes."""
    global _installed, _original
    if _installed:
        return
    if os.environ.get("GFX803_LINEAR_REPAIR") != "cpu-fp32":
        raise RuntimeError("Explicit GFX803_LINEAR_REPAIR=cpu-fp32 required")
    import torch
    import torch.nn.functional as F

    if torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError("CPU linear repair requires a real HIP GPU")
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    arch = str(getattr(props, "gcnArchName", ""))
    if not arch.startswith("gfx803"):
        raise RuntimeError("CPU linear repair is scoped to observed gfx803, got " + arch)
    original = F.linear

    def repaired(input: Any, weight: Any, bias: Any = None) -> Any:
        global _calls
        if not input.is_cuda:
            return original(input, weight, bias)
        if input.dtype != torch.float32 or weight.dtype != torch.float32 or (
            bias is not None and bias.dtype != torch.float32
        ):
            raise RuntimeError("gfx803 FP32 linear diagnostic refuses non-FP32 GPU tensor")
        if weight.device != input.device or (
            bias is not None and bias.device != input.device
        ):
            raise RuntimeError("gfx803 repair refuses mixed-device linear operands")
        # Autograd follows differentiable device transfers; gradients reach
        # original GPU parameters. No .detach(), .data or no_grad shortcuts.
        _calls += 1
        return original(
            input.to(device="cpu"), weight.to(device="cpu"),
            None if bias is None else bias.to(device="cpu")
        ).to(device=input.device)

    _original = original
    F.linear = repaired
    _installed = True


def repair_receipt() -> dict:
    return {
        "kind": "cpu-fp32-linear-diagnostic",
        "installed": _installed,
        "gpu_linear_calls_repaired": _calls,
        "gpu_gemm_forced_wgm1": False,
        "full_gpu_execution": False,
        "training_admission": "denied",
        "note": "CPU GEMM substitution identifies residual GPU failures; it cannot certify a WGM1 repair or GPU speed.",
    }
