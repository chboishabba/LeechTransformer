"""Optional real-gfx803 WGM1 QKV dispatch inside a normal Leech PyTorch forward.

Only the exact original QKV weight+shape uses an explicitly selected,
profiled rocblas_gemm_ex solution whose retained 100-repeat receipt proves
scoped numerical correctness. All other FP32 nn.Linear calls remain on
differentiable CPU GEMM (gfx803_linear_repair). Backward for forced QKV is
also CPU GEMM. This is an experimental hybrid *operator bridge* and not an
in-place repair of torch's original rocblas_sgemm dispatch.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path

_calls = 0
_first_input_sha = None


def validate_join(join: dict, *, teacher_sha: str, arch: str, hip: str) -> None:
    trace=join.get("trace") or {}
    kernels=trace.get("unique_kernel_names")
    if not (join.get("schema_version") == 2 and
            join.get("attribution_complete") is True and
            join.get("no_training_admission") is True and
            join.get("profile_observer_effect_absent_for_first_output") is True and
            isinstance(join.get("unprofiled_repeats_completed"), int) and
            join["unprofiled_repeats_completed"] >= 100 and
            join.get("unprofiled_numerics") == "correct-at-tested-shape" and
            join.get("profiler_numerics") == "correct-at-tested-shape" and
            trace.get("status") == "identified" and
            trace.get("wgm") == 1 and
            isinstance(kernels, list) and len(kernels) == 1 and
            "WGM1" in kernels[0] and
            isinstance(join.get("solution_id"), int) and
            join.get("teacher_sha256") == teacher_sha and
            join.get("gpu_arch") == arch and
            join.get("torch_hip") == hip):
        raise RuntimeError("WGM1 join is not a complete original-teacher 100-repeat kernel-attributed correct result")
    expected_gemm=join.get("gemm") or {}
    if (expected_gemm.get("transA_transB") != ["T","N"] or
        (expected_gemm.get("M"),expected_gemm.get("N"),expected_gemm.get("K")) !=
        (1152,128,384)):
        raise RuntimeError("WGM1 join does not describe the original T,N,1152,128,384 QKV")
    for key in ("rocblas_sha256","weight_sha256","input_sha256"):
        v=join.get(key)
        if not isinstance(v,str) or len(v)!=64:
            raise RuntimeError("incomplete WGM1 source SHA: "+key)


def build_bridge() -> object:
    """Return a scoped PyTorch-compatible linear bridge; no implicit defaults."""
    import torch
    from scripts.gfx803_qkv_solution_ab import (
        ABRefused, RocblasBound, bound_rocblas_path,
        make_rocblas_ex_arguments, sha256_file, SUCCESS,
    )

    join_path=os.environ.get("GFX803_WGM1_QKV_JOIN")
    teacher_sha=os.environ.get("GFX803_EXPECT_TEACHER_SHA256")
    root_text=os.environ.get("GFX803_EXPECT_ROCBLAS_ROOT")
    if not join_path or not teacher_sha or not root_text:
        raise RuntimeError("WGM1 requires exact joined receipt, teacher hash and current rocBLAS root")
    join=json.loads(Path(join_path).read_text())
    kernels=join.get("trace",{}).get("unique_kernel_names",[])
    props=torch.cuda.get_device_properties(0)
    arch=str(getattr(props,"gcnArchName",""))
    validate_join(join,teacher_sha=teacher_sha,arch=arch,hip=str(torch.version.hip))
    expected=Path(join["rocblas_realpath"]).resolve(strict=True)
    root=Path(root_text).resolve(strict=True)
    if not expected.is_relative_to(root):
        raise RuntimeError("joined WGM1 rocBLAS escapes selected extracted root")
    if sha256_file(expected)!=join["rocblas_sha256"]:
        raise RuntimeError("actual rocBLAS SHA differs from WGM1 100-repeat baseline")
    if os.environ.get("LD_PRELOAD"):
        raise RuntimeError("WGM1 bridge refuses SGEMM or foreign LD_PRELOAD")
    # A direct load is solely to attest actual mapping; all GEMM operations
    # remain in the current torch HIP stream, not in a separate GPU context.
    ctypes.CDLL(str(expected))
    mapped=bound_rocblas_path(Path("/proc/self/maps").read_text(), root)
    if mapped!=str(expected):
        raise RuntimeError("WGM1 rocBLAS path differs from actually mapped library")

    dims=(128,1152,384)
    cache: dict[tuple[int,int], bool]={}

    def weight_matches(weight) -> bool:
        key=(int(weight.data_ptr()),int(weight._version))
        if key not in cache:
            data=weight.detach().cpu().contiguous().numpy().astype("<f4",copy=False)
            cache[key]=(hashlib.sha256(data.tobytes()).hexdigest()==
                        join["weight_sha256"])
        return cache[key]

    class _Forced(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, weight):
            global _calls, _first_input_sha
            m,n,k=dims
            x=input.contiguous().view(m,k)
            w=weight.contiguous()
            if _first_input_sha is None:
                raw=x.detach().cpu().contiguous().numpy().astype("<f4",copy=False)
                _first_input_sha=hashlib.sha256(raw.tobytes()).hexdigest()
            out=torch.empty((m,n),device=input.device,dtype=torch.float32)
            api=RocblasBound(str(expected),int(torch.cuda.current_stream().cuda_stream))
            try:
                args=make_rocblas_ex_arguments(
                    api.handle,api.alpha,int(w.data_ptr()),int(x.data_ptr()),
                    api.beta,int(out.data_ptr()),m,n,k)
                ids=api.enumerate(args)
                if join["solution_id"] not in ids:
                    raise ABRefused("WGM1 solution ID not enumerated by same bound rocBLAS")
                status=api.execute(args,join["solution_id"])
                if status!=SUCCESS:
                    raise ABRefused("forced rocblas_gemm_ex failed: "+str(status))
                torch.cuda.synchronize()
            finally:
                api.close()
            ctx.save_for_backward(input,weight)
            _calls+=1
            return out.view(*input.shape[:-1],n)

        @staticmethod
        def backward(ctx, grad_output):
            input,weight=ctx.saved_tensors
            g=grad_output.contiguous().reshape(-1,weight.shape[0]).cpu()
            x=input.contiguous().reshape(-1,weight.shape[1]).cpu()
            w=weight.cpu()
            dx=(g @ w).to(input.device).reshape(input.shape)
            dw=(g.T @ x).to(weight.device)
            return dx,dw

    class Bridge:
        def matches(self,input,weight) -> bool:
            m,n,k=dims
            return (input.is_cuda and input.dtype==torch.float32 and
                    weight.dtype==torch.float32 and
                    input.shape[-1]==k and
                    input.numel()==m*k and
                    tuple(weight.shape)==(n,k) and
                    weight_matches(weight))

        def linear(self,input,weight,bias):
            out=_Forced.apply(input,weight)
            return out if bias is None else out+bias

        def receipt(self) -> dict:
            return {
                "solution_id":join["solution_id"],
                "source_join_sha256":hashlib.sha256(Path(join_path).read_bytes()).hexdigest(),
                "original_input_sha256":join["input_sha256"],
                "first_model_input_sha256":_first_input_sha,
                "model_input_bytes_equal_original_fixture":(
                    _first_input_sha==join["input_sha256"] if _first_input_sha else None),
                "rocblas_sha256":join["rocblas_sha256"],
                "reported_kernel":kernels[0],
                "forced_qkv_gpu_forward_calls":_calls,
                "gpu_backward":False, "non_qkv_gpu_linear":False,
                "torch_default_rocblas_sgemm_fixed":False,
                "admission":"denied",
            }

    return Bridge()
