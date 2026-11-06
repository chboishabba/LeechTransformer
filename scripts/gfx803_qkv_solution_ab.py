"""Exact Leech QKV rocBLAS solution A/B and stock SGEMM shim probes.

Diagnostic-only. No training, no broad ROCm success claim, no randomized
substitutes. Uses the beta rocblas_gemm_ex_get_solutions / rocblas_gemm_ex
algorithm originally employed by Schaka's 6.4.4 solution scanner:
https://github.com/Schaka/rocm-gfx803/blob/8fdcf7b1305d0cfc6fc8c1fc7b1bfb5f73494e78/rocm6.4.4/tools/rocblas_solution_scan.cpp

The original torch QKV trace used rocblas_sgemm, whereas solution-forcing
uses rocblas_gemm_ex; the marker/kernel identity must be established before
asserting that a forced index is the same execution as the original trace.
Solution IDs are *library-local* and never copied to another ROCm build.
"""
from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from scripts.export_gfx803_qkv_fixture import verify_fixture, sha256_file

# Original ROCm 6.4 rocblas-types.h enumerants (see Schaka's scanner/API).
OP_T = 112
OP_N = 111
F32 = 151
SOLUTION_INDEX = 1
SUCCESS = 0
_EXPECTED_BASELINE_INPUT = "6e33dd589ce2fdc69f953fe1e195016076b883a7948f72a002de05fb49056cb5"


class ABRefused(RuntimeError):
    """An unverified ABI, fixture, runtime or solution cannot be promoted."""


def mapped_library_paths(maps: str, prefix: str) -> list[str]:
    result = set()
    for line in maps.splitlines():
        parts = line.strip().split(maxsplit=5)
        if len(parts) >= 6 and parts[5].startswith("/") and Path(parts[5].split(" (deleted)")[0]).name.startswith(prefix):
            result.add(parts[5].split(" (deleted)")[0])
    return sorted(result)


def bound_rocblas_path(maps: str, allowed_root: Path) -> str:
    paths = mapped_library_paths(maps, "librocblas.so")
    if not paths:
        raise ABRefused("No mapped rocBLAS library; initialise the original torch QKV first")
    normalized = sorted({str(Path(x).resolve()) for x in paths})
    if len(normalized) != 1:
        raise ABRefused("Multiple actual rocBLAS libraries mapped: " + repr(normalized))
    root = Path(allowed_root).resolve()
    if not Path(normalized[0]).is_relative_to(root):
        raise ABRefused("rocBLAS selected outside explicit runtime root: " + normalized[0])
    return normalized[0]


def verify_sample_fixture(folder: Path, expected_input_sha: str | None) -> dict:
    meta = verify_fixture(folder, expected_input_sha=expected_input_sha)
    m, n, k = meta["gemm_logical_m_n_k"]
    roc = meta["rocblas"]
    if (roc["M"], roc["N"], roc["K"], roc["lda"], roc["ldb"], roc["ldc"]) != (n, m, k, k, k, n):
        raise ABRefused("Unexpected QKV GEMM column-major transpose/stride contract")
    return meta


def make_rocblas_ex_arguments(handle: C.c_void_p, alpha: C.c_float,
                              weight_ptr: int, input_ptr: int, beta: C.c_float,
                              out_ptr: int, m: int, n: int, k: int) -> tuple:
    """Column-major: A=weight (K,N) transposed; B=input (K,M) untransposed."""
    return (handle, OP_T, OP_N, n, m, k,
            C.cast(C.pointer(alpha), C.c_void_p),
            C.c_void_p(weight_ptr), F32, k,
            C.c_void_p(input_ptr), F32, k,
            C.cast(C.pointer(beta), C.c_void_p),
            C.c_void_p(out_ptr), F32, n,
            C.c_void_p(out_ptr), F32, n, F32, SOLUTION_INDEX)


def make_signatures(lib) -> None:
    ptr, integer = C.c_void_p, C.c_int32
    lib.rocblas_create_handle.argtypes = [C.POINTER(ptr)]
    lib.rocblas_create_handle.restype = integer
    lib.rocblas_destroy_handle.argtypes = [ptr]
    lib.rocblas_destroy_handle.restype = integer
    lib.rocblas_set_stream.argtypes = [ptr, ptr]
    lib.rocblas_set_stream.restype = integer
    common = [ptr, integer, integer, integer, integer, integer, ptr,
              ptr, integer, integer, ptr, integer, integer, ptr,
              ptr, integer, integer, ptr, integer, integer, integer, integer]
    lib.rocblas_gemm_ex_get_solutions.argtypes = common + [C.c_uint32,
                                                          C.POINTER(integer),
                                                          C.POINTER(integer)]
    lib.rocblas_gemm_ex_get_solutions.restype = integer
    lib.rocblas_gemm_ex.argtypes = common + [integer, C.c_uint32]
    lib.rocblas_gemm_ex.restype = integer


def allclose_diagnostics(output: np.ndarray, reference: np.ndarray,
                         atol: float, rtol: float) -> dict:
    if output.shape != reference.shape:
        raise ValueError("GEMM output shape mismatch")
    residual = np.abs(output.astype(np.float64) - reference.astype(np.float64))
    finite = np.isfinite(output).all()
    threshold = atol + rtol * np.abs(reference.astype(np.float64))
    error_count = int(np.count_nonzero(~(residual <= threshold)))
    return {"max_abs": float(residual.max()),
            "mean_abs": float(residual.mean()),
            "max_relative": float(np.max(residual / (np.abs(reference) + 1e-6))),
            "nonfinite_count": int(np.count_nonzero(~np.isfinite(output))),
            "outside_tolerance": error_count,
            "pass": bool(finite and error_count == 0)}


class RocblasBound:
    def __init__(self, path: str, stream: int):
        self.path = path
        self.lib = C.CDLL(path)
        for required in ("rocblas_create_handle", "rocblas_destroy_handle",
                         "rocblas_set_stream", "rocblas_gemm_ex_get_solutions",
                         "rocblas_gemm_ex"):
            if not hasattr(self.lib, required):
                raise ABRefused("required rocBLAS 6.4 beta API missing: " + required)
        make_signatures(self.lib)
        self.handle = C.c_void_p()
        status = self.lib.rocblas_create_handle(C.byref(self.handle))
        if status != SUCCESS:
            raise ABRefused(f"rocblas_create_handle failed: {status}")
        status = self.lib.rocblas_set_stream(self.handle, C.c_void_p(stream))
        if status != SUCCESS:
            self.lib.rocblas_destroy_handle(self.handle)
            raise ABRefused(f"rocblas_set_stream failed: {status}")
        self.alpha = C.c_float(1.0)
        self.beta = C.c_float(0.0)

    def close(self):
        if self.handle:
            self.lib.rocblas_destroy_handle(self.handle)
            self.handle = None

    def enumerate(self, args: tuple) -> list[int]:
        size = C.c_int32(0)
        status = self.lib.rocblas_gemm_ex_get_solutions(
            *args, C.c_uint32(0), None, C.byref(size))
        if status != SUCCESS or not (0 < size.value <= 100000):
            raise ABRefused(f"cannot enumerate original QKV solutions: status={status} count={size.value}")
        array = (C.c_int32 * size.value)()
        capacity = C.c_int32(size.value)
        status = self.lib.rocblas_gemm_ex_get_solutions(
            *args, C.c_uint32(0), array, C.byref(capacity))
        if status != SUCCESS or not (0 < capacity.value <= size.value):
            raise ABRefused(f"rocblas solution list invalid: status={status} count={capacity.value}")
        result = [int(array[i]) for i in range(capacity.value)]
        if len(set(result)) != len(result):
            raise ABRefused("duplicate solution IDs from one rocBLAS library")
        return result

    def execute(self, args: tuple, solution_id: int) -> int:
        return int(self.lib.rocblas_gemm_ex(*args, C.c_int32(solution_id), C.c_uint32(0)))


def load_f32(folder: Path, meta: dict, name: str):
    entry = meta["tensors"][name]
    return np.fromfile(folder / entry["file"], dtype="<f4").reshape(entry["shape"]).copy()


def setup_gpu(args):
    import torch
    from scripts.profile_gfx803_leech_qkv import RoctxRange
    if torch.version.hip is None or not torch.cuda.is_available():
        raise ABRefused("real HIP GPU unavailable; never fall back to CPU for A/B")
    props = torch.cuda.get_device_properties(0)
    arch = getattr(props, "gcnArchName", None)
    # An actual gfx803 device name is independently present in older gate;
    # reject a known non-gfx803 arch instead of testing another device silently.
    if arch and not str(arch).startswith("gfx803"):
        raise ABRefused("requested gfx803 but physical architecture is " + str(arch))
    meta = verify_sample_fixture(args.fixture, args.expected_input_sha)
    source = load_f32(args.fixture, meta, "input")
    weights = load_f32(args.fixture, meta, "weight")
    ref = load_f32(args.fixture, meta, "reference_f64_cast_f32")
    x = torch.from_numpy(source).to("cuda").contiguous()
    w = torch.from_numpy(weights).to("cuda").contiguous()
    d = torch.zeros((source.shape[0], weights.shape[0]), dtype=torch.float32,
                    device="cuda")
    # Load the actual library through PyTorch's ordinary GEMM path first.
    with torch.no_grad():
        _ = torch.nn.functional.linear(x, w, bias=None)
    torch.cuda.synchronize()
    maps = Path("/proc/self/maps").read_text()
    path = bound_rocblas_path(maps, args.expected_rocm_root)
    # Preload is a separately tested 10.1 condition; do not mix it into 6.4 WGM.
    if args.mode in ("enumerate", "run") and "libgfx803_sgemm_shim.so" in os.environ.get("LD_PRELOAD", ""):
        raise ABRefused("WGM6.4 solution-index arm refuses foreign SGEMM preload")
    api = RocblasBound(path, torch.cuda.current_stream().cuda_stream)
    m, n, k = meta["gemm_logical_m_n_k"]
    arguments = make_rocblas_ex_arguments(api.handle, api.alpha, w.data_ptr(),
                                          x.data_ptr(), api.beta, d.data_ptr(), m, n, k)
    ids = api.enumerate(arguments)
    baseline = {"purpose": "QKV solution attribution, never GPU training admission",
                "library_local_only": True, "api": "rocblas_gemm_ex_get_solutions/rocblas_gemm_ex",
                "original_pytorch_dispatch_api": "rocblas_sgemm",
                "api_difference_is_causal_scope_limit": True,
                "gpu_arch": arch, "device": str(props.name), "torch_hip": str(torch.version.hip),
                "rocblas_realpath": path, "rocblas_sha256": sha256_file(Path(path)),
                "fixture_manifest_sha256": sha256_file(args.fixture / "manifest.json"),
                "teacher_sha256": meta["teacher_sha256"],
                "input_sha256": meta["tensors"]["input"]["sha256"],
                "weight_sha256": meta["tensors"]["weight"]["sha256"],
                "gemm": meta["rocblas"], "solution_ids": ids,
                "threshold": {"atol": args.atol, "rtol": args.rtol},
                "admission": "denied"}
    return torch, RoctxRange, meta, ref, api, arguments, ids, d, baseline


def run_solution(args) -> dict:
    torch, RoctxRange, meta, ref, api, arguments, ids, d, baseline = setup_gpu(args)
    try:
        if args.mode == "enumerate":
            baseline["status"] = "enumerated-without-evaluating-solutions"
            return baseline
        index = args.solution_id
        if index not in ids:
            raise ABRefused("requested solution not enumerated by exact current rocBLAS build: " + str(index))
        if args.repeats < 1:
            raise ABRefused("repeats must be >= 1")
        values, hashes, codes = [], [], []
        try:
            for rep in range(args.repeats):
                d.zero_()
                torch.cuda.synchronize()
                # One marked dispatch per process for unambiguous trace->index.
                if rep == 0:
                    with RoctxRange("leech:qkv:solution:" + str(index)):
                        code = api.execute(arguments, index)
                        torch.cuda.synchronize()
                else:
                    code = api.execute(arguments, index)
                    torch.cuda.synchronize()
                codes.append(code)
                if code != SUCCESS:
                    break
                observed = d.detach().float().cpu().numpy()
                if not np.isfinite(observed).all():
                    values.append({"pass": False, "nonfinite_count": int(
                        np.count_nonzero(~np.isfinite(observed)))})
                else:
                    values.append(allclose_diagnostics(observed, ref,
                                                       args.atol, args.rtol))
                hashes.append(hashlib.sha256(observed.tobytes()).hexdigest())
        except Exception as exc:
            raise ABRefused("solution GPU execution failed; preserve device/reset logs: " + repr(exc)) from exc
        status = ("inapplicable" if codes[-1] != SUCCESS else
                  "correct-at-tested-shape" if all(x["pass"] for x in values)
                  else "incorrect-at-tested-shape")
        baseline.update({"solution_id": index, "repeats_requested": args.repeats,
                         "repeats_completed": len(values), "status": status,
                         "rocblas_status_codes": codes, "repeat_metrics": values,
                         "output_sha256_by_repeat": hashes,
                         "repeat_sha256_identical": len(set(hashes)) == 1 if hashes else None,
                         "max_abs": max((v.get("max_abs", float("inf")) for v in values), default=None),
                         "observed_tensile_kernel": "requires-rocprofv3-trace-join",
                         "admission": "denied"})
        return baseline
    finally:
        api.close()


def shim_probe(args) -> dict:
    import torch
    from scripts.profile_gfx803_leech_qkv import RoctxRange
    if torch.version.hip is None or not torch.cuda.is_available():
        raise ABRefused("Shim arm needs real GPU")
    if args.repeats < 2:
        raise ABRefused("shim arm requires >= 2 repeats")
    shim = Path(args.shim_path).resolve(strict=True)
    if shim.name != "libgfx803_sgemm_shim.so" or not shim.is_relative_to(args.expected_rocm_root.resolve()):
        raise ABRefused("Shim must be the exact selected 10.1 runtime's own interposer")
    preload = [Path(x).resolve() for x in os.environ.get("LD_PRELOAD", "").split(":") if x]
    if preload != [shim]:
        raise ABRefused("Require process-start LD_PRELOAD to be exactly the selected Schaka SGEMM shim")
    if args.shim_mode not in ("on", "off"):
        raise ABRefused("shim mode must be on or off")
    disabled = os.environ.get("GFX803_SGEMM_SHIM_DISABLE", "")
    if disabled != ("0" if args.shim_mode == "on" else "1"):
        raise ABRefused("shim enable/disable env does not match selected A/B arm")
    meta = verify_sample_fixture(args.fixture, args.expected_input_sha)
    x = torch.from_numpy(load_f32(args.fixture, meta, "input")).to("cuda")
    w = torch.from_numpy(load_f32(args.fixture, meta, "weight")).to("cuda")
    bias = (torch.from_numpy(load_f32(args.fixture, meta, "bias")).to("cuda")
            if meta["bias_present"] else None)
    ref = load_f32(args.fixture, meta, "reference_full_torch")
    # Import first, then verify actual device-mapped libraries.
    with torch.no_grad():
        _ = torch.nn.functional.linear(x, w, bias)
    torch.cuda.synchronize()
    maps = Path("/proc/self/maps").read_text()
    rocblas = bound_rocblas_path(maps, args.expected_rocm_root)
    mapped_shim = [Path(v).resolve() for v in mapped_library_paths(maps, shim.name)]
    if mapped_shim != [shim]:
        raise ABRefused("Declared SGEMM interposer not actually mapped into Python process")
    recorded = []
    with torch.no_grad():
        for rep in range(args.repeats):
            if rep == 0:
                with RoctxRange("leech:qkv:sgemm-shim:" + args.shim_mode):
                    y = torch.nn.functional.linear(x, w, bias)
                    torch.cuda.synchronize()
            else:
                y = torch.nn.functional.linear(x, w, bias)
                torch.cuda.synchronize()
            out = y.detach().cpu().numpy()
            record = allclose_diagnostics(out, ref, args.atol, args.rtol)
            record["sha256"] = hashlib.sha256(out.tobytes()).hexdigest()
            recorded.append(record)
    return {"purpose": "coherent same-image SGEMM shim A/B; never GPU admission",
            "mode": "shim", "arm": args.shim_mode,
            "fixture_manifest_sha256": sha256_file(args.fixture / "manifest.json"),
            "teacher_sha256": meta["teacher_sha256"],
            "rocblas_realpath": rocblas, "rocblas_sha256": sha256_file(Path(rocblas)),
            "shim_realpath": str(shim), "shim_sha256": sha256_file(shim),
            "shim_actually_mapped": True, "torch_hip": str(torch.version.hip),
            "required_image_root": str(args.expected_rocm_root.resolve()),
            "threshold": {"atol": args.atol, "rtol": args.rtol},
            "repeat_metrics": recorded,
            "status": ("correct-at-tested-shape" if all(r["pass"] for r in recorded)
                       else "incorrect-at-tested-shape"),
            "output_sha256_by_repeat": [v["sha256"] for v in recorded],
            "admission": "denied", "selected_kernel": "requires-rocprofv3-trace-join"}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("enumerate", "run", "shim"), required=True)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--expected-rocm-root", type=Path, required=True,
                   help="Real runtime library root, never a generic system path")
    p.add_argument("--expected-input-sha", default=_EXPECTED_BASELINE_INPUT)
    p.add_argument("--solution-id", type=int)
    p.add_argument("--repeats", type=int, default=100)
    p.add_argument("--shim-mode", choices=("on", "off"))
    p.add_argument("--shim-path", type=Path)
    p.add_argument("--atol", type=float, default=0.001)
    p.add_argument("--rtol", type=float, default=0.0001)
    p.add_argument("--receipt", type=Path, required=True)
    args = p.parse_args()
    if args.atol < 0 or args.rtol < 0 or args.repeats < 1:
        p.error("tolerances must be nonnegative and repeats >=1")
    if args.mode == "run" and args.solution_id is None:
        p.error("--mode run needs --solution-id")
    if args.mode == "shim" and (args.shim_path is None or args.shim_mode is None):
        p.error("--mode shim needs both --shim-path and --shim-mode")
    try:
        result = shim_probe(args) if args.mode == "shim" else run_solution(args)
    except (OSError, ValueError, ABRefused, RuntimeError) as exc:
        p.exit(4, "QKV AB REFUSED: " + str(exc) + "\n")
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"receipt": str(args.receipt),
                      "status": result["status"],
                      "admission": "denied"}, sort_keys=True), flush=True)
    # A completed diagnostic, even if numerically broken, is not an admission.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
