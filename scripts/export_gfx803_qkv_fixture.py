"""Freeze original Leech block-0 QKV operands for same-library rocBLAS A/B.

Produces raw little-endian float32 tensors and CPU float32/float64 GEMM
references. No random substitutes. Original owner: LeechTransformer gfx803
first-party reproduction; the solution enumeration methodology belongs to
Schaka (rocm-gfx803, rocm6.4.4/tools/rocblas_solution_scan.cpp).

CPU-only export. It NEVER runs GPU training or asserts a kernel fix.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import shutil
import os
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_f32(path: Path, array: np.ndarray) -> dict:
    a = np.ascontiguousarray(array, dtype="<f4")
    if not np.isfinite(a).all():
        raise ValueError("QKV fixture must contain finite FP32 values")
    data = a.tobytes(order="C")
    path.write_bytes(data)
    return {"file": path.name, "shape": list(a.shape),
            "dtype": "float32-little-endian", "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def verify_fixture(folder: Path, *, expected_input_sha: str | None = None) -> dict:
    meta = json.loads((folder / "manifest.json").read_text())
    if meta.get("schema_version") != 1 or meta.get("kind") != "leech-qkv-original-fp32":
        raise ValueError("not an original Leech QKV fixture")
    if meta.get("rocblas", {}).get("transA_transB") != ["T", "N"]:
        raise ValueError("expected column-major T,N QKV configuration")
    tensors = meta.get("tensors", {})
    required = {"input", "weight", "reference_f32", "reference_f64_cast_f32"}
    if not required.issubset(tensors):
        raise ValueError("missing required model tensors")
    for key, rec in tensors.items():
        name = rec.get("file")
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("unsafe tensor filename")
        path = folder / name
        expected_len = 4
        for v in rec["shape"]:
            if not isinstance(v, int) or v < 1:
                raise ValueError("invalid tensor dimension")
            expected_len *= v
        if path.stat().st_size != expected_len or rec.get("bytes") != expected_len:
            raise ValueError("tensor byte length mismatch: " + key)
        if sha256_file(path) != rec["sha256"]:
            raise ValueError("tensor SHA256 mismatch: " + key)
        if not np.isfinite(np.fromfile(path, dtype="<f4")).all():
            raise ValueError("nonfinite tensor: " + key)
    m, n, k = meta["gemm_logical_m_n_k"]
    if tensors["input"]["shape"] != [m, k] or tensors["weight"]["shape"] != [n, k]:
        raise ValueError("QKV dimensions do not match fixture")
    if tensors["reference_f32"]["shape"] != [m, n]:
        raise ValueError("CPU reference dimensions do not match fixture")
    if expected_input_sha and tensors["input"]["sha256"] != expected_input_sha:
        raise ValueError("wrong original input SHA256 (different model or token slice)")
    return meta


def export(args) -> dict:
    import torch
    from scripts.profile_gfx803_leech_qkv import load_teacher, qkv_input

    destination = Path(args.out_dir)
    if destination.exists():
        raise ValueError("refuse to overwrite original QKV fixture; use a new directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix=".qkv-fixture-", dir=destination.parent))
    rows = np.load(args.token_ids, allow_pickle=False)
    if rows.ndim != 2 or rows.shape[0] < args.batch or rows.shape[1] < args.tokens:
        raise ValueError("token source must contain exact requested batch and sequence")
    ids = torch.as_tensor(rows[:args.batch, :args.tokens], dtype=torch.long)
    teacher = load_teacher(Path(args.teacher),
                           Path(args.config) if args.config else None).cpu().eval()
    with torch.no_grad():
        x = qkv_input(teacher, ids).detach().contiguous()
        linear = teacher.blocks[0].attn.qkv
        weight = linear.weight.detach().cpu().contiguous()
        bias = None if linear.bias is None else linear.bias.detach().cpu().contiguous()
        if x.dtype != torch.float32 or weight.dtype != torch.float32:
            raise ValueError("this original failure is FP32-only")
        M, K = int(x.numel() // x.shape[-1]), int(x.shape[-1])
        N = int(weight.shape[0])
        x2 = x.reshape(M, K)
        # No bias: solution enumeration executes exactly alpha*A*B+beta*C.
        ref32 = torch.nn.functional.linear(x2, weight, bias=None).cpu().numpy()
        ref64 = (x2.double() @ weight.double().T).float().cpu().numpy()
        full_reference = torch.nn.functional.linear(x2, weight, bias=bias).cpu().numpy()
    contract = {"schema_version": 1, "kind": "leech-qkv-original-fp32",
                "source": "chboishabba/LeechTransformer original dense checkpoint",
                "attribution": {
                    "model_and_physical_controls": "chboishabba / LeechTransformer gfx803 original work",
                    "solution_API_method": "Schaka/rocm-gfx803 rocm6.4.4/tools/rocblas_solution_scan.cpp",
                    "original_rocblas_scan": "https://github.com/Schaka/rocm-gfx803/blob/8fdcf7b1305d0cfc6fc8c1fc7b1bfb5f73494e78/rocm6.4.4/tools/rocblas_solution_scan.cpp"},
                "teacher_sha256": sha256_file(Path(args.teacher)),
                "tokens_sha256": sha256_file(Path(args.token_ids)),
                "config_sha256": sha256_file(Path(args.config)) if args.config else None,
                "token_batch_sha256": hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
                "batch": args.batch, "tokens": args.tokens, "gemm_logical_m_n_k": [M, N, K],
                "rocblas": {"transA_transB": ["T", "N"], "M": N, "N": M,
                            "K": K, "lda": K, "ldb": K, "ldc": N,
                            "alpha": 1.0, "beta": 0.0,
                            "layout": "QKV weight (N,K) row-major = rocBLAS A(K,N) T; input (M,K) row-major = B(K,M) N"},
                "bias_present": bias is not None, "tensors": {}}
    arrays = {"input": x2.numpy(), "weight": weight.numpy(),
              "reference_f32": ref32, "reference_f64_cast_f32": ref64,
              "reference_full_torch": full_reference}
    if bias is not None:
        arrays["bias"] = bias.numpy()
    for name, arr in arrays.items():
        contract["tensors"][name] = write_f32(folder / (name + ".f32"), arr)
    (folder / "manifest.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    try:
        verify_fixture(folder, expected_input_sha=args.expect_input_sha)
        if destination.exists():
            raise ValueError("QKV fixture destination appeared during export")
        os.replace(folder, destination)
    finally:
        if folder.exists():
            shutil.rmtree(folder)
    return contract


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--config", type=Path)
    p.add_argument("--token-ids", type=Path, required=True)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--expect-input-sha", default=None,
                   help="Fail closed if not the same input as the reported physical QKV run")
    args = p.parse_args()
    if args.batch < 1 or args.tokens < 1:
        p.error("--batch and --tokens must be positive")
    try:
        out = export(args)
    except (OSError, RuntimeError, ValueError) as exc:
        p.exit(3, "QKV EXPORT NOT VERIFIED: " + str(exc) + "\n")
    print(json.dumps({"out": str(args.out_dir), "shape": out["gemm_logical_m_n_k"],
                      "input_sha256": out["tensors"]["input"]["sha256"],
                      "teacher_sha256": out["teacher_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
