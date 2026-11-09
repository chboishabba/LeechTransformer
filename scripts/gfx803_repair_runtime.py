"""Runtime map guard for Schaka's original ROCm 10.1 SGEMM interposer.

A successful LD_PRELOAD launch without the selected library mapped is not
evidence of patched torch inference. We do not claim shim interception solely
from a mapped .so: original QKV ON/OFF profiler receipts are still required.
"""
from __future__ import annotations

import os
from pathlib import Path

ROCM_NAMES = (
    "librocblas.so", "libamdhip64.so", "libhsa-runtime64.so",
    "libhipblas.so", "libhipblaslt.so", "libMIOpen.so",
    "librocsolver.so", "librocfft.so", "libhiprtc.so",
    "libamd_comgr.so",
)


def mapped_paths(proc_maps: str) -> set[str]:
    paths = set()
    for line in proc_maps.splitlines():
        cols = line.strip().split(maxsplit=5)
        if len(cols) >= 6 and cols[5].startswith("/"):
            paths.add(cols[5].removesuffix(" (deleted)"))
    return paths


def verify_loaded_sgemm_shim(shim: str, *, proc_maps: str | None = None,
                             env: dict | None = None, require_rocblas: bool = True) -> dict:
    values = os.environ if env is None else env
    expected = Path(shim).resolve(strict=True)
    root_value = values.get("GFX803_EXPECT_ROCM_ROOT", "")
    if not root_value:
        raise RuntimeError("GFX803_EXPECT_ROCM_ROOT must name the same Nix runtime")
    root = Path(root_value).resolve(strict=True)
    if not str(root).startswith("/nix/store/"):
        raise RuntimeError("selected Schaka ROCm root is not in /nix/store")
    if not expected.is_relative_to(root):
        raise RuntimeError("the SGEMM shim escapes the selected ROCm root")
    if expected.name != "libgfx803_sgemm_shim.so":
        raise RuntimeError("wrong SGEMM shim filename")
    if values.get("GFX803_SGEMM_SHIM_DISABLE") not in (None, "", "0"):
        raise RuntimeError("SGEMM shim explicitly disabled")
    rocblas_preload=root/"lib/librocblas.so.5"
    if not rocblas_preload.is_file() or not rocblas_preload.resolve(strict=True).is_relative_to(root):
        raise RuntimeError("selected rocBLAS preload is missing or escapes selected Nix store root")
    if values.get("LD_PRELOAD") != f"{rocblas_preload}:{expected}":
        raise RuntimeError("LD_PRELOAD must select exact same-root rocBLAS and Schaka shim, in tested order")
    maps = Path("/proc/self/maps").read_text() if proc_maps is None else proc_maps
    loaded = {str(Path(p).resolve()) for p in mapped_paths(maps)}
    if str(expected) not in loaded:
        raise RuntimeError("requested SGEMM shim is not actually mapped")
    rocm = sorted(p for p in loaded if Path(p).name.startswith(ROCM_NAMES))
    if not any(Path(p).name.startswith("libamdhip64.so") for p in rocm):
        raise RuntimeError("HIP runtime not mapped")
    if require_rocblas and not any(Path(p).name.startswith("librocblas.so") for p in rocm):
        raise RuntimeError("rocBLAS not mapped")
    foreign = [p for p in rocm if not Path(p).is_relative_to(root)]
    if foreign:
        raise RuntimeError("foreign/mixed ROCm library maps: "+repr(foreign))
    return {
        "shim_mapped": True, "shim_enabled": True,
        "shim_sha256": __import__("hashlib").sha256(expected.read_bytes()).hexdigest(),
        "shim_path": str(expected), "selected_rocm_root": str(root),
        "rocblas_preload": str(rocblas_preload),
        "mapped_rocm_libraries": rocm,
        "interception_confirmed_by_profiler": False,
        "full_training_admission": False,
    }
