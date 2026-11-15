"""Read-only provenance for the *effective* legacy gfx803 numerical runtime.

The old 2026-11-04 physical receipt has no individual MIOpen fields.
A source wrapper's defaults are not proof the current child process
received them: explicit user overrides and loaded dynamic libraries matter.
This module never changes environment/driver state or admits training.
"""
from __future__ import annotations

import os
from pathlib import Path

SAFE_DEFAULTS = {
    "HSA_OVERRIDE_GFX_VERSION": "8.0.3",
    "ROC_ENABLE_PRE_VEGA": "1",
    "TORCH_BLAS_PREFER_HIPBLASLT": "0",
    "MIOPEN_DEBUG_CONV_WINOGRAD": "0",
    "MIOPEN_DEBUG_CONV_FFT": "0",
    "MIOPEN_DEBUG_CONV_DIRECT": "1",
    "MIOPEN_DEBUG_CONV_GEMM": "0",
    "MIOPEN_DEBUG_CONV_DET": "1",
    "MIOPEN_DEBUG_DISABLE_FIND_DB": "1",
    "MIOPEN_FIND_ENFORCE": "3",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "HIP_LAUNCH_BLOCKING": "1",
}

LIBRARY_PREFIXES = (
    "librocblas", "libMIOpen", "libamdhip64", "libhsa-runtime64",
    "libhiprtc", "libamd_comgr", "libdrm", "libLLVM",
    "libtorch", "libc10",
)

def mapped_gpu_libraries(proc_maps: str) -> dict[str, list[str]]:
    """Only paths from memory maps, no inferred or host-default libraries."""
    result: dict[str, set[str]] = {}
    for line in proc_maps.splitlines():
        columns = line.strip().split(maxsplit=5)
        if len(columns) < 6 or not columns[5].startswith("/"):
            continue
        path = columns[5].removesuffix(" (deleted)")
        name = Path(path).name
        if any(name.startswith(prefix) for prefix in LIBRARY_PREFIXES):
            result.setdefault(name, set()).add(path)
    return {k: sorted(v) for k, v in sorted(result.items())}

def attest_runtime(*, environ=None, proc_maps=None) -> dict:
    """Report config matches, not safe/unsafe global inference."""
    env = os.environ if environ is None else environ
    values = {key: env.get(key) for key in SAFE_DEFAULTS}
    mismatch = {key: {"expected": wanted, "actual": values[key]}
                for key, wanted in SAFE_DEFAULTS.items()
                if values[key] != wanted}
    if proc_maps is None:
        try:
            proc_maps = Path("/proc/self/maps").read_text()
        except OSError:
            proc_maps = ""
    libraries = mapped_gpu_libraries(proc_maps)
    all_paths = [p for paths in libraries.values() for p in paths]
    return {
        "scope": "observed-child-process-numerical-profile-not-a-correctness-claim",
        "selected_flags": values,
        "expected_flags": dict(SAFE_DEFAULTS),
        "deviations_from_wrapper_defaults": mismatch,
        "matches_legacy_wrapper_defaults": not mismatch,
        "loaded_library_paths": libraries,
        "loaded_library_paths_observed": bool(libraries),
        "nix_store_gpu_library_paths": [
            p for p in all_paths if "/nix/store/" in p and
            any(Path(p).name.startswith(x) for x in
                ("librocblas", "libMIOpen", "libamdhip64", "libhsa-runtime64"))
        ],
        "caveat": "MIOpen convolution direct-only flags do not govern rocBLAS "
                  "GEMM / attention. Matching flags does not prove parity. "
                  "A library map reflects the instant of capture, not "
                  "libraries not loaded yet.",
    }
