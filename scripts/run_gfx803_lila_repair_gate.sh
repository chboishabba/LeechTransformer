#!/usr/bin/env bash
# Run real original LILA checkpoint-bound dense/C2 numerical gates on gfx803.
# 6.4: explicit slow CPU-GEMM substitution, not a GPU kernel fix.
# 10.1: Schaka's genuine SGEMM interposer in same-root runtime.
# Never launches training or silently admits the hybrid 6.4 path.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
LANE='' OUT='' PYTHON_BIN='' ROOT='' SHIM='' WGM_JOIN='' ROCBLAS_ROOT='' TORCH_OVERLAY='' REPEATS=3 STABILITY_STEPS=1 DENSE_ONLY=0 C2_FORWARD_ONLY=0 C2_BACKWARD_ONLY=0 PRELOAD_MODE=rocblas-plus-shim GDB_CAPTURE=0 ROUTER_SELECTOR=topk
GATE_ARGS=()
usage() {
  cat >&2 <<'EOF'
Usage:
  bash scripts/run_gfx803_lila_repair_gate.sh --lane 6.4-cpu-linear \
    --receipt artifacts/repair64.json -- --teacher data/best_model.pt \
    --student /path/C2/student.pt --config /path/config.json --train-ids /path/train_ids.npy
  bash scripts/run_gfx803_lila_repair_gate.sh --lane 6.4-wgm1-qkv \
    --wgm-join /path/to/successful_WGM1/trace_join.json \
    --rocblas-root /path/to/gfx803_compat_graph/lib-compat \
    --receipt artifacts/repair64_wgm1.json -- --teacher ... --student ... --train-ids ...
  env -u LD_PRELOAD bash scripts/run_gfx803_lila_repair_gate.sh --lane 10.1-schaka \
    --python /nix/store/.../bin/python --rocm-root /nix/store/...-rocm \
    --shim /nix/store/...-rocm/lib/libgfx803_sgemm_shim.so \
    [--torch-overlay /path/to/patched/torch/overlay] \
    --receipt artifacts/repair101.json -- --teacher ... --student ... --train-ids ...
All evidence is an original-model FP32 CPU/GPU forward/backward/optimizer
comparison. Use a fresh receipt path. Pass --repeats N (>=2) for repeated
unmodified forwards and --stability-steps N (>=1) for matched C2 AdamW updates. No automatic training, no mixed ROCm environment.
EOF
}
while (($#)); do
  case "$1" in
    --lane) LANE="${2:?}"; shift 2;;
    --wgm-join) WGM_JOIN="${2:?}"; shift 2;;
    --rocblas-root) ROCBLAS_ROOT="${2:?}"; shift 2;;
    --python) PYTHON_BIN="${2:?}"; shift 2;;
    --torch-overlay) TORCH_OVERLAY="${2:?}"; shift 2;;
    --rocm-root) ROOT="${2:?}"; shift 2;;
    --shim) SHIM="${2:?}"; shift 2;;
    --receipt) OUT="${2:?}"; shift 2;;
    --repeats) REPEATS="${2:?}"; shift 2;;
    --stability-steps) STABILITY_STEPS="${2:?}"; shift 2;;
    --dense-only) DENSE_ONLY=1; shift;;
    --c2-forward-only) C2_FORWARD_ONLY=1; shift;;
    --c2-backward-only) C2_BACKWARD_ONLY=1; shift;;
    --gdb) GDB_CAPTURE=1; shift;;
    --preload-mode) PRELOAD_MODE="${2:?}"; shift 2;;
    --router-selector) ROUTER_SELECTOR="${2:?}"; shift 2;;
    --) shift; GATE_ARGS=("$@"); break;;
    --help|-h) usage; exit 0;;
    *) echo "Unknown option: $1" >&2; usage; exit 2;;
  esac
done
[[ "$LANE" == 6.4-cpu-linear || "$LANE" == 6.4-wgm1-qkv || "$LANE" == 10.1-schaka ]] || { usage; exit 2; }
[[ -n "$OUT" && "$REPEATS" =~ ^[0-9]+$ && "$REPEATS" -ge 2 && "$STABILITY_STEPS" =~ ^[0-9]+$ && "$STABILITY_STEPS" -ge 1 ]] || { usage; exit 2; }
[[ "$PRELOAD_MODE" == rocblas-plus-shim || "$PRELOAD_MODE" == shim-only ]] || { echo "Invalid --preload-mode" >&2; exit 2; }
[[ "$ROUTER_SELECTOR" == topk || "$ROUTER_SELECTOR" == sort ]] || { echo "Invalid --router-selector" >&2; exit 2; }
((DENSE_ONLY + C2_FORWARD_ONLY + C2_BACKWARD_ONLY <= 1)) || { echo "Choose only one bounded stage mode" >&2; exit 2; }
for key in --teacher --student --train-ids; do
  found=0
  for item in "${GATE_ARGS[@]}"; do [[ "$item" != "$key" ]] || found=1; done
  [[ "$found" == 1 ]] || { echo "Missing mandatory $key" >&2; exit 2; }
done
for item in "${GATE_ARGS[@]}"; do
  case "$item" in --receipt|--repeats|--stability-steps|--dense-only|--stop-after-c2-forward|--stop-after-c2-backward|--progress-file) echo "Don't duplicate gate options" >&2; exit 2;; esac
done
OUT="$(realpath -m "$OUT")"
[[ ! -e "$OUT" ]] || { echo "Refuse to overwrite old repair evidence: $OUT" >&2; exit 2; }
mkdir -p "$(dirname "$OUT")"
export HIP_LAUNCH_BLOCKING=1 PYTHONNOUSERSITE=1
export HSA_OVERRIDE_GFX_VERSION=8.0.3 ROC_ENABLE_PRE_VEGA=1
export TORCH_BLAS_PREFER_HIPBLASLT=0
case "$LANE" in
  6.4-cpu-linear|6.4-wgm1-qkv)
    [[ -z "$ROOT" && -z "$SHIM" && -z "$PYTHON_BIN" && -z "${LD_PRELOAD:-}" ]] || {
      echo "6.4 hybrid lane rejects 10.1 arguments or inherited preload" >&2; exit 2; }
    export GFX803_LINEAR_REPAIR=cpu-fp32
    if [[ "$LANE" == 6.4-wgm1-qkv ]]; then
      [[ -n "$WGM_JOIN" && -f "$WGM_JOIN" && -n "$ROCBLAS_ROOT" && -d "$ROCBLAS_ROOT" ]] || {
        echo "WGM1 QKV requires a completed 100-repeat trace_join.json and original rocBLAS root" >&2; exit 2; }
      export GFX803_WGM1_QKV_JOIN="$(realpath "$WGM_JOIN")"
      export GFX803_EXPECT_ROCBLAS_ROOT="$(realpath "$ROCBLAS_ROOT")"
    else
      [[ -z "$WGM_JOIN" && -z "$ROCBLAS_ROOT" ]] || {
        echo "CPU-only 6.4 lane refuses WGM1 args" >&2; exit 2; }
      unset GFX803_WGM1_QKV_JOIN GFX803_EXPECT_ROCBLAS_ROOT
    fi
    unset GFX803_EXPECT_SHIM GFX803_EXPECT_ROCM_ROOT GFX803_EXPECT_HIP_PREFIX GFX803_SGEMM_SHIM_DISABLE
    exec bash scripts/run_gfx803_moe_nix.sh \
      "${GATE_ARGS[@]}" --repeats "$REPEATS" --stability-steps "$STABILITY_STEPS" --receipt "$OUT"
    ;;
  10.1-schaka)
    [[ -z "${LD_PRELOAD:-}" && -z "${GFX803_LINEAR_REPAIR:-}" && -z "$WGM_JOIN" && -z "$ROCBLAS_ROOT" && -z "${GFX803_WGM1_QKV_JOIN:-}" ]] || {
      echo "Refuse inherited LD_PRELOAD/6.4 hybrid mode" >&2; exit 2; }
    [[ -n "$PYTHON_BIN" && -n "$ROOT" && -n "$SHIM" ]] || { usage; exit 2; }
    # Preserve the imported venv launcher: realpath may strip venv site-packages.
    ROOT="$(realpath "$ROOT")"
    SHIM="$(realpath "$SHIM")"
    [[ -x "$PYTHON_BIN" && -f "$SHIM" && "$ROOT" == /nix/store/* &&
       "$SHIM" == "$ROOT/"* && "$SHIM" == */libgfx803_sgemm_shim.so &&
       "$PYTHON_BIN" != *docker-venv* && "$ROOT" != *lib-compat* ]] || {
      echo "Refuse unpinned, mixed or missing Schaka 10.1 runtime" >&2; exit 2; }
    [[ "${LD_LIBRARY_PATH:-}" != *lib-compat* ]] || {
      echo "6.4 lib-compat detected in 10.1 environment" >&2; exit 2; }
    if [[ -n "$TORCH_OVERLAY" ]]; then
      TORCH_OVERLAY="$(realpath "$TORCH_OVERLAY")"
      [[ -f "$TORCH_OVERLAY/torch/__init__.py" ]] || {
        echo "Missing torch package in --torch-overlay: $TORCH_OVERLAY" >&2; exit 2; }
    fi
    ROCBLAS_PRELOAD="$ROOT/lib/librocblas.so.5"
    [[ -f "$ROCBLAS_PRELOAD" ]] || { echo "Missing selected ROCm rocBLAS preload" >&2; exit 2; }
    export GFX803_EXPECT_SHIM="$SHIM" GFX803_EXPECT_ROCM_ROOT="$ROOT"
    export GFX803_EXPECT_HIP_PREFIX=7.16.26385
    if [[ "$PRELOAD_MODE" == shim-only ]]; then CHILD_PRELOAD="$SHIM"; else CHILD_PRELOAD="$ROCBLAS_PRELOAD:$SHIM"; fi
    export GFX803_SGEMM_SHIM_DISABLE=0 GFX803_SGEMM_PRELOAD_MODE="$PRELOAD_MODE" LD_PRELOAD="$CHILD_PRELOAD"
    export LD_LIBRARY_PATH="$ROOT/lib:$ROOT/core-10.1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
    # Process-level attestation and original checkpoint test happen in the
    # same process; external source-only checks never substitute for hardware.
    DENSE_ARGS=()
    ((DENSE_ONLY)) && DENSE_ARGS=(--dense-only)
    C2_ARGS=()
    ((C2_FORWARD_ONLY)) && C2_ARGS=(--stop-after-c2-forward)
    ((C2_BACKWARD_ONLY)) && C2_ARGS=(--stop-after-c2-backward)
    PROGRESS="${OUT%.json}.progress.json"
    SUPERVISOR="${OUT%.json}.supervisor.json"
    python3 - "$SUPERVISOR" "$PROGRESS" <<'PY'
import json,sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({"state":"python-child-started","progress_file":sys.argv[2]},indent=2)+"\n")
PY
    if [[ -n "$TORCH_OVERLAY" ]]; then
      CMD=("$PYTHON_BIN" -c 'import runpy,sys; overlay=sys.argv.pop(1); module=sys.argv.pop(1); sys.path.insert(0, overlay); runpy.run_module(module, run_name="__main__")' "$TORCH_OVERLAY" scripts.gfx803_moe_gpu_gate "${GATE_ARGS[@]}" --repeats "$REPEATS" --stability-steps "$STABILITY_STEPS" --router-selector "$ROUTER_SELECTOR" "${DENSE_ARGS[@]}" "${C2_ARGS[@]}" --progress-file "$PROGRESS" --receipt "$OUT")
    else
      CMD=("$PYTHON_BIN" -m scripts.gfx803_moe_gpu_gate "${GATE_ARGS[@]}" --repeats "$REPEATS" --stability-steps "$STABILITY_STEPS" --router-selector "$ROUTER_SELECTOR" "${DENSE_ARGS[@]}" "${C2_ARGS[@]}" --progress-file "$PROGRESS" --receipt "$OUT")
    fi
    set +e
    if ((GDB_CAPTURE)); then
      gdb --batch --return-child-result -ex 'set pagination off' -ex run -ex 'thread apply all bt full' --args "${CMD[@]}" >"${OUT%.json}.gdb.log" 2>&1
      rc=$?
    else
      "${CMD[@]}"
      rc=$?
    fi
    set -e
    python3 - "$SUPERVISOR" "$rc" "$OUT" "$PROGRESS" <<'PY'
import json,sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({"state":"python-child-exited","exit_code":int(sys.argv[2]),"receipt_exists":Path(sys.argv[3]).is_file(),"progress_file":sys.argv[4]},indent=2)+"\n")
PY
    exit "$rc"
    ;;
esac
