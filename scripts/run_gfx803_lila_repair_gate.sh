#!/usr/bin/env bash
# Run real original LILA checkpoint-bound dense/C2 numerical gates on gfx803.
# 6.4: explicit slow CPU-GEMM substitution, not a GPU kernel fix.
# 10.1: Schaka's genuine SGEMM interposer in same-root runtime.
# Never launches training or silently admits the hybrid 6.4 path.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
LANE='' OUT='' PYTHON_BIN='' ROOT='' SHIM='' REPEATS=3
GATE_ARGS=()
usage() {
  cat >&2 <<'EOF'
Usage:
  bash scripts/run_gfx803_lila_repair_gate.sh --lane 6.4-cpu-linear \
    --receipt artifacts/repair64.json -- --teacher data/best_model.pt \
    --student /path/C2/student.pt --config /path/config.json --train-ids /path/train_ids.npy
  env -u LD_PRELOAD bash scripts/run_gfx803_lila_repair_gate.sh --lane 10.1-schaka \
    --python /nix/store/.../bin/python --rocm-root /nix/store/...-rocm \
    --shim /nix/store/...-rocm/lib/libgfx803_sgemm_shim.so \
    --receipt artifacts/repair101.json -- --teacher ... --student ... --train-ids ...
All evidence is an original-model FP32 CPU/GPU forward/backward/optimizer
comparison. Use a fresh receipt path. Pass --repeats N (>=2) for repeated
unmodified forwards. No automatic training, no mixed ROCm environment.
EOF
}
while (($#)); do
  case "$1" in
    --lane) LANE="${2:?}"; shift 2;;
    --python) PYTHON_BIN="${2:?}"; shift 2;;
    --rocm-root) ROOT="${2:?}"; shift 2;;
    --shim) SHIM="${2:?}"; shift 2;;
    --receipt) OUT="${2:?}"; shift 2;;
    --repeats) REPEATS="${2:?}"; shift 2;;
    --) shift; GATE_ARGS=("$@"); break;;
    --help|-h) usage; exit 0;;
    *) echo "Unknown option: $1" >&2; usage; exit 2;;
  esac
done
[[ "$LANE" == 6.4-cpu-linear || "$LANE" == 10.1-schaka ]] || { usage; exit 2; }
[[ -n "$OUT" && "$REPEATS" =~ ^[0-9]+$ && "$REPEATS" -ge 2 ]] || { usage; exit 2; }
for key in --teacher --student --train-ids; do
  found=0
  for item in "${GATE_ARGS[@]}"; do [[ "$item" != "$key" ]] || found=1; done
  [[ "$found" == 1 ]] || { echo "Missing mandatory $key" >&2; exit 2; }
done
for item in "${GATE_ARGS[@]}"; do
  case "$item" in --receipt|--repeats) echo "Don't duplicate --receipt/--repeats" >&2; exit 2;; esac
done
OUT="$(realpath -m "$OUT")"
[[ ! -e "$OUT" ]] || { echo "Refuse to overwrite old repair evidence: $OUT" >&2; exit 2; }
mkdir -p "$(dirname "$OUT")"
export HIP_LAUNCH_BLOCKING=1 PYTHONNOUSERSITE=1
export HSA_OVERRIDE_GFX_VERSION=8.0.3 ROC_ENABLE_PRE_VEGA=1
export TORCH_BLAS_PREFER_HIPBLASLT=0
case "$LANE" in
  6.4-cpu-linear)
    [[ -z "$ROOT" && -z "$SHIM" && -z "$PYTHON_BIN" && -z "${LD_PRELOAD:-}" ]] || {
      echo "6.4 hybrid lane rejects 10.1 arguments or inherited preload" >&2; exit 2; }
    export GFX803_LINEAR_REPAIR=cpu-fp32
    unset GFX803_EXPECT_SHIM GFX803_EXPECT_ROCM_ROOT GFX803_SGEMM_SHIM_DISABLE
    exec bash scripts/run_gfx803_moe_nix.sh \
      "${GATE_ARGS[@]}" --repeats "$REPEATS" --receipt "$OUT"
    ;;
  10.1-schaka)
    [[ -z "${LD_PRELOAD:-}" && -z "${GFX803_LINEAR_REPAIR:-}" ]] || {
      echo "Refuse inherited LD_PRELOAD/6.4 hybrid mode" >&2; exit 2; }
    [[ -n "$PYTHON_BIN" && -n "$ROOT" && -n "$SHIM" ]] || { usage; exit 2; }
    PYTHON_BIN="$(realpath "$PYTHON_BIN")"
    ROOT="$(realpath "$ROOT")"
    SHIM="$(realpath "$SHIM")"
    [[ -x "$PYTHON_BIN" && -f "$SHIM" && "$ROOT" == /nix/store/* &&
       "$SHIM" == "$ROOT/"* && "$SHIM" == */libgfx803_sgemm_shim.so &&
       "$PYTHON_BIN" != *docker-venv* && "$ROOT" != *lib-compat* ]] || {
      echo "Refuse unpinned, mixed or missing Schaka 10.1 runtime" >&2; exit 2; }
    [[ "${LD_LIBRARY_PATH:-}" != *lib-compat* ]] || {
      echo "6.4 lib-compat detected in 10.1 environment" >&2; exit 2; }
    export GFX803_EXPECT_SHIM="$SHIM" GFX803_EXPECT_ROCM_ROOT="$ROOT"
    export GFX803_SGEMM_SHIM_DISABLE=0 LD_PRELOAD="$SHIM"
    export LD_LIBRARY_PATH="$ROOT/lib:$ROOT/core-10.1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
    # Process-level attestation and original checkpoint test happen in the
    # same process; external source-only checks never substitute for hardware.
    exec "$PYTHON_BIN" -m scripts.gfx803_moe_gpu_gate \
      "${GATE_ARGS[@]}" --repeats "$REPEATS" --receipt "$OUT"
    ;;
esac
