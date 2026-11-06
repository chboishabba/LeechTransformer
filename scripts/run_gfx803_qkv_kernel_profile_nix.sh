#!/usr/bin/env bash
# Attribute the real block-0 QKV GEMM to profiler-visible GPU kernels.
# Diagnostic only: never trains, changes a checkpoint, or approves gfx803.
set -euo pipefail

THIS_SCRIPT="$(realpath "${BASH_SOURCE[0]}")"
LEECH_ROOT="$(cd "$(dirname "$THIS_SCRIPT")/.." && pwd)"
if [[ -n "${GFX803_COMPAT_ROOT:-}" ]]; then COMPAT_ROOT="$GFX803_COMPAT_ROOT"
elif [[ -d /mnt/truenas/comet-net/gfx803_compat_graph ]]; then COMPAT_ROOT=/mnt/truenas/comet-net/gfx803_compat_graph
else COMPAT_ROOT="$HOME/Documents/code/__OTHER/gfx803_compat_graph"; fi
STACK="${GFX803_NIX_STACK:-gfx803-pytorch-stack}"
[[ "$STACK" == gfx803-pytorch-stack || "$STACK" == gfx803-pytorch-stack-upgrade ]] || { echo "refuse unknown Nix stack: $STACK" >&2; exit 2; }
[[ -f "$COMPAT_ROOT/gfx803_flake_v1/flake.nix" && -x "$COMPAT_ROOT/scripts/host-docker-python.sh" ]] || { echo "missing gfx803 compatibility runtime" >&2; exit 2; }

if [[ "${GFX803_QKV_PROFILE_IN_NIX:-0}" != 1 ]]; then
  export GFX803_QKV_PROFILE_IN_NIX=1 GFX803_COMPAT_ROOT="$COMPAT_ROOT"
  exec nix develop "$COMPAT_ROOT/gfx803_flake_v1#$STACK" -c bash "$THIS_SCRIPT" "$@"
fi

OUT_DIR=""
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
[[ -n "$OUT_DIR" ]] || { echo "usage: $0 --out-dir DIR --teacher ... --token-ids ... [profile args]" >&2; exit 2; }
mkdir -p "$OUT_DIR/profiler"
export HIP_LAUNCH_BLOCKING="${HIP_LAUNCH_BLOCKING:-1}" PYTHONNOUSERSITE=1 ROCBLAS_LAYER="${ROCBLAS_LAYER:-3}"
cd "$LEECH_ROOT"

run_python() { bash "$COMPAT_ROOT/scripts/host-docker-python.sh" -m scripts.profile_gfx803_leech_qkv "$@"; }
printf 'stack=%s\ncompat_root=%s\nROCBLAS_LAYER=%s\n' "$STACK" "$COMPAT_ROOT" "$ROCBLAS_LAYER" > "$OUT_DIR/environment.txt"
run_python "${ARGS[@]}" --phase unprofiled --receipt "$OUT_DIR/unprofiled.json" > "$OUT_DIR/unprofiled.log" 2>&1

# rocprofv3's host-side SDK needs the host libstdc++ ABI.  Do not prepend the
# extracted Conda lib directory: its older libstdc++ breaks rocprofv3 (and made
# its child shell report unrelated missing libsqlite/libglog errors).
SYSTEM_LIBSTDCXX=/usr/lib/libstdc++.so.6
[[ -f "$SYSTEM_LIBSTDCXX" ]] || {
  echo "missing host profiler libstdc++ dependency: $SYSTEM_LIBSTDCXX" >&2
  exit 2
}
set +e
LD_PRELOAD="$SYSTEM_LIBSTDCXX${LD_PRELOAD:+:$LD_PRELOAD}" \
PATH="/usr/bin:/bin:/opt/rocm/bin:$PATH" \
rocprofv3 --runtime-trace --marker-trace --kernel-trace --summary --output-format csv \
  --selected-regions -d "$OUT_DIR/profiler" -- \
  /usr/bin/bash "$COMPAT_ROOT/scripts/host-docker-python.sh" -m scripts.profile_gfx803_leech_qkv \
  "${ARGS[@]}" --phase profiled --receipt "$OUT_DIR/profiled.json" > "$OUT_DIR/profiled.log" 2>&1
PROFILE_STATUS=$?
set -e
printf '%s\n' "$PROFILE_STATUS" > "$OUT_DIR/rocprofv3.exitcode"
python -m scripts.parse_gfx803_qkv_kernel_trace --trace-dir "$OUT_DIR/profiler" --output "$OUT_DIR/kernel_candidates.json" || true
echo "[gfx803-qkv] output=$OUT_DIR rocprofv3_exit=$PROFILE_STATUS"
exit "$PROFILE_STATUS"
