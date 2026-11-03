#!/usr/bin/env bash
# Reuse gfx803_compat_graph's ORIGINAL Nix flake and extracted Python.
# No pip installs, ROCm overlays, or alternative CUDA/torch environment.
# GPU training is FORBIDDEN until the real-checkpoint numerical gate passes.
set -euo pipefail

THIS_SCRIPT="$(realpath "${BASH_SOURCE[0]}")"
LEECH_ROOT="$(cd "$(dirname "$THIS_SCRIPT")/.." && pwd)"
# Resolve relative checkpoint and token paths from LeechTransformer both
# before and after entering nix develop from the compat flake directory.
cd "$LEECH_ROOT"
COMPAT_ROOT="${GFX803_COMPAT_ROOT:-${HOME}/Documents/code/__OTHER/gfx803_compat_graph}"
STACK="${GFX803_NIX_STACK:-gfx803-pytorch-stack}"
case "$STACK" in
  gfx803-pytorch-stack|gfx803-pytorch-stack-upgrade) ;;
  *) echo "ERROR: unknown or unapproved Nix stack: $STACK" >&2; exit 2;;
esac

usage() {
  cat >&2 <<'USAGE'
Usage (no CPU fallback):
  GFX803_COMPAT_ROOT=/path/to/gfx803_compat_graph \
  bash scripts/run_gfx803_moe_nix.sh \
    --teacher data/best_model.pt --student /path/to/C2/student_model.pt \
    --config /path/to/config.json --train-ids /path/to/fit_ids.npy \
    --receipt /path/to/gfx803_gpu_gate.json \
    -- scripts.train_guarded_moe_continuation \
    --teacher data/best_model.pt --student /path/to/C2/student_model.pt \
    --config /path/to/config.json --train-ids /path/to/fit_ids.npy \
    --heldout-ids /path/to/development_ids.npy --focus-layers 1 \
    --matched-control-objective --epochs 3 --batch 2 \
    --output /path/to/guarded_gpu_candidate.pt

Omit "-- module ..." to run only the numerical GPU preflight.
Use GFX803_NIX_STACK=gfx803-pytorch-stack-upgrade for the separately
tracked upgrade lane; the same numerical gate still applies.
USAGE
}
if [[ $# -lt 1 ]]; then usage; exit 2; fi

# Keep arguments unchanged across the nix develop re-exec.
ORIGINAL_ARGS=("$@")
PROBE=()
while [[ $# -gt 0 && "$1" != "--" ]]; do
  PROBE+=("$1"); shift
done
MODULE=""
TRAIN_ARGS=()
if [[ $# -gt 0 && "$1" == "--" ]]; then
  shift
  if [[ $# -lt 1 ]]; then usage; exit 2; fi
  MODULE="$1"; shift
  TRAIN_ARGS=("$@")
  case "$MODULE" in
    scripts.train_guarded_moe_continuation) ;;
    *) echo "ERROR: module $MODULE is not admitted for GPU training" >&2; exit 2;;
  esac
fi

argument_value() {
  local flag="$1"; shift
  while [[ $# -gt 0 ]]; do
    if [[ "$1" == "$flag" ]]; then
      [[ $# -gt 1 ]] || return 1
      printf '%s' "$2"
      return 0
    fi
    shift
  done
  return 1
}
for required in --teacher --student --train-ids --receipt; do
  if ! argument_value "$required" "${PROBE[@]}" >/dev/null; then
    echo "ERROR: preflight missing $required" >&2; exit 2
  fi
done
if [[ -n "$MODULE" ]]; then
  for required in --teacher --student --train-ids; do
    probe_path="$(argument_value "$required" "${PROBE[@]}")"
    train_path="$(argument_value "$required" "${TRAIN_ARGS[@]}")" || {
      echo "ERROR: train command is missing $required" >&2; exit 2;
    }
    if [[ "$(realpath "$probe_path")" != "$(realpath "$train_path")" ]]; then
      echo "ERROR: $required differs between GPU gate and training" >&2; exit 2
    fi
  done
  if argument_value --device "${TRAIN_ARGS[@]}" >/dev/null; then
    echo "ERROR: do not supply --device; launcher forces GPU cuda" >&2; exit 2
  fi
  probe_batch="$(argument_value --probe-batch "${PROBE[@]}" || echo 2)"
  train_batch="$(argument_value --batch "${TRAIN_ARGS[@]}" || echo 2)"
  if [[ "$probe_batch" != "$train_batch" ]]; then
    echo "ERROR: preflight batch $probe_batch does not match training $train_batch" >&2; exit 2
  fi
  # If config is supplied in both, it must refer to the same file.
  probe_config="$(argument_value --config "${PROBE[@]}" || true)"
  train_config="$(argument_value --config "${TRAIN_ARGS[@]}" || true)"
  if [[ -n "$probe_config" && -n "$train_config" &&
        "$(realpath "$probe_config")" != "$(realpath "$train_config")" ]]; then
    echo "ERROR: config differs between GPU preflight and training" >&2; exit 2
  fi
fi

if [[ ! -f "$COMPAT_ROOT/gfx803_flake_v1/flake.nix" ||
      ! -f "$COMPAT_ROOT/scripts/host-docker-python.sh" ]]; then
  echo "ERROR: gfx803_compat_graph runtime not found at $COMPAT_ROOT" >&2
  echo "Set GFX803_COMPAT_ROOT to your existing checkout; do not reinstall torch." >&2
  exit 2
fi

if [[ "${GFX803_IN_NIX:-0}" != "1" ]]; then
  command -v nix >/dev/null || {
    echo "ERROR: nix is unavailable; refusing any CPU-only fallback" >&2; exit 2;
  }
  cd "$COMPAT_ROOT/gfx803_flake_v1"
  export GFX803_COMPAT_ROOT="$COMPAT_ROOT"
  export GFX803_IN_NIX=1
  exec nix develop ".#$STACK" -c bash "$THIS_SCRIPT" "${ORIGINAL_ARGS[@]}"
fi

cd "$LEECH_ROOT"
export HIP_LAUNCH_BLOCKING="${HIP_LAUNCH_BLOCKING:-1}"
export PYTHONNOUSERSITE=1
# Device-lib/ABI resolution belongs to the compat repo's original wrapper.
# The host-docker-python.sh runner sanitizes incompatible Nix ROCm paths.
RUNNER="$COMPAT_ROOT/scripts/host-docker-python.sh"
receipt="$(argument_value --receipt "${PROBE[@]}")"
echo "[gfx803] Nix stack=$STACK, HIP_LAUNCH_BLOCKING=$HIP_LAUNCH_BLOCKING"
echo "[gfx803] Running checkpoint-bound CPU/HIP forward/backward/step gate"
if ! bash "$RUNNER" -m scripts.gfx803_moe_gpu_gate "${PROBE[@]}"; then
  echo "ERROR: numerical GPU gate did not pass. Training was NOT launched." >&2
  exit 2
fi

if [[ -z "$MODULE" ]]; then
  echo "[gfx803] Gate passed; check-only mode. No training launched."
  exit 0
fi
echo "[gfx803] Gate passed: launching $MODULE on ROCm/HIP cuda:0"
exec bash "$RUNNER" -m "$MODULE" "${TRAIN_ARGS[@]}" --device cuda
