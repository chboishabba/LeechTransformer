#!/usr/bin/env bash
# Diagnostic only. Reuses the original gfx803_compat_graph Nix/PyTorch
# runtime; cannot launch training or bypass rejected numerical admission.
set -euo pipefail

THIS_SCRIPT="$(realpath "${BASH_SOURCE[0]}")"
LEECH_ROOT="$(cd "$(dirname "$THIS_SCRIPT")/.." && pwd)"
cd "$LEECH_ROOT"

if [[ -n "${GFX803_COMPAT_ROOT:-}" ]]; then
  COMPAT_ROOT="$GFX803_COMPAT_ROOT"
elif [[ -d /mnt/truenas/comet-net/gfx803_compat_graph ]]; then
  COMPAT_ROOT=/mnt/truenas/comet-net/gfx803_compat_graph
else
  COMPAT_ROOT="$HOME/Documents/code/__OTHER/gfx803_compat_graph"
fi
STACK="${GFX803_NIX_STACK:-gfx803-pytorch-stack}"
case "$STACK" in
  gfx803-pytorch-stack|gfx803-pytorch-stack-upgrade) ;;
  *) echo "refuse unknown Nix stack: $STACK" >&2; exit 2;;
esac
[[ -f "$COMPAT_ROOT/gfx803_flake_v1/flake.nix" &&
   -f "$COMPAT_ROOT/scripts/host-docker-python.sh" ]] || {
  echo "missing original gfx803_compat_graph Nix runtime" >&2; exit 2;
}
ARGS=("$@")
if [[ "${GFX803_DIAG_IN_NIX:-0}" != "1" ]]; then
  command -v nix >/dev/null || { echo "nix not found" >&2; exit 2; }
  export GFX803_DIAG_IN_NIX=1 GFX803_COMPAT_ROOT="$COMPAT_ROOT"
  cd "$COMPAT_ROOT/gfx803_flake_v1"
  exec nix develop ".#$STACK" -c bash "$THIS_SCRIPT" "${ARGS[@]}"
fi
export HIP_LAUNCH_BLOCKING="${HIP_LAUNCH_BLOCKING:-1}"
export PYTHONNOUSERSITE=1
cd "$LEECH_ROOT"
echo "[gfx803] Diagnostic ONLY; never authorizes GPU training"
echo "[gfx803] stack=$STACK HIP_LAUNCH_BLOCKING=$HIP_LAUNCH_BLOCKING"
exec bash "$COMPAT_ROOT/scripts/host-docker-python.sh" -m \
  scripts.diagnose_gfx803_leech_operator_boundary "${ARGS[@]}"
