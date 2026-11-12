#!/usr/bin/env bash
# Existing composed Schaka ROCm 10.1 + patched PyTorch only.
set -euo pipefail
cd "$(dirname "$0")/.."
WRAPPER="${GFX803_SCHAKA101_PYTHON:-/nix/store/vq1lgpc49im7qvvxgxabykyfs47l2ykb-gfx803-schaka101-torch-candidate-overlay/bin/gfx803-schaka101-python}"
[[ -x "$WRAPPER" ]] || { echo "Missing provided ROCm 10.1 wrapper: $WRAPPER" >&2; exit 2; }
[[ -z "${GFX803_LINEAR_REPAIR:-}" ]] || { echo "Refuse CPU GEMM substitution" >&2; exit 2; }
unset LD_PRELOAD
export PYTHONNOUSERSITE=1 HSA_OVERRIDE_GFX_VERSION=8.0.3 ROC_ENABLE_PRE_VEGA=1
export TORCH_BLAS_PREFER_HIPBLASLT=0
case "${1:-}" in
  --probe)
    [[ $# == 1 ]] || exit 2
    exec "$WRAPPER" -c 'import torch; assert torch.cuda.is_available(), "native HIP unavailable"; x=torch.ones((8,8),device="cuda"); print({"device":str(x.device),"sum":float((x@x).sum().item()),"hip":torch.version.hip})'
    ;;
  --module)
    [[ $# -ge 4 && "$3" == "--" ]] || { echo "Usage: $0 --module scripts.train_signed_wave_experts -- [original args without --device]" >&2; exit 2; }
    MODULE="$2"; shift 3
    case "$MODULE" in
      scripts.train_signed_wave_experts|scripts.train_guarded_moe_continuation|scripts.train_early_layer_dense_control|scripts.native_streamed_selected_experts|scripts.run_real_moe_jq1_experiment|scripts.capture_joint_moe_state) ;;
      *) echo "Unrecognised source module $MODULE" >&2; exit 2;;
    esac
    for arg in "$@"; do case "$arg" in --device|--device=*) echo "Device is forced to cuda" >&2; exit 2;; esac; done
    "$WRAPPER" -c 'import torch; assert torch.cuda.is_available(), "native HIP unavailable"; print("native GPU:",torch.cuda.get_device_name(0),"HIP:",torch.version.hip)'
    exec "$WRAPPER" -m "$MODULE" "$@" --device cuda
    ;;
  *) echo "Usage: $0 --probe | --module <approved module> -- <original model args>" >&2; exit 2;;
esac
