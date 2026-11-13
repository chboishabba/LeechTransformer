#!/usr/bin/env bash
# Focused contracts; do not touch active experiments or rebuild ROCm.
set -euo pipefail
cd "$(dirname "$0")/.."
WRAPPER="${GFX803_SCHAKA101_PYTHON:-/nix/store/vq1lgpc49im7qvvxgxabykyfs47l2ykb-gfx803-schaka101-torch-candidate-overlay/bin/gfx803-schaka101-python}"
[[ -x "$WRAPPER" ]] || { echo "Selected Schaka Python wrapper missing: $WRAPPER" >&2; exit 2; }
"$WRAPPER" -m unittest discover -s tests -p 'test_native_streamed_selected_experts.py' -v
"$WRAPPER" -m unittest discover -s tests -p 'test_native_federated_selected_worker.py' -v
"$WRAPPER" -m unittest discover -s tests -p 'test_native_multiworker_coordinator.py' -v
"$WRAPPER" -m unittest discover -s tests -p 'test_compare_native_selected_federation.py' -v
echo "SOURCE CONTRACT TESTS PASSED; native multi-host execution and long-term durability remain unmeasured."
