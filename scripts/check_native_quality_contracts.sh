#!/usr/bin/env bash
# One quality-only source-contract check. Native GPU training is a separate run.
set -euo pipefail
cd "$(dirname "$0")/.."
WRAPPER="${GFX803_SCHAKA101_PYTHON:-/nix/store/vq1lgpc49im7qvvxgxabykyfs47l2ykb-gfx803-schaka101-torch-candidate-overlay/bin/gfx803-schaka101-python}"
[[ -x "$WRAPPER" ]] || { echo "Missing supplied Schaka native Python: $WRAPPER" >&2; exit 2; }
"$WRAPPER" -m unittest discover -s tests -p 'test_native_student_quality_curriculum.py' -v
"$WRAPPER" -m unittest discover -s tests -p 'test_guarded_original_teacher_identity.py' -v
echo 'Source identity, zero-gate teacher, curriculum split contracts passed; physical quality result not yet measured.'
