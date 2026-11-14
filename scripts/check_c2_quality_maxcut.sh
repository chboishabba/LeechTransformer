#!/usr/bin/env bash
# Local source-semantic tests for Q2.1-Q3; not a native RX 580 acceptance run.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${GFX803_SCHAKA101_PYTHON:-/nix/store/vq1lgpc49im7qvvxgxabykyfs47l2ykb-gfx803-schaka101-torch-candidate-overlay/bin/gfx803-schaka101-python}"
[[ -x "$PY" ]] || { echo "Working Schaka torch interpreter required: $PY" >&2; exit 2; }
"$PY" -m unittest discover -s tests -p 'test_c2_quality_maxcut.py' -v
"$PY" -m unittest discover -s tests -p 'test_original_c2c3_source_actions.py' -v
"$PY" -m unittest discover -s tests -p 'test_c2_history_even_repair.py' -v
"$PY" -m unittest discover -s tests -p 'test_c2_native_bounded_warmstart.py' -v
"$PY" -m unittest discover -s tests -p 'test_balanced_ternary_c3_grokking_source.py' -v
"$PY" -m unittest discover -s tests -p 'test_bounded_c2_c2c3_matched_screen.py' -v
"$PY" -m unittest discover -s tests -p 'test_frozen_c2c3_lockbox.py' -v
"$PY" -m unittest discover -s tests -p 'test_c3_rotating_energy_null.py' -v
"$PY" -m unittest discover -s tests -p 'test_audit_c2_forward_only_cost.py' -v
"$PY" -m unittest discover -s tests -p 'test_guarded_original_teacher_identity.py' -v
"$PY" -m unittest discover -s tests -p 'test_native_student_quality_curriculum.py' -v
echo "Q2-Q3 model semantic contracts accepted; source GPU training and teacher-free promotion are separate evidence gates."
