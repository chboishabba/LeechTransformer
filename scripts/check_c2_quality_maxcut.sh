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
"$PY" -m unittest discover -s tests -p 'test_c3_covariance_f1_maxcut.py' -v
"$PY" -m unittest discover -s tests -p 'test_c3_source_actions_maxcut.py' -v
"$PY" -m unittest discover -s tests -p 'test_c3_causal_router_ce.py' -v
"$PY" -m unittest discover -s tests -p 'test_c3_square_three_arm_f6.py' -v
"$PY" -m unittest discover -s tests -p 'test_trained_c3_task_orbits.py' -v
"$PY" -m unittest discover -s tests -p 'test_frozen_three_arm_c3square_f6.py' -v
"$PY" -m unittest discover -s tests -p 'test_x369_original_torus_maxcut.py' -v
"$PY" -m unittest discover -s tests -p 'test_certify_c3_task_orbits_original_tokens.py' -v
"$PY" -m unittest discover -s tests -p 'test_audit_x369_agda_original_crosswalk.py' -v
"$PY" -m unittest discover -s tests -p 'test_c3_matched_router_control.py' -v
"$PY" -m unittest discover -s tests -p 'test_prepare_x369_document_disjoint_fresh.py' -v
"$PY" -m unittest discover -s tests -p 'test_x369_f1_rank_and_orientation_null.py' -v
"$PY" -m unittest discover -s tests -p 'test_x369_f1_spectral_pair_identifiability.py' -v
"$PY" -m unittest discover -s tests -p 'test_x369_f4_development_fairness.py' -v
"$PY" -m unittest discover -s tests -p 'test_x369_f3_f4_router_feature_metric.py' -v
"$PY" -m unittest discover -s tests -p 'test_x369_f3_mixed_orientation_ensemble_ce.py' -v
"$PY" -m unittest discover -s tests -p 'test_slr_spacy_parallel_surface.py' -v
"$PY" -m unittest discover -s tests -p 'test_audit_x369_slr_spacy_grammar_early_epoch.py' -v
"$PY" -m unittest discover -s tests -p 'test_x369_long_grokking_receipts.py' -v
"$PY" -m unittest discover -s tests -p 'test_x369_slr_joint_grokking.py' -v
"$PY" -m unittest discover -s tests -p 'test_x369_f3_two_seed_maxcut.py' -v
echo "Q2-Q3 model semantic contracts accepted; source GPU training and teacher-free promotion are separate evidence gates."
