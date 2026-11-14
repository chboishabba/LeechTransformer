#!/usr/bin/env bash
# X369 CPU exact finite / source-geometric contracts only.
# Does NOT replace pinned Schaka native suite, model GPU runs or Agda kernel.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${X369_PYTHON:-python3}"
for pattern in \
  test_x369_original_torus_maxcut.py \
  test_certify_c3_task_orbits_original_tokens.py \
  test_audit_x369_agda_original_crosswalk.py \
  test_c3_matched_router_control.py \
  test_c3_covariance_f1_maxcut.py \
  test_prepare_x369_document_disjoint_fresh.py \
  test_x369_f1_rank_and_orientation_null.py \
  test_x369_f1_spectral_pair_identifiability.py \
  test_x369_f4_development_fairness.py \
  test_x369_f3_f4_router_feature_metric.py \
  test_x369_f3_mixed_orientation_ensemble_ce.py; do
  "$PY" -m unittest discover -s tests -p "$pattern" -v
done
"$PY" -m scripts.certify_x369_original_torus_maxcut
if [[ -n "${X369_AGDA_A00_SOURCE:-}" ]]; then
  OUTDIR="$(mktemp -d)"
  trap 'rm -rf "$OUTDIR"' EXIT
  "$PY" -m scripts.audit_x369_agda_original_crosswalk \
    --agda-source "$X369_AGDA_A00_SOURCE" \
    --output "$OUTDIR/crosswalk.json"
else
  echo "A00 actual cross-repo source file not supplied; fixture regression only."
fi
echo "X369 CPU source contracts completed; no Agda kernel, native suite or neural mechanism admission."
