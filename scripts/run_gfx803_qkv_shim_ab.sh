#!/usr/bin/env bash
# Complete same-image 10.1 QKV SGEMM shim ON/OFF microrepro.
# **Requires** a coherent REAL Schaka 10.1 Python/ROCm runtime provided by
# the operator; does not use the legacy extracted 6.4 wrapper or train.
set -euo pipefail
LEECH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT='' FIXTURE='' PYTHON_BIN='' ROOT='' SHIM='' REPEATS=100
TIMEOUT_SECS=240 PROFILE=1 HIP_PREFIX=10.1 TRIAL=0
while (($#)); do
  case "$1" in
    --out-dir) OUT="$2"; shift 2 ;;
    --fixture) FIXTURE="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --rocm-root) ROOT="$2"; shift 2 ;;
    --shim) SHIM="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --hip-prefix) HIP_PREFIX="$2"; shift 2 ;;
    --timeout-seconds) TIMEOUT_SECS="$2"; shift 2 ;;
    --skip-profiling) PROFILE=0; shift ;;
    --trial) TRIAL=1; PROFILE=0; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$OUT" && -f "$FIXTURE/manifest.json" && -x "$PYTHON_BIN" &&
   -d "$ROOT" && -f "$SHIM" ]] || {
  echo "Usage: $0 --out-dir DIR --fixture ORIGINAL_QKV_FIXTURE --python EXACT_ROCM101_PYTHON --rocm-root EXACT_ROCM101_ROOT --shim SAME_ROOT/lib/libgfx803_sgemm_shim.so" >&2; exit 2; }
[[ "$REPEATS" =~ ^[1-9][0-9]*$ && "$TIMEOUT_SECS" =~ ^[1-9][0-9]*$ ]] || {
  echo "repeats and timeout must be positive" >&2; exit 2; }
# Full statistical A/B requires 100+ identical original-QKV repeats.
# Use --limit to bound the number of solutions, not weaker repeat evidence.
if ((REPEATS < 3)); then
  echo "Refuse weak A/B: --repeats must be >= 3" >&2
  exit 2
fi
[[ "$ROOT" == *"/nix/store/"* || "$ROOT" == /nix/store/* ]] || {
  echo "Refusing non-Nix modern ROCm root without explicit pin" >&2; exit 2; }
ROOT="$(realpath "$ROOT")" SHIM="$(realpath "$SHIM")" PYTHON_BIN="$(realpath "$PYTHON_BIN")"
[[ "$SHIM" == "$ROOT/"* && "$SHIM" == */libgfx803_sgemm_shim.so ]] || {
  echo "SGEMM shim must belong to the SAME runtime root" >&2; exit 2; }
[[ "$ROOT" != *"lib-compat"* && "$PYTHON_BIN" != *"docker-venv"* ]] || {
  echo "Do not mix the original 6.4 extracted Python with ROCm 10.1" >&2; exit 2; }
[[ -n "${LD_LIBRARY_PATH:-}" && "${LD_LIBRARY_PATH}" == *lib-compat* ]] && {
  echo "Inherited old lib-compat LD_LIBRARY_PATH would mix runtimes" >&2; exit 2; }
# Only a profiler from the selected 10.1 Nix runtime may trace the
# original shim calculation. Never inject the host's legacy 6.4 profiler
# runtime into a modern same-image causal comparison.
if [[ -n "${LD_PRELOAD:-}" ]]; then
  echo "Refuse inherited LD_PRELOAD in supervisor; invoke with env -u LD_PRELOAD" >&2
  exit 2
fi
PROFILE_BIN=''
if ((PROFILE)); then
  for candidate in "$ROOT/core-$HIP_PREFIX/bin/rocprofv3" "$ROOT/bin/rocprofv3"; do
    if [[ -x "$candidate" && "$(realpath "$candidate")" == "$ROOT/"* ]]; then
      PROFILE_BIN="$(realpath "$candidate")"
      break
    fi
  done
  [[ -n "$PROFILE_BIN" ]] || {
    echo "No same-root 10.1 rocprofv3; use --skip-profiling for numeric-only evidence" >&2
    exit 3
  }
fi
OUT="$(realpath -m "$OUT")"
FIXTURE="$(realpath "$FIXTURE")"
if [[ -e "$OUT" && -n "$(find "$OUT" -mindepth 1 -print -quit)" ]]; then
  echo "Refuse pre-existing shim A/B receipts; choose a fresh --out-dir" >&2
  exit 2
fi
mkdir -p "$OUT"
cd "$LEECH_ROOT"
export HIP_LAUNCH_BLOCKING=1 PYTHONNOUSERSITE=1
export HSA_OVERRIDE_GFX_VERSION=8.0.3 ROC_ENABLE_PRE_VEGA=1 TORCH_BLAS_PREFER_HIPBLASLT=0
# Preserve only this selected ROCm root ahead of separately pinned system libs.
export LD_LIBRARY_PATH="$ROOT/lib:$ROOT/core-$HIP_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
printf 'owner=Schaka original FP32 SGEMM shim\nruntime_root=%s\nshim=%s\npython=%s\nprofiler=%s\n' \
  "$ROOT" "$SHIM" "$PYTHON_BIN" "$PROFILE_BIN" >"$OUT/runtime-selection.txt"
date -Is >>"$OUT/runtime-selection.txt"
uname -a >>"$OUT/runtime-selection.txt"
common=(--mode shim --fixture "$FIXTURE" --expected-rocm-root "$ROOT"
        --shim-path "$SHIM" --expected-hip-prefix "$HIP_PREFIX"
        --repeats "$REPEATS")
failed=0
for arm in on off; do
  case "$arm" in on) disabled=0 ;; off) disabled=1 ;; esac
  dest="$OUT/$arm"
  mkdir -p "$dest"
  if [[ ! -f "$dest/unprofiled.json" ]]; then
    if ! timeout --kill-after=15 "$TIMEOUT_SECS" \
      env LD_PRELOAD="$SHIM" GFX803_SGEMM_SHIM_DISABLE="$disabled" \
      "$PYTHON_BIN" -m scripts.gfx803_qkv_solution_ab \
      "${common[@]}" --shim-mode "$arm" \
      --output-bin "$dest/first-output.f32" \
      --receipt "$dest/unprofiled.json" >"$dest/unprofiled.log" 2>&1; then
      echo "QKV SGEMM shim $arm original-model run failed; see $dest/unprofiled.log" >&2
      failed=$((failed+1)); continue
    fi
  fi
  if ((!PROFILE)); then continue; fi
  if [[ ! -f "$dest/profiled.json" ]]; then
    mkdir -p "$dest/profiler"
    set +e
    timeout --kill-after=15 "$TIMEOUT_SECS" \
      "$PROFILE_BIN" --runtime-trace --marker-trace --kernel-trace \
      --summary --output-format csv --selected-regions -d "$dest/profiler" -- \
      /usr/bin/env LD_PRELOAD="$SHIM" GFX803_SGEMM_SHIM_DISABLE="$disabled" \
      "$PYTHON_BIN" -m scripts.gfx803_qkv_solution_ab \
      "${common[@]}" --shim-mode "$arm" \
      --output-bin "$dest/profile-output.f32" \
      --receipt "$dest/profiled.json" >"$dest/profiled.log" 2>&1
    rc=$?
    set -e
    printf '%s\n' "$rc" >"$dest/rocprofv3.exitcode"
    if ((rc)); then failed=$((failed+1)); continue; fi
  fi
done
if ((PROFILE)); then
  if ! python3 -m scripts.parse_gfx803_qkv_shim_ab --out-dir "$OUT"; then
    echo "No completed, observer-effect-free shim A/B attribution; not a repair pass" >&2
    failed=$((failed+1))
  fi
else
  python3 - "$OUT" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
arms={}
for name in ("on","off"):
    receipt=p/name/"unprofiled.json"
    if not receipt.is_file():
        continue
    obj=json.loads(receipt.read_text())
    arms[name]={
        "status":obj.get("status"),
        "repeats_completed":obj.get("repeats_completed"),
        "teacher_sha256":obj.get("teacher_sha256"),
        "input_sha256":obj.get("input_sha256"),
        "rocblas_sha256":obj.get("rocblas_sha256"),
        "shim_sha256":obj.get("shim_sha256"),
        "first_gpu_output_sha256":obj.get("raw_first_output_sha256"),
        "shim_actually_mapped":obj.get("shim_actually_mapped"),
    }
complete=(set(arms)=={"on","off"} and
          all(a["repeats_completed"]==3 and a["shim_actually_mapped"] is True
              for a in arms.values()) and
          all(arms["on"][k]==arms["off"][k]
              for k in ("teacher_sha256","input_sha256",
                        "rocblas_sha256","shim_sha256")))
(p/"shim_ab_summary.json").write_text(json.dumps(
    {"status":"three-repeat-numeric-trial-only" if complete
        else "numeric-only-incomplete-or-nontrial",
     "trial_complete":complete,
     "arm_receipts":arms,
     "both_profiler_observer_controls_complete":False,
     "interposer_effect_attributed":False,
     "gpu_training_admitted":False,"release_accepted":False},
    indent=2,sort_keys=True)+"\n")
PY
fi
python3 - "$OUT" "$failed" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1])
files=[p for p in root.rglob("*") if p.is_file() and p.name!="artifact-index.json"]
(root/"artifact-index.json").write_text(json.dumps({
  "no_training_admission":True,
  "failed_or_unfinished_processes":int(sys.argv[2]),
  "sha256":{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(files)}},indent=2,sort_keys=True)+"\n")
print("QKV shim A/B artefacts:",root)
PY
echo "Shim A/B is a scoped diagnostic: GPU training remains blocked."
if ((failed)); then exit 5; fi
