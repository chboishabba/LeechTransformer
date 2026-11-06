#!/usr/bin/env bash
# First-party QKV exact-operand rocBLAS solution A/B using Schaka's method.
# Intentionally no GPU training or admission; failures remain on disk.
set -euo pipefail
THIS_SCRIPT="$(realpath "${BASH_SOURCE[0]}")"
LEECH_ROOT="$(cd "$(dirname "$THIS_SCRIPT")/.." && pwd)"
if [[ -n "${GFX803_COMPAT_ROOT:-}" ]]; then COMPAT_ROOT="$GFX803_COMPAT_ROOT"
elif [[ -d /mnt/truenas/comet-net/gfx803_compat_graph ]]; then COMPAT_ROOT=/mnt/truenas/comet-net/gfx803_compat_graph
else COMPAT_ROOT="$HOME/Documents/code/__OTHER/gfx803_compat_graph"; fi
STACK="${GFX803_NIX_STACK:-gfx803-pytorch-stack}"
[[ "$STACK" == gfx803-pytorch-stack ]] || { echo "Refuse non-baseline stack" >&2; exit 2; }
if [[ "${LD_PRELOAD:-}" == *"libgfx803_sgemm_shim"* ]]; then
  echo "Refuse foreign libgfx803_sgemm_shim preload in original ROCm 6.4 WGM scan" >&2
  exit 2
fi
[[ -f "$COMPAT_ROOT/gfx803_flake_v1/flake.nix" && -f "$COMPAT_ROOT/scripts/host-docker-python.sh" ]] || {
  echo "Missing original gfx803 compatibility runtime" >&2; exit 2; }
if [[ "${GFX803_QKV_AB_IN_NIX:-0}" != 1 ]]; then
  export GFX803_QKV_AB_IN_NIX=1 GFX803_COMPAT_ROOT="$COMPAT_ROOT"
  exec nix develop "$COMPAT_ROOT/gfx803_flake_v1#$STACK" -c bash "$THIS_SCRIPT" "$@"
fi
OUT='' TEACHER='' CONFIG='' TOKEN_IDS='' REPEATS=100 LIMIT=0
BATCH=2 TOKENS=64 TIMEOUT_SECS=240 PROFILE=1 IDS=()
while (($#)); do
  case "$1" in
    --out-dir) OUT="$2"; shift 2 ;;
    --teacher) TEACHER="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --token-ids) TOKEN_IDS="$2"; shift 2 ;;
    --batch) BATCH="$2"; shift 2 ;;
    --tokens) TOKENS="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --timeout-seconds) TIMEOUT_SECS="$2"; shift 2 ;;
    --solution-id) IDS+=("$2"); shift 2 ;;
    --skip-profiling) PROFILE=0; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$OUT" && -f "$TEACHER" && -f "$TOKEN_IDS" ]] || {
  echo "Usage: $0 --out-dir DIR --teacher FILE --token-ids FILE [--config FILE] [--repeats 100] [--solution-id ID]" >&2; exit 2; }
[[ "$REPEATS" =~ ^[1-9][0-9]*$ && "$LIMIT" =~ ^[0-9]+$ &&
 "$BATCH" =~ ^[1-9][0-9]*$ && "$TOKENS" =~ ^[1-9][0-9]*$ &&
 "$TIMEOUT_SECS" =~ ^[1-9][0-9]*$ ]] || { echo "Nonpositive integer" >&2; exit 2; }
[[ -z "$CONFIG" || -f "$CONFIG" ]] || { echo "No config" >&2; exit 2; }
for id in "${IDS[@]}"; do [[ "$id" =~ ^-?[0-9]+$ ]] || { echo "Bad solution ID" >&2; exit 2; }; done
[[ -d "$COMPAT_ROOT/lib-compat" ]] || { echo "No original lib-compat" >&2; exit 2; }
OUT="$(realpath -m "$OUT")"
mkdir -p "$OUT/solutions"
cd "$LEECH_ROOT"
export HIP_LAUNCH_BLOCKING=1 PYTHONNOUSERSITE=1 ROCBLAS_LAYER="${ROCBLAS_LAYER:-3}"
EXPECTED=6e33dd589ce2fdc69f953fe1e195016076b883a7948f72a002de05fb49056cb5
runner=(bash "$COMPAT_ROOT/scripts/host-docker-python.sh")
fixture="$OUT/fixture"
if [[ ! -f "$fixture/manifest.json" ]]; then
  export_args=(--teacher "$TEACHER" --token-ids "$TOKEN_IDS" --batch "$BATCH"
    --tokens "$TOKENS" --out-dir "$fixture" --expect-input-sha "$EXPECTED")
  [[ -z "$CONFIG" ]] || export_args+=(--config "$CONFIG")
  "${runner[@]}" -m scripts.export_gfx803_qkv_fixture "${export_args[@]}" >"$OUT/export.log" 2>&1
fi
python3 - "$fixture" "$EXPECTED" "$TEACHER" "$TOKEN_IDS" "$CONFIG" <<'PY'
from pathlib import Path
from scripts.export_gfx803_qkv_fixture import verify_fixture,sha256_file
import sys
m=verify_fixture(Path(sys.argv[1]),expected_input_sha=sys.argv[2])
assert m["teacher_sha256"]==sha256_file(Path(sys.argv[3])), "reused fixture has a different teacher checkpoint"
assert m["tokens_sha256"]==sha256_file(Path(sys.argv[4])), "reused fixture has a different token source"
if sys.argv[5]:
    assert m["config_sha256"]==sha256_file(Path(sys.argv[5])), "reused fixture has a different config"
else:
    assert m["config_sha256"] is None, "reused fixture had a config but this run omitted it"
assert m["gemm_logical_m_n_k"]==[128,1152,384], "original QKV shape changed"
print("input SHA verified:",m["tensors"]["input"]["sha256"])
PY
base=(--fixture "$fixture" --expected-rocm-root "$COMPAT_ROOT/lib-compat"
  --expected-input-sha "$EXPECTED")
enumeration="$OUT/enumeration.json"
if [[ ! -f "$enumeration" ]]; then
  "${runner[@]}" -m scripts.gfx803_qkv_solution_ab --mode enumerate "${base[@]}" \
    --repeats 1 --receipt "$enumeration" > "$OUT/enumeration.log" 2>&1
fi
python3 - "$enumeration" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
assert r["status"]=="enumerated-without-evaluating-solutions"
assert r["admission"]=="denied" and len(r["solution_ids"])>0
print("actual same-build solution count:",len(r["solution_ids"]))
PY
# The selected solution IDs belong to the exact enumerated library+fixture.
read -r pinned_lib pinned_fixture < <(python3 - "$enumeration" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
print(r["rocblas_sha256"],r["fixture_manifest_sha256"])
PY
)
base+=(--expected-library-sha "$pinned_lib" --expected-fixture-sha "$pinned_fixture")
if ((${#IDS[@]})); then
  chosen=("${IDS[@]}")
else
  mapfile -t chosen < <(python3 - "$enumeration" "$LIMIT" <<'PY'
import json,sys
ids=json.load(open(sys.argv[1]))["solution_ids"]
if int(sys.argv[2]): ids=ids[:int(sys.argv[2])]
print("\n".join(map(str,ids)))
PY
)
fi
if ((PROFILE)); then command -v rocprofv3 >/dev/null || {
 echo "rocprofv3 missing; use --skip-profiling for numeric-only scan" >&2; exit 3; }; fi
printf 'stack=%s\ncompat=%s\ninput_sha=%s\nrepeats=%s\nprofiling=%s\n' \
 "$STACK" "$COMPAT_ROOT" "$EXPECTED" "$REPEATS" "$PROFILE" >"$OUT/environment.txt"
date -Is >>"$OUT/environment.txt"
git rev-parse HEAD >>"$OUT/environment.txt" || true
uname -a >>"$OUT/environment.txt"
failed=0
for id in "${chosen[@]}"; do
  dest="$OUT/solutions/$id"
  mkdir -p "$dest"
  if [[ ! -f "$dest/unprofiled.json" ]]; then
    echo "[QKV] original-operand solution $id ($REPEATS repeats)"
    if ! timeout --kill-after=15 "$TIMEOUT_SECS" \
        "${runner[@]}" -m scripts.gfx803_qkv_solution_ab --mode run \
        "${base[@]}" --solution-id "$id" --repeats "$REPEATS" \
        --output-bin "$dest/first-output.f32" \
        --receipt "$dest/unprofiled.json" >"$dest/unprofiled.log" 2>&1; then
      echo "Failed unprofiled $id; retained $dest/unprofiled.log" >&2
      failed=$((failed+1)); continue
    fi
  fi
  if ((!PROFILE)); then continue; fi
  if [[ ! -f "$dest/profiled.json" ]]; then
    SYSTEM_LIBSTDCXX=/usr/lib/libstdc++.so.6
    [[ -f "$SYSTEM_LIBSTDCXX" ]] || { echo "Profiler host libstdc++ unavailable" >&2; exit 3; }
    mkdir -p "$dest/profiler"
    set +e
    LD_PRELOAD="$SYSTEM_LIBSTDCXX" \
    PATH="/usr/bin:/bin:/opt/rocm/bin:$PATH" \
    timeout --kill-after=15 "$TIMEOUT_SECS" \
      rocprofv3 --runtime-trace --marker-trace --kernel-trace --summary \
      --output-format csv --selected-regions -d "$dest/profiler" -- \
      /usr/bin/bash "$COMPAT_ROOT/scripts/host-docker-python.sh" \
      -m scripts.gfx803_qkv_solution_ab --mode run \
      "${base[@]}" --solution-id "$id" --repeats 1 \
      --output-bin "$dest/profile-output.f32" \
      --receipt "$dest/profiled.json" >"$dest/profiled.log" 2>&1
    rc=$?
    set -e
    printf '%s\n' "$rc" >"$dest/rocprofv3.exitcode"
    if ((rc!=0)); then
      echo "Profiler failed solution $id; cannot name kernel" >&2
      failed=$((failed+1)); continue
    fi
  fi
  if [[ -f "$dest/profiled.json" && ! -f "$dest/joined.json" ]]; then
    if ! python3 -m scripts.parse_gfx803_qkv_solution_ab join \
      --trace "$dest/profiler" --unprofiled "$dest/unprofiled.json" \
      --profiled "$dest/profiled.json" --out "$dest/joined.json" \
      >"$dest/join.log" 2>&1; then
      echo "No valid rocprofv3 WGM symbol attribution for solution $id" >&2
      failed=$((failed+1))
    fi
  fi
done
if ((PROFILE)); then
  if ! python3 -m scripts.parse_gfx803_qkv_solution_ab summarize \
    --root "$OUT/solutions" --out "$OUT/wgm_ab_summary.json"; then
    echo "No completed observer-effect-free same-library WGM8-bad/WGM1-good contrast; keep rejection" >&2
    failed=$((failed+1))
  fi
fi
python3 - "$OUT" "$failed" <<'PY'
import json,sys,hashlib
from pathlib import Path
root=Path(sys.argv[1])
files=[p for p in root.rglob("*") if p.is_file() and p.name!="artifact-index.json"]
index={"purpose":"same-library exact-QKV A/B, never a training gate",
       "unprocessed_or_failed_solution_processes":int(sys.argv[2]),
       "admission":"denied",
       "sha256":{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(files)}}
(root/"artifact-index.json").write_text(json.dumps(index,indent=2,sort_keys=True)+"\n")
print("Receipts:",root,"failures:",index["unprocessed_or_failed_solution_processes"])
PY
echo "No training, WGM fix, or 10.1 release admission was performed."
if ((failed)); then exit 5; fi
