"""Join per-solution rocprofv3 ROCTX intervals to actual Tensile kernel names.

Fail closed on renamed marker-only traces, missing ranges, ambiguous symbols,
different library hashes or profiler observer effects. Does NOT infer WGM
from the solution index (those IDs are only local to a specific rocBLAS build).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

MARKER = re.compile(r"^leech:qkv:solution:(-?\d+)$")
WGM = re.compile(r"(?:^|_)WGM(-?\d+)(?:_|$)")
TENSILE = re.compile(r"(?:^|_)(?:Cijk|Tensile)", re.IGNORECASE)


def rows(root: Path, pattern: str):
    for file in sorted(Path(root).rglob(pattern)):
        with file.open(newline="", errors="replace") as fd:
            for row in csv.DictReader(fd):
                yield file, row


def exact_marker_intervals(trace: Path, expected_id: int) -> list[tuple[int, int]]:
    matches = []
    for file, row in rows(trace, "*marker_api_trace.csv"):
        text = row.get("Function", "")
        m = MARKER.fullmatch(text)
        if m is None or int(m.group(1)) != expected_id:
            continue
        start, end = int(row["Start_Timestamp"]), int(row["End_Timestamp"])
        if start < end:
            matches.append((start, end))
    return sorted(set(matches))


def match_tensile(trace: Path, solution_id: int) -> dict:
    intervals = exact_marker_intervals(trace, solution_id)
    if len(intervals) != 1:
        return {"status": "missing-or-ambiguous-marker", "intervals": intervals,
                "kernels": []}
    low, high = intervals[0]
    kernels = []
    for file, row in rows(trace, "*kernel_trace.csv"):
        name = (row.get("Kernel_Name") or "").strip()
        if not TENSILE.search(name):
            continue
        try:
            start = int(row["Start_Timestamp"])
            end = int(row["End_Timestamp"])
        except (ValueError, KeyError):
            continue
        if low <= start <= end <= high:
            group = WGM.search(name)
            kernels.append({"name": name, "wgm": int(group.group(1)) if group else None,
                            "start": start, "end": end,
                            "correlation_id": row.get("Correlation_Id"),
                            "csv": str(file.relative_to(trace))})
    unique = sorted({k["name"] for k in kernels})
    status = "identified" if len(unique) == 1 and all(k["wgm"] is not None for k in kernels) else "missing-or-ambiguous-kernel"
    return {"status": status, "intervals": intervals,
            "kernels": kernels, "unique_kernel_names": unique,
            "wgm": kernels[0]["wgm"] if status == "identified" else None}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def join_profile(unprofiled: dict, profiled: dict, matched: dict) -> dict:
    required = ("solution_id", "rocblas_sha256", "fixture_manifest_sha256",
                "input_sha256", "weight_sha256")
    for key in required:
        if unprofiled.get(key) != profiled.get(key):
            raise ValueError("profiled/unprofiled solution or same-library/fixture identity mismatch: " + key)
    if (unprofiled.get("solution_id") is None or
            unprofiled.get("status") == "inapplicable" or
            profiled.get("status") == "inapplicable"):
        raise ValueError("cannot attribute an inapplicable solution")
    hashes1, hashes2 = (unprofiled.get("output_sha256_by_repeat") or [],
                         profiled.get("output_sha256_by_repeat") or [])
    if not hashes1 or not hashes2:
        raise ValueError("missing actual GPU output hashes")
    stable = hashes1[0] == hashes2[0]
    return {
        "schema_version": 1,
        "solution_id": unprofiled["solution_id"],
        "rocblas_realpath": unprofiled["rocblas_realpath"],
        "rocblas_sha256": unprofiled["rocblas_sha256"],
        "fixture_manifest_sha256": unprofiled["fixture_manifest_sha256"],
        "matches_original_problem_api": False,
        "reason": "solution-index path is rocblas_gemm_ex; original QKV used rocblas_sgemm",
        "profile_observer_effect_absent_for_first_output": stable,
        "unprofiled_gpu_output_sha256": hashes1[0],
        "profiled_gpu_output_sha256": hashes2[0],
        "unprofiled_numerics": unprofiled["status"],
        "profiler_numerics": profiled["status"],
        "trace": matched,
        "attribution_complete": matched["status"] == "identified" and stable,
        "no_training_admission": True,
    }


def summarize(joins: list[dict]) -> dict:
    by_mapping = {}
    original = "Cijk_Alik_Bljk_SB_MT64x128x16_SN_AMAS3_BL1_BS1_EPS1_GLVWA4_GLVWB4_GRVWn1_GSU1_GSUASB_ISA803_K1_KLA_LPA4_LPB4_LRVW4_MMFGLC_NLCA1_NLCB1_PGR1_PLR1_SIA1_SU32_SUS256_SVW4_TT4_8_USFGROn1_VAW1_VSn1_VW4_VWB4_WG16_16_1_WGM8"
    if not joins:
        return {"status": "no-measured-solutions", "repair_accepted": False}
    one_build = len({j["rocblas_sha256"] for j in joins}) == 1
    one_fixture = len({j["fixture_manifest_sha256"] for j in joins}) == 1
    for item in joins:
        trace = item.get("trace", {})
        if not item.get("attribution_complete"):
            continue
        wgm = trace.get("wgm")
        by_mapping.setdefault(str(wgm), []).append({
            "solution_id": item["solution_id"],
            "kernel": trace["unique_kernel_names"][0],
            "original_exact_kernel": trace["unique_kernel_names"][0] == original,
            "unprofiled_numerics": item["unprofiled_numerics"],
            "observer_effect_absent": item["profile_observer_effect_absent_for_first_output"]})
    wgm8_fails = any(x["unprofiled_numerics"] == "incorrect-at-tested-shape"
                     for x in by_mapping.get("8", []))
    wgm1_passes = any(x["unprofiled_numerics"] == "correct-at-tested-shape"
                      for x in by_mapping.get("1", []))
    original_exact = any(x["original_exact_kernel"] for x in by_mapping.get("8", []))
    observed = one_build and one_fixture and wgm8_fails and wgm1_passes
    return {"status": ("supports-scoped-mapping-hypothesis" if observed
                       else "causal-discrimination-incomplete"),
            "one_library_build": one_build, "one_fixture": one_fixture,
            "original_sgemm_kernel_exactly_reproduced_by_gemm_ex": original_exact,
            "wgm8_wrong_and_wgm1_correct": observed,
            "mapping_groups": by_mapping,
            "repair_accepted": False,
            "caveat": ("Forced rocblas_gemm_ex can select a different code path from the "
                       "original rocblas_sgemm. WGM1 success is necessary evidence, "
                       "not proof that an in-place WGM patch fixes the complete model."),
            "no_training_admission": True}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    one = sub.add_parser("join")
    one.add_argument("--trace", type=Path, required=True)
    one.add_argument("--unprofiled", type=Path, required=True)
    one.add_argument("--profiled", type=Path, required=True)
    one.add_argument("--out", type=Path, required=True)
    group = sub.add_parser("summarize")
    group.add_argument("--root", type=Path, required=True)
    group.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    try:
        if args.mode == "join":
            old, new = json.loads(args.unprofiled.read_text()), json.loads(args.profiled.read_text())
            matched = match_tensile(args.trace, int(old["solution_id"]))
            result = join_profile(old, new, matched)
        else:
            joins = [json.loads(p.read_text()) for p in sorted(args.root.rglob("joined.json"))]
            result = summarize(joins)
    except (OSError, ValueError, KeyError) as exc:
        p.exit(3, "QKV TRACE NOT ATTRIBUTED: " + str(exc) + "\n")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(args.out), "status": result["status"],
                      "admission": False}, sort_keys=True))
    if args.mode == "join":
        return 0 if result["attribution_complete"] else 4
    return 0 if result["status"] == "supports-scoped-mapping-hypothesis" else 4


if __name__ == "__main__":
    raise SystemExit(main())
