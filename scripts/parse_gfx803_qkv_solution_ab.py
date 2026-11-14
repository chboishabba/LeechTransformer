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
    """Bind a traced solution to one *complete*, byte-attested unprofiled run.

    The profiled run uses one dispatch while the unprofiled experiment must
    cover at least 100 repeats. A first-output match by itself is not evidence
    that the unprofiled run completed or had the claimed tolerance.
    """
    required = ("solution_id", "rocblas_sha256", "rocblas_realpath",
                "fixture_manifest_sha256", "teacher_sha256",
                "input_sha256", "weight_sha256", "threshold", "gemm",
                "gpu_arch", "torch_hip")
    for key in required:
        if key not in unprofiled or key not in profiled:
            raise ValueError("missing required same-run evidence: " + key)
        if unprofiled[key] != profiled[key]:
            raise ValueError("profiled/unprofiled solution or same-library/fixture identity mismatch: " + key)
    if not isinstance(unprofiled["solution_id"], int):
        raise ValueError("missing integer solution identity")
    if any(r.get("status") not in ("correct-at-tested-shape",
                                    "incorrect-at-tested-shape")
           for r in (unprofiled, profiled)):
        raise ValueError("cannot attribute an inapplicable solution")
    hashes1 = unprofiled.get("output_sha256_by_repeat") or []
    hashes2 = profiled.get("output_sha256_by_repeat") or []
    n1, n2 = unprofiled.get("repeats_completed"), profiled.get("repeats_completed")
    if (not isinstance(n1, int) or n1 < 3 or
            n1 != unprofiled.get("repeats_requested") or
            n1 != len(hashes1) or
            n2 != 1 or n2 != profiled.get("repeats_requested") or
            len(hashes2) != 1):
        raise ValueError("missing complete 100+ repeat baseline or single-dispatch profile")
    for receipt, hashes in ((unprofiled, hashes1), (profiled, hashes2)):
        if receipt.get("raw_first_output_sha256") != hashes[0]:
            raise ValueError("first GPU output bytes do not match retained raw output")
        if receipt.get("admission") != "denied":
            raise ValueError("unexpected training admission in diagnostic receipt")
    if unprofiled["status"] != profiled["status"]:
        raise ValueError("profiling changes tested-shape numerical classification")
    stable = hashes1[0] == hashes2[0]
    return {
        "schema_version": 2,
        "solution_id": unprofiled["solution_id"],
        "rocblas_realpath": unprofiled["rocblas_realpath"],
        "rocblas_sha256": unprofiled["rocblas_sha256"],
        "fixture_manifest_sha256": unprofiled["fixture_manifest_sha256"],
        "teacher_sha256": unprofiled["teacher_sha256"],
        "input_sha256": unprofiled["input_sha256"],
        "weight_sha256": unprofiled["weight_sha256"],
        "threshold": unprofiled["threshold"],
        "gemm": unprofiled["gemm"],
        "gpu_arch": unprofiled["gpu_arch"],
        "torch_hip": unprofiled["torch_hip"],
        "unprofiled_repeats_completed": n1,
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
    """A same-build contrast is not a model repair or an exhaustive scan."""
    by_mapping = {}
    original = "Cijk_Alik_Bljk_SB_MT64x128x16_SN_AMAS3_BL1_BS1_EPS1_GLVWA4_GLVWB4_GRVWn1_GSU1_GSUASB_ISA803_K1_KLA_LPA4_LPB4_LRVW4_MMFGLC_NLCA1_NLCB1_PGR1_PLR1_SIA1_SU32_SUS256_SVW4_TT4_8_USFGROn1_VAW1_VSn1_VW4_VWB4_WG16_16_1_WGM8"
    if not joins:
        return {"status": "no-measured-solutions", "repair_accepted": False,
                "no_training_admission": True}
    identity_fields = ("rocblas_sha256", "rocblas_realpath", "fixture_manifest_sha256",
                       "teacher_sha256", "input_sha256", "weight_sha256",
                       "threshold", "gemm", "gpu_arch", "torch_hip")
    same = {key: all(key in j for j in joins) and
            len({json.dumps(j[key], sort_keys=True) for j in joins}) == 1
            for key in identity_fields}
    valid = all(j.get("attribution_complete") is True and
                j.get("no_training_admission") is True and
                isinstance(j.get("unprofiled_repeats_completed"), int) and
                j["unprofiled_repeats_completed"] >= 3
                for j in joins)
    unique_ids = len({j.get("solution_id") for j in joins}) == len(joins)
    for item in joins:
        trace = item.get("trace", {})
        if not item.get("attribution_complete"):
            continue
        wgm = trace.get("wgm")
        if wgm not in (1, 8) or not trace.get("unique_kernel_names"):
            continue
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
    observed = all(same.values()) and valid and unique_ids and wgm8_fails and wgm1_passes
    return {"status": ("supports-scoped-mapping-hypothesis" if observed
                       else "causal-discrimination-incomplete"),
            "one_library_build": same["rocblas_sha256"],
            "one_fixture": same["fixture_manifest_sha256"],
            "same_run_identity": same, "all_receipts_complete": valid,
            "solution_ids_unique": unique_ids,
            "original_sgemm_kernel_exactly_reproduced_by_gemm_ex": original_exact,
            "wgm8_wrong_and_wgm1_correct": observed,
            "mapping_groups": by_mapping,
            "repair_accepted": False,
            "caveat": ("Forced rocblas_gemm_ex can select a different code path from the "
                       "original rocblas_sgemm. WGM1 success is scoped evidence, "
                       "not proof of an in-place WGM patch or complete-model repair."),
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
