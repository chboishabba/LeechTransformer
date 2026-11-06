"""Proof-scoped trace matching for a SAME-image Schaka SGEMM shim ON/OFF A/B.

Actual kernel names come from rocprofv3 and the selected ROCTX ranges, not
from Schaka patch filenames. Explicitly reject profiler observer effects.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def trace_kernels(root: Path, arm: str) -> dict:
    label = "leech:qkv:sgemm-shim:" + arm
    intervals=[]
    for p in sorted(root.rglob("*marker_api_trace.csv")):
        with p.open(newline="", errors="replace") as handle:
            for row in csv.DictReader(handle):
                if row.get("Function") != label:
                    continue
                start=int(row["Start_Timestamp"])
                end=int(row["End_Timestamp"])
                if start < end:
                    intervals.append([start,end])
    intervals=sorted({tuple(v) for v in intervals})
    kernels=[]
    if len(intervals)==1:
        low,high=intervals[0]
        for p in sorted(root.rglob("*kernel_trace.csv")):
            with p.open(newline="", errors="replace") as handle:
                for row in csv.DictReader(handle):
                    try:
                        start=int(row["Start_Timestamp"])
                        end=int(row["End_Timestamp"])
                    except (KeyError,ValueError):
                        continue
                    if low <= start <= end <= high:
                        kernels.append({"name":row.get("Kernel_Name",""),
                                        "start":start,"end":end,
                                        "correlation_id":row.get("Correlation_Id"),
                                        "trace_csv":str(p.relative_to(root))})
    useful=[k for k in kernels if k["name"] and not k["name"].startswith("__amd_rocclr_fill")]
    return {"status":"identified" if len(intervals)==1 and useful else "missing-profile-kernel",
            "marker":label, "intervals":intervals,"kernels":useful}


def join(arm: str, unprofiled: dict, profiled: dict, matched: dict) -> dict:
    required=("mode", "arm","fixture_manifest_sha256","rocblas_sha256",
              "shim_sha256","shim_realpath","rocblas_realpath","required_image_root")
    if arm not in ("on","off"):
        raise ValueError("unknown shim A/B arm")
    for k in required:
        if unprofiled.get(k)!=profiled.get(k):
            raise ValueError("profiled vs unprofiled mismatch: "+k)
    if unprofiled["arm"]!=arm or unprofiled["mode"]!="shim":
        raise ValueError("wrong original shim arm or comparison mode")
    before=unprofiled.get("output_sha256_by_repeat") or []
    after=profiled.get("output_sha256_by_repeat") or []
    if not before or not after:
        raise ValueError("missing original GPU output bytes hashes")
    same=before[0]==after[0]
    return {
        "schema_version":1,
        "purpose":"Schaka stock SGEMM shim versus original rocBLAS GPU comparison only",
        "arm":arm,
        "same_library_build":True,
        "rocblas_sha256":unprofiled["rocblas_sha256"],
        "shim_sha256":unprofiled["shim_sha256"],
        "fixture_manifest_sha256":unprofiled["fixture_manifest_sha256"],
        "original_unprofiled_output_sha256":before[0],
        "profiled_output_sha256":after[0],
        "profiler_observer_effect_absent":same,
        "unprofiled_status":unprofiled["status"],
        "profiled_status":profiled["status"],
        "trace":matched,
        "scope":"one shape, one pinned image, original QKV operands; not model-wide parity",
        "attribution_complete":bool(same and matched["status"]=="identified"),
        "no_training_admission":True,
    }


def summarize(on: dict, off: dict) -> dict:
    required=("rocblas_sha256","shim_sha256","fixture_manifest_sha256")
    same=all(on.get(k)==off.get(k) for k in required)
    have_trace=bool(on.get("attribution_complete") and off.get("attribution_complete"))
    effect=bool(same and have_trace and
                on.get("unprofiled_status")=="correct-at-tested-shape" and
                off.get("unprofiled_status")=="incorrect-at-tested-shape")
    return {"status":"supports-scoped-interposer-effect" if effect else "inconclusive-or-incomplete",
            "on_status":on.get("unprofiled_status"),
            "off_status":off.get("unprofiled_status"),
            "same_library_shim_fixture_hashes":same,
            "both_traces_observer_effect_free":have_trace,
            "shim_effect_at_original_QKV_shape":effect,
            "not_WGM_mapping_proof":True,
            "release_accepted":False,"gpu_training_admitted":False,
            "scope":"isolated exact-input Leech block-0 QKV only"}


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir",type=Path,required=True)
    args=p.parse_args()
    out=args.out_dir
    joined={}
    for arm in ("on","off"):
        one=out/arm
        if not (one/"profiled.json").is_file() or not (one/"unprofiled.json").is_file():
            print("Missing profile or unprofiled data for "+arm)
            continue
        try:
            original=json.loads((one/"unprofiled.json").read_text())
            profiled=json.loads((one/"profiled.json").read_text())
            trace=trace_kernels(one/"profiler",arm)
            report=join(arm,original,profiled,trace)
        except (OSError,ValueError,KeyError) as exc:
            print("No "+arm+" kernel attribution: "+str(exc))
            continue
        (one/"trace_join.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
        joined[arm]=report
    if len(joined)!=2:
        summary={"status":"inconclusive-or-incomplete","completed_arms":list(joined),
                 "gpu_training_admitted":False, "release_accepted":False}
    else:
        summary=summarize(joined["on"],joined["off"])
    (out/"shim_ab_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    print(json.dumps(summary,sort_keys=True))
    return 0 if summary["status"]=="supports-scoped-interposer-effect" else 4


if __name__=="__main__":
    raise SystemExit(main())
