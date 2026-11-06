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
    """Only an actual GEMM kernel inside exactly one QKV marker is attributable."""
    label = "leech:qkv:sgemm-shim:" + arm
    intervals = []
    for p in sorted(root.rglob("*marker_api_trace.csv")):
        with p.open(newline="", errors="replace") as handle:
            for row in csv.DictReader(handle):
                if row.get("Function") != label:
                    continue
                try:
                    start, end = int(row["Start_Timestamp"]), int(row["End_Timestamp"])
                except (KeyError, ValueError):
                    continue
                if start < end:
                    intervals.append((start, end))
    intervals = sorted(set(intervals))
    kernels = []
    if len(intervals) == 1:
        low, high = intervals[0]
        for p in sorted(root.rglob("*kernel_trace.csv")):
            with p.open(newline="", errors="replace") as handle:
                for row in csv.DictReader(handle):
                    try:
                        start, end = int(row["Start_Timestamp"]), int(row["End_Timestamp"])
                    except (KeyError, ValueError):
                        continue
                    name = (row.get("Kernel_Name") or "").strip()
                    # No marker-only, profiler-renamed, memset or unrelated device
                    # symbol may impersonate the original rocBLAS or Schaka GEMM.
                    if not any(v in name for v in ("Cijk", "Tensile", "gfx803_sgemm_kernel")):
                        continue
                    if low <= start <= end <= high:
                        kernels.append({"name": name, "start": start, "end": end,
                                        "correlation_id": row.get("Correlation_Id"),
                                        "trace_csv": str(p.relative_to(root))})
    names = sorted({k["name"] for k in kernels})
    return {"status": "identified" if len(intervals) == 1 and len(names) == 1
            else "missing-or-ambiguous-profile-kernel",
            "marker": label, "intervals": intervals, "kernels": kernels,
            "unique_kernel_names": names}


def join(arm: str, unprofiled: dict, profiled: dict, matched: dict) -> dict:
    required = ("mode", "arm", "fixture_manifest_sha256", "teacher_sha256",
                "input_sha256", "weight_sha256", "rocblas_sha256",
                "rocblas_realpath", "shim_sha256", "shim_realpath",
                "required_image_root", "torch_hip", "gpu_arch", "gpu_device",
                "threshold")
    if arm not in ("on", "off"):
        raise ValueError("unknown shim A/B arm")
    for k in required:
        if k not in unprofiled or k not in profiled:
            raise ValueError("missing required source/runtime identity: " + k)
        if unprofiled[k] != profiled[k]:
            raise ValueError("profiled vs unprofiled mismatch: " + k)
    if unprofiled["arm"] != arm or unprofiled["mode"] != "shim":
        raise ValueError("wrong original shim arm or comparison mode")
    if (unprofiled.get("shim_actually_mapped") is not True or
            profiled.get("shim_actually_mapped") is not True or
            unprofiled.get("admission") != "denied" or
            profiled.get("admission") != "denied"):
        raise ValueError("unattested shim mapping or unexpected admission")
    before = unprofiled.get("output_sha256_by_repeat") or []
    after = profiled.get("output_sha256_by_repeat") or []
    n1 = unprofiled.get("repeats_completed")
    n2 = profiled.get("repeats_completed")
    if (not isinstance(n1, int) or n1 < 100 or
            n1 != unprofiled.get("repeats_requested") or n1 != len(before) or
            n2 != profiled.get("repeats_requested") or n2 != len(after) or n2 < 1):
        raise ValueError("shim comparison requires complete 100+ unprofiled repeats and complete profiled run")
    if (unprofiled.get("raw_first_output_sha256") != before[0] or
            profiled.get("raw_first_output_sha256") != after[0]):
        raise ValueError("first output SHA does not match preserved tensor bytes")
    if unprofiled.get("status") != profiled.get("status"):
        raise ValueError("profiler changes numerical classification")
    same = before[0] == after[0]
    names = matched.get("unique_kernel_names", [])
    return {
        "schema_version": 2,
        "purpose": "Schaka stock SGEMM shim versus original rocBLAS GPU comparison only",
        "arm": arm,
        "rocblas_sha256": unprofiled["rocblas_sha256"],
        "rocblas_realpath": unprofiled["rocblas_realpath"],
        "shim_sha256": unprofiled["shim_sha256"],
        "shim_realpath": unprofiled["shim_realpath"],
        "required_image_root": unprofiled["required_image_root"],
        "fixture_manifest_sha256": unprofiled["fixture_manifest_sha256"],
        "teacher_sha256": unprofiled["teacher_sha256"],
        "input_sha256": unprofiled["input_sha256"],
        "weight_sha256": unprofiled["weight_sha256"],
        "torch_hip": unprofiled["torch_hip"],
        "gpu_arch": unprofiled["gpu_arch"],
        "gpu_device": unprofiled["gpu_device"],
        "threshold": unprofiled["threshold"],
        "unprofiled_repeats_completed": n1,
        "original_unprofiled_output_sha256": before[0],
        "profiled_output_sha256": after[0],
        "profiler_observer_effect_absent": same,
        "unprofiled_status": unprofiled["status"],
        "profiled_status": profiled["status"],
        "trace": matched,
        "interposer_kernel_identified": bool(
            matched.get("status") == "identified" and
            len(names) == 1 and "gfx803_sgemm_kernel" in names[0]),
        "scope": "one shape, one pinned image, original QKV operands; not model-wide parity",
        "attribution_complete": bool(same and matched.get("status") == "identified"),
        "no_training_admission": True,
    }


def summarize(on: dict, off: dict) -> dict:
    required = ("rocblas_sha256", "rocblas_realpath",
                "shim_sha256", "shim_realpath", "required_image_root",
                "fixture_manifest_sha256", "teacher_sha256",
                "input_sha256", "weight_sha256", "torch_hip",
                "gpu_arch", "gpu_device", "threshold")
    same = all(k in on and k in off and on[k] == off[k] for k in required)
    complete = (on.get("arm") == "on" and off.get("arm") == "off" and
                on.get("no_training_admission") is True and
                off.get("no_training_admission") is True and
                isinstance(on.get("unprofiled_repeats_completed"), int) and
                isinstance(off.get("unprofiled_repeats_completed"), int) and
                on["unprofiled_repeats_completed"] >= 100 and
                off["unprofiled_repeats_completed"] >= 100 and
                on.get("attribution_complete") is True and
                off.get("attribution_complete") is True)
    # The ON arm must actually dispatch Schaka's published replacement kernel;
    # the OFF arm must not use that kernel. Different text labels alone are not
    # an independently established shim effect.
    named_interposer = (on.get("interposer_kernel_identified") is True and
                        off.get("interposer_kernel_identified") is False)
    effect = bool(same and complete and named_interposer and
                  on.get("unprofiled_status") == "correct-at-tested-shape" and
                  off.get("unprofiled_status") == "incorrect-at-tested-shape")
    return {
        "status": ("supports-scoped-interposer-effect" if effect
                   else "inconclusive-or-incomplete"),
        "on_status": on.get("unprofiled_status"),
        "off_status": off.get("unprofiled_status"),
        "same_library_shim_fixture_and_numeric_identity": same,
        "both_traces_complete_and_observer_effect_free": complete,
        "actual_interposer_kernel_discrimination": named_interposer,
        "shim_effect_at_original_QKV_shape": effect,
        "not_WGM_mapping_proof": True,
        "release_accepted": False, "gpu_training_admitted": False,
        "scope": "isolated exact-input Leech block-0 QKV only"
    }


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
