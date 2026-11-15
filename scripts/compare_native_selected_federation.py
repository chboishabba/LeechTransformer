"""Compare exact same-work local resident and multiworker native GPU receipts.

Refuses comparison unless checkpoint/token/layer/row count and source-sampling
policy match. Results are observational, not a universal deployment speedup.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

def compare(local,remote):
    if (local.get("kind")!="native_gpu_local_resident_selected_expert_baseline"
        or remote.get("kind")!="native_gpu_multiworker_source_selected"):
        raise ValueError("incorrect native selected-expert receipt types")
    keys=(("teacher_sha256","source_teacher_sha256"),
          ("student_sha256","student_sha256"),
          ("token_source_sha256","token_source_sha256"),
          ("rows_completed","rows_completed"),
          ("layer","layer"),
          ("source_verify_every","source_verify_every"))
    for lk,rk in keys:
        if local.get(lk) is None or remote.get(rk) is None or local[lk]!=remote[rk]:
            raise ValueError("not the same source work: "+lk)
    if not (remote.get("workers",0)>=2 and remote.get("all_configured_workers_responded")
            and remote.get("hmac_authenticated") is True):
        raise ValueError("remote federation did not complete authenticated two-peer workload")
    if (remote.get("physical_peer_device_independence_unverified") is not True or
        local.get("physical_multiworker_execution") is not False):
        raise ValueError("receipt source declaration mismatch")
    if (local.get("rows_with_original_source_expert_parity_checked") !=
        remote.get("rows_with_original_source_expert_parity_checked")):
        raise ValueError("reference-check workload must match")
    if local.get("rows_with_original_source_expert_parity_checked",0)<1:
        raise ValueError("no verified source rows")
    for side in (local,remote):
        if side.get("max_output_abs_error_on_verified_rows") is None:
            raise ValueError("no source parity evidence")
        if not side.get("completed_rows_per_second") or side["completed_rows_per_second"]<=0:
            raise ValueError("missing throughput")
        if not side.get("elapsed_seconds") or side["elapsed_seconds"]<=0:
            raise ValueError("missing wall-clock observation")
    ratio=remote["completed_rows_per_second"]/local["completed_rows_per_second"]
    return {
        "schema":1,"comparison":"real_source_matched_selected_expert_local_vs_multiworker",
        "original_teacher_sha256":local["teacher_sha256"],
        "original_student_sha256":local["student_sha256"],
        "original_tokens_sha256":local["token_source_sha256"],
        "layer":local["layer"],"rows":local["rows_completed"],
        "source_verify_every":local["source_verify_every"],
        "source_verified_rows":local["rows_with_original_source_expert_parity_checked"],
        "local_rows_per_second":local["completed_rows_per_second"],
        "remote_rows_per_second":remote["completed_rows_per_second"],
        "observed_remote_over_local_throughput_ratio":ratio,
        "observed_remote_slower":ratio<1,
        "local_p95_seconds":local.get("p95_row_latency_seconds"),
        "remote_p95_seconds":remote.get("p95_row_latency_seconds"),
        "remote_request_bytes":remote.get("request_bytes"),
        "remote_response_bytes":remote.get("response_bytes"),
        "native_gpu_experiment":True,
        "hardware_independent_workers_attested":False,
        "transport_encrypted":False,
        "same_physical_host_or_network_condition_guaranteed":False,
        "autoregressive_full_token_throughput_proven":False,
        "diffusion_trained":False,
        "release_admitted":False
    }

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--local",type=Path,required=True)
    p.add_argument("--remote",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError("refuse to overwrite comparison evidence")
    report=compare(json.loads(a.local.read_text()),
                   json.loads(a.remote.read_text()))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report,indent=2,sort_keys=True))
if __name__=="__main__":
    main()
