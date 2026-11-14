"""Matched local native-GPU reference for real multiworker selected expert execution.

Exactly the same source teacher first-position activations, original router and
sampled source verification policy as native_multiworker_coordinator, but all
selected experts execute on the coordinator's GPU with no TCP or WAN. This is
the local-resident comparator, NOT full autoregressive token throughput.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch
from scripts.diagnose_moe_quality_collapse import load_pair
from scripts.native_federated_selected_worker import need_gpu,sha256_file,decode_float
from scripts.native_multiworker_coordinator import (
    prepare_source_row,reduce_source_selected)

def run(args):
    if args.device not in ("cuda","cuda:0"):
        raise ValueError("native GPU only")
    device=need_gpu()
    if args.max_rows<2 or args.source_verify_every<0 or args.progress_every<1:
        raise ValueError("invalid row or verification settings")
    if args.output.exists():
        raise FileExistsError("refuse overwrite local selected-expert receipt")
    teacher,student=load_pair(args.teacher,args.student,args.config)
    teacher=teacher.to(device).eval()
    student=student.to(device).eval()
    if args.layer<0 or args.layer>=len(teacher.blocks):
        raise ValueError("invalid source layer")
    moe=student.blocks[args.layer].moe
    rows=np.load(args.token_ids,allow_pickle=False,mmap_mode="r")
    if (rows.ndim!=2 or len(rows)<2 or rows.shape[1]<2
        or not np.issubdtype(rows.dtype,np.integer)):
        raise ValueError("invalid original token rows")
    rows=np.asarray(rows[:min(args.max_rows,len(rows))])
    if np.any(rows<0) or np.any(rows>=teacher.cfg.vocab_size):
        raise ValueError("token ID outside checkpoint vocabulary")
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started=time.perf_counter()
    times=[]
    parity_count=0
    max_output_error=max_history_error=0.
    for rowno,row in enumerate(rows):
        row_start=time.perf_counter()
        verify=(rowno % args.source_verify_every == 0) if args.source_verify_every else False
        source,reference,weights,expected=prepare_source_row(
            teacher,moe,row,args.layer,device,verify_source=verify)
        selected=sorted(weights)
        if len(selected)!=moe.top_k:
            raise AssertionError("original route cardinality mismatch")
        x=decode_float(source.activation,moe.d_model,device).view(1,moe.d_model)
        histories=torch.stack([decode_float(source.histories[e],moe.d_model,device)
                               for e in selected],dim=0).unsqueeze(0)
        ids=torch.tensor([selected],device=device,dtype=torch.long)
        with torch.no_grad():
            unweighted,updated,_=moe.execute_selected_subset(x,ids,histories)
            contribution={e:unweighted[0,j] for j,e in enumerate(selected)}
            changes={e:updated[0,j] for j,e in enumerate(selected)}
            _,err,herr,checked=reduce_source_selected(
                weights,contribution,changes,d_model=moe.d_model,
                device=device,dtype=x.dtype,reference=reference,
                expected=expected,atol=args.atol,rtol=args.rtol)
        torch.cuda.synchronize(device)
        if checked:
            parity_count+=1
            max_output_error=max(max_output_error,err)
            max_history_error=max(max_history_error,herr)
        times.append(time.perf_counter()-row_start)
        if (rowno+1)%args.progress_every==0 or rowno+1==len(rows):
            elapsed=time.perf_counter()-started
            print(json.dumps({
                "rows_completed":rowno+1,"total":len(rows),
                "completed_rows_per_second":(rowno+1)/elapsed,
                "source_verified_rows":parity_count}),flush=True)
    elapsed=time.perf_counter()-started
    samples=sorted(times)
    def pct(p):
        i=(len(samples)-1)*p/100
        lo=int(i);hi=min(lo+1,len(samples)-1)
        return samples[lo]+(samples[hi]-samples[lo])*(i-lo)
    report={
        "schema":1,"kind":"native_gpu_local_resident_selected_expert_baseline",
        "scope":"same_first_normalized_position_per_row_zero_history_as_multiworker",
        "teacher_sha256":sha256_file(args.teacher),
        "student_sha256":sha256_file(args.student),
        "token_source_sha256":sha256_file(args.token_ids),
        "layer":args.layer,"rows_completed":len(rows),
        "source_verify_every":args.source_verify_every,
        "rows_with_original_source_expert_parity_checked":parity_count,
        "max_output_abs_error_on_verified_rows":max_output_error if parity_count else None,
        "max_selected_history_abs_error_on_verified_rows":max_history_error if parity_count else None,
        "elapsed_seconds":elapsed,
        "completed_rows_per_second":len(rows)/elapsed,
        "p50_row_latency_seconds":pct(50),"p95_row_latency_seconds":pct(95),
        "p99_row_latency_seconds":pct(99),
        "gpu":torch.cuda.get_device_name(0),"hip":torch.version.hip,
        "pytorch_peak_allocated_bytes":torch.cuda.max_memory_allocated(device),
        "pytorch_peak_reserved_bytes":torch.cuda.max_memory_reserved(device),
        "local_resident_experts":True,
        "peer_traffic_bytes":0,
        "physical_multiworker_execution":False,
        "network_transport_or_codec_overhead_included":False,
        "source_teacher_normalization_and_sampling_included":True,
        "source_parity_unchecked_on_other_rows":True,
        "durable_history":False,"diffusion_trained":False,
        "end_to_end_completed_token_speedup_proven":False,
        "release_admitted":False
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report,indent=2,sort_keys=True))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher",type=Path,required=True)
    p.add_argument("--student",type=Path,required=True)
    p.add_argument("--config",type=Path)
    p.add_argument("--token-ids",type=Path,required=True)
    p.add_argument("--layer",type=int,default=1)
    p.add_argument("--max-rows",type=int,default=256)
    p.add_argument("--source-verify-every",type=int,default=16)
    p.add_argument("--progress-every",type=int,default=16)
    p.add_argument("--atol",type=float,default=1e-3)
    p.add_argument("--rtol",type=float,default=1e-3)
    p.add_argument("--device",choices=("cuda","cuda:0"),default="cuda")
    p.add_argument("--output",type=Path,required=True)
    run(p.parse_args())
if __name__=="__main__":
    main()
