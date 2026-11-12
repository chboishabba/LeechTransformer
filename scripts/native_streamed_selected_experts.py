"""Native GPU selected-expert microbatch pipeline on authentic teacher/student checkpoints.

Scope: same-device selected expert execution with pinned-host CPU->GPU staging,
source-owned routing and selected histories. No WAN, remote peer, authentication,
autoregressive recurrent-history transport, or training claim.
Original Leech-LILA: Anatolii Kornienko. Uses existing source HistoryOrbitMoE.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from scripts.diagnose_moe_quality_collapse import load_pair
from scripts.train_signed_wave_experts import layer_inputs

def filehash(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda:f.read(1<<20),b""):
            h.update(part)
    return h.hexdigest()

def _check_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("native GPU unavailable: no CPU fallback")
    return torch.device("cuda:0")

def _pin(t):
    return t.detach().to("cpu",copy=True).contiguous().pin_memory()

def make_jobs(teacher,student,ids,*,layer,batch,device):
    """Prepare independent token-window row jobs; source owns routing.

    This fixture uses the first token's normalized activation of real rows.
    Each row has its own zero history. It must not be interpreted as a full
    recurrent temporal token trace or complete-model inference.
    """
    moe=student.blocks[layer].moe
    jobs=[]
    references=[]
    histories=[]
    torch.cuda.synchronize(device)
    prepare_start=time.perf_counter()
    with torch.no_grad():
        for start in range(0,len(ids),batch):
            rows=torch.as_tensor(ids[start:start+batch],dtype=torch.long,device=device)
            hidden=layer_inputs(teacher,rows,layer)[:,0,:].contiguous()
            original_history=hidden.new_zeros(
                hidden.shape[0],moe.num_experts,moe.d_model)
            reference,next_history,selected,_,weights=moe.step(
                hidden,original_history,train_sign=False)
            selected_history=original_history.gather(
                1,selected.unsqueeze(-1).expand(-1,-1,moe.d_model))
            selected_updated=next_history.gather(
                1,selected.unsqueeze(-1).expand(-1,-1,moe.d_model))
            jobs.append(tuple(_pin(t) for t in (
                hidden, selected, selected_history, weights)))
            references.append(reference.detach().clone())
            histories.append(selected_updated.detach().clone())
    torch.cuda.synchronize(device)
    return jobs,references,histories,time.perf_counter()-prepare_start

def _device_batch(items, device, stream):
    with torch.cuda.stream(stream):
        return tuple(t.to(device, non_blocking=True) for t in items)

def _execute(moe, batch, stream):
    with torch.cuda.stream(stream),torch.no_grad():
        return moe.execute_selected(
            batch[0],batch[1],batch[2],selected_weights=batch[3],
            train_sign=False)

def sequential(moe,jobs,device):
    """Every copy+selected-execute is a dependency barrier (negative control)."""
    torch.cuda.synchronize(device)
    transfer=torch.cuda.Stream(device=device)
    compute=torch.cuda.Stream(device=device)
    torch.cuda.reset_peak_memory_stats(device)
    started=time.perf_counter()
    outputs=[]
    for host in jobs:
        device_data=_device_batch(host,device,transfer)
        event=torch.cuda.Event()
        event.record(transfer)
        compute.wait_event(event)
        output=_execute(moe,device_data,compute)
        compute.synchronize()
        outputs.append(tuple(t.detach().clone() for t in output))
    torch.cuda.synchronize(device)
    elapsed=time.perf_counter()-started
    return outputs,elapsed,torch.cuda.max_memory_allocated(device),torch.cuda.max_memory_reserved(device)

def pipelined(moe,jobs,device,capacity):
    """Schedule copies and selected-expert GPU work on separate CUDA streams.

    Bounded in-flight jobs; same model and worker GPU. No CPU fallback.
    Input tensors are pinned; output readiness is event-gated. CUDA events
    establish stream dependencies; no race on mutable recurrent state.
    """
    if capacity<1:raise ValueError("queue capacity must be positive")
    torch.cuda.synchronize(device)
    transfer=torch.cuda.Stream(device=device)
    compute=torch.cuda.Stream(device=device)
    torch.cuda.reset_peak_memory_stats(device)
    started=time.perf_counter()
    pending=[]
    outputs=[]
    for host in jobs:
        data=_device_batch(host,device,transfer)
        ready=torch.cuda.Event()
        ready.record(transfer)
        compute.wait_event(ready)
        result=_execute(moe,data,compute)
        done=torch.cuda.Event()
        done.record(compute)
        pending.append((done,host,data,result))
        if len(pending)>=capacity:
            event,host_ref,device_ref,value=pending.pop(0)
            event.synchronize()
            outputs.append(tuple(t.detach().clone() for t in value))
    for event,host_ref,device_ref,value in pending:
        event.synchronize()
        outputs.append(tuple(t.detach().clone() for t in value))
    torch.cuda.synchronize(device)
    elapsed=time.perf_counter()-started
    return outputs,elapsed,torch.cuda.max_memory_allocated(device),torch.cuda.max_memory_reserved(device)

def compare(outputs, reference, histories, *, atol,rtol):
    if len(outputs)!=len(reference):
        raise AssertionError("missing streamed batches")
    max_abs=0.
    max_h=0.
    bad=0
    with torch.no_grad():
        for output,expected,expected_h in zip(outputs,reference,histories):
            actual,updated,signs=output
            if not torch.isfinite(actual).all() or not torch.isfinite(updated).all():
                bad+=1
                continue
            max_abs=max(max_abs,float((actual-expected).abs().max().item()))
            max_h=max(max_h,float((updated-expected_h).abs().max().item()))
            if not torch.allclose(actual,expected,atol=atol,rtol=rtol):
                bad+=1
            if not torch.allclose(updated,expected_h,atol=atol,rtol=rtol):
                bad+=1
    return {"max_output_abs":max_abs,"max_selected_history_abs":max_h,
            "failing_batches":bad,"allclose":bad==0,"atol":atol,"rtol":rtol}

def evaluate(args):
    device=_check_cuda()
    if args.queue_depth<1 or args.batch<1 or args.max_rows<2 or args.repeats<1:
        raise ValueError("invalid benchmark budget")
    for source in (args.teacher,args.student,args.token_ids):
        if not Path(source).is_file():
            raise FileNotFoundError(source)
    rows=np.load(args.token_ids,allow_pickle=False,mmap_mode="r")
    if rows.ndim!=2 or rows.shape[1]<2 or len(rows)<2:
        raise ValueError("real token rows must be a two-dimensional next-token dataset")
    if not np.issubdtype(rows.dtype,np.integer):
        raise ValueError("token IDs must be integer")
    subset=np.asarray(rows[:min(args.max_rows,len(rows))])
    if len(subset)<2:raise ValueError("at least two real token rows required")
    teacher,student=load_pair(args.teacher,args.student,args.config)
    teacher=teacher.to(device).eval()
    student=student.to(device).eval()
    if not 0<=args.layer<len(teacher.blocks):
        raise ValueError("invalid model layer")
    if np.any(subset<0) or np.any(subset>=teacher.cfg.vocab_size):
        raise ValueError("real token ID outside vocabulary")
    moe=student.blocks[args.layer].moe
    jobs,refs,histories,prepare_s=make_jobs(
        teacher,student,subset,layer=args.layer,batch=args.batch,device=device)
    if len(jobs)<2:raise ValueError("require at least two independent microbatches")
    sample_bytes=sum(sum(t.numel()*t.element_size() for t in job) for job in jobs)
    # Warm up both paths before measuring. Original source checkpoint unchanged.
    sequential(moe,jobs[:2],device)
    pipelined(moe,jobs[:2],device,min(args.queue_depth,2))
    repetitions=[]
    for i in range(args.repeats):
        # Alternate order to limit one-sided cache/thermal warmup bias.
        if i%2:
            modes=("pipeline","sequential")
        else:
            modes=("sequential","pipeline")
        current={}
        for mode in modes:
            if mode=="sequential":
                output,sec,alloc,reserved=sequential(moe,jobs,device)
            else:
                output,sec,alloc,reserved=pipelined(
                    moe,jobs,device,args.queue_depth)
            parity=compare(output,refs,histories,atol=args.atol,rtol=args.rtol)
            current[mode]={
                "seconds":sec,"microbatches":len(jobs),
                "microbatches_per_second":len(jobs)/sec,
                "rows_per_second":len(subset)/sec,
                "bytes_staged":sample_bytes,
                "staging_bytes_per_second":sample_bytes/sec,
                "pytorch_peak_allocated_bytes":alloc,
                "pytorch_peak_reserved_bytes":reserved,
                "parity":parity}
        repetitions.append(current)
    accepted=all(x[m]["parity"]["allclose"]
                 for x in repetitions for m in ("pipeline","sequential"))
    return {
        "schema":1,"experiment":"native_selected_expert_streaming_pcie",
        "scope":"real_checkpoint_source_routed_selected_expert_per_independent_row",
        "gpu":torch.cuda.get_device_name(0),"hip":torch.version.hip,
        "teacher_sha256":filehash(args.teacher),
        "student_sha256":filehash(args.student),
        "tokens_sha256":filehash(args.token_ids),
        "config_sha256":filehash(args.config) if args.config else None,
        "rows":len(subset),"input_sequence_length":int(subset.shape[1]),
        "layer":args.layer,"batch_rows":args.batch,
        "microbatches":len(jobs),"queue_depth":args.queue_depth,
        "prepare_source_routing_seconds":prepare_s,
        "preparation_excluded_from_throughput":True,
        "staged_bytes":sample_bytes,
        "repetitions":repetitions,
        "source_selected_execution_parity":accepted,
        "gpu_model_forward_and_expert_execution":True,
        "gpu_input_staging":True,
        "real_wan_traffic":False,
        "remote_worker_execution":False,
        "full_autoregressive_history_replay":False,
        "diffusion_model_execution":False,
        "end_to_end_model_tps":False,
        "release_admitted":False}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher",type=Path,required=True)
    parser.add_argument("--student",type=Path,required=True)
    parser.add_argument("--config",type=Path)
    parser.add_argument("--token-ids",type=Path,required=True)
    parser.add_argument("--layer",type=int,default=1)
    parser.add_argument("--batch",type=int,default=2)
    parser.add_argument("--max-rows",type=int,default=16)
    parser.add_argument("--queue-depth",type=int,default=2)
    parser.add_argument("--repeats",type=int,default=3)
    parser.add_argument("--atol",type=float,default=1e-3)
    parser.add_argument("--rtol",type=float,default=1e-3)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refuse to overwrite prior source execution receipt")
    result=evaluate(args)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))
    if not result["source_selected_execution_parity"]:
        raise SystemExit("selected-expert parity failed; receipt preserved")

if __name__=="__main__":
    main()
