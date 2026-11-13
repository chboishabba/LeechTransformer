"""Native-GPU multi-worker source-routed expert execution across TCP peers.

Independent miner/worker host endpoints are described in a JSON manifest.
Each worker has an authenticated, revision-checked session and *owns a
disjoint subset of expert IDs*. The coordinator owns router weights, all
inactive histories and the final reduction; peers return unweighted expert
vectors and selected updated histories. Bounded request window and
per-peer socket queues permit concurrency across separate GPU workers.

Scope: first normalized position of each real token window, independent
zero-initialized histories. This is NOT recurrent generation, durable
exactly-once service, a trained diffusion model, or an encrypted transport.
Use individually authenticated SSH tunnels for remote hosts.
"""
from __future__ import annotations
import argparse
from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
import socket
import statistics
import time

import numpy as np
import torch
from scripts.diagnose_moe_quality_collapse import load_pair
from scripts.train_signed_wave_experts import layer_inputs
from scripts.federated_authenticated_session import seal,SessionReceiver
from scripts.federated_joint_state_quotient import JointState,merge_selected_into_authoritative_state
from scripts.native_federated_selected_worker import (
    need_gpu,sha256_file,encode_float,read_shared_key,
    request_wire,encode_message,decode_message,recv_frame,send_frame,
    history_revision,session_id,check_session)

def read_workers(path,expert_count):
    doc=json.loads(Path(path).read_text())
    workers=doc.get("workers")
    if not isinstance(workers,list) or len(workers)<2:
        raise ValueError("require >=2 separately owned workers")
    seen=set()
    names=set()
    endpoints=set()
    sessions=set()
    key_files=set()
    for w in workers:
        if not isinstance(w,dict):raise ValueError("invalid worker")
        if set(("name","host","port","session","key_file","experts"))-w.keys():
            raise ValueError("missing worker configuration")
        if (not isinstance(w["name"],str) or not w["name"] or
            w["name"] in names or not isinstance(w["host"],str) or
            not w["host"] or type(w["port"])!=int or not 1<=w["port"]<=65535 or
            not isinstance(w["experts"],list) or not w["experts"] or
            any(type(e)!=int or e<0 or e>=expert_count for e in w["experts"]) or
            len(set(w["experts"]))!=len(w["experts"])):
            raise ValueError("invalid worker name, port or expert ownership")
        if (not isinstance(w["session"],str) or not w["session"] or
            not isinstance(w["key_file"],str) or not w["key_file"]):
            raise ValueError("worker session and key-file path required")
        endpoint=(w["host"],w["port"])
        if endpoint in endpoints or w["session"] in sessions or w["key_file"] in key_files:
            raise ValueError("worker endpoints, sessions and key files must be distinct")
        if seen.intersection(w["experts"]):
            raise ValueError("expert owned by multiple workers")
        seen.update(w["experts"])
        names.add(w["name"])
        endpoints.add(endpoint)
        sessions.add(w["session"])
        key_files.add(w["key_file"])
        check_session(w["session"])
    if seen!=set(range(expert_count)):
        raise ValueError("worker union must cover every original model expert")
    return workers

@dataclass
class Peer:
    name: str
    expert_ids: tuple[int,...]
    session: str
    key: bytes
    sock: socket.socket
    receiver: SessionReceiver
    sent: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    completed: int = 0
    worker_device: str | None = None
    worker_gpu_name: str | None = None
    worker_hip: str | None = None

    def dispatch(self,source,selected,*,model,layer,count):
        seq=self.sent
        joint=request_wire(source,selected,count=count,
            model_revision=model,
            history_revision_text=history_revision(seq).decode("ascii"))
        payload=encode_message({
            "kind":"selected-expert-request","sequence":seq,
            "layer":layer,"model":model,"selected":list(selected)},joint)
        packet=seal(self.key,session_id("request",self.session),
            seq,history_revision(seq),payload)
        self.bytes_sent+=send_frame(self.sock,packet)
        self.sent+=1
        return seq

    def acknowledge(self,expected_sequence):
        if self.receiver.next_sequence!=expected_sequence:
            raise ValueError("response commit sequence mismatch")
        self.receiver.commit(history_revision(expected_sequence+1))
        self.completed+=1

    def receive(self,*,expected_sequence,model,layer,selected,width):
        packet,n=recv_frame(self.sock)
        self.bytes_received+=n
        payload=self.receiver.verify(packet)
        meta,data=decode_message(payload)
        k=len(selected)
        if (meta.get("kind")!="selected-expert-response" or
            meta.get("model")!=model or meta.get("layer")!=layer or
            meta.get("sequence")!=expected_sequence or
            meta.get("selected")!=list(selected) or
            meta.get("width")!=width or len(data)!=(2*k*width+k)*4):
            raise ValueError("worker response has incorrect origin or payload")
        self.worker_device=meta.get("worker_device")
        self.worker_gpu_name=meta.get("worker_gpu_name")
        self.worker_hip=meta.get("worker_hip")
        # Delay receiver sequence advancement until the complete source
        # reduction and all selected-history receipts are accepted.
        return (np.frombuffer(data[:k*width*4],dtype="<f4").copy().reshape(k,width),
                np.frombuffer(data[k*width*4:2*k*width*4],dtype="<f4").copy().reshape(k,width),
                float(meta.get("execution_seconds",0.)))

def connect_workers(specs,timeout):
    peers=[]
    try:
        for w in specs:
            key=read_shared_key(w["key_file"])
            sock=socket.create_connection((w["host"],w["port"]),timeout=timeout)
            sock.settimeout(timeout)
            peers.append(Peer(w["name"],tuple(w["experts"]),w["session"],key,sock,
                SessionReceiver(key,session_id("response",w["session"]),
                                history_revision(0))))
    except Exception:
        for p in peers:p.sock.close()
        raise
    return peers

def prepare_source_row(teacher,moe,row,layer,device,*,verify_source):
    with torch.no_grad():
        tokens=torch.as_tensor(row[None,:],dtype=torch.long,device=device)
        activation=layer_inputs(teacher,tokens,layer)[:,0,:]
        history=activation.new_zeros(1,moe.num_experts,moe.d_model)
        if verify_source:
            output,next_h,ids,_,weights=moe.step(activation,history)
            source_output=output[0].detach().clone()
            expected_histories={
                int(e):next_h[0,int(e)].detach().clone()
                for e in ids[0].tolist()}
        else:
            ids,weights=moe.select_source_experts(activation,history)
            source_output=None
            expected_histories=None
        selected=ids[0].tolist()
        weight_map={int(e):weights[0,j].detach().clone()
                    for j,e in enumerate(selected)}
        full=JointState(encode_float(activation[0]),{
            e:encode_float(history[0,e]) for e in range(moe.num_experts)})
    return full,source_output,weight_map,expected_histories


def reduce_source_selected(weights,contributions,updates,*,d_model,device,dtype,
                           reference=None,expected=None,atol=1e-3,rtol=1e-3):
    """Exact source-owned weighted reduction, including unchecked rows.

    Reference None means *unchecked*, not a zero reference. Original source
    values and selected histories are compared only when a reference exists.
    """
    selected=set(weights)
    if not selected or selected!=set(contributions) or selected!=set(updates):
        raise AssertionError("selected expert coverage is incomplete")
    if (reference is None)!=(expected is None):
        raise ValueError("reference and expected histories must travel together")
    with torch.no_grad():
        merged=torch.zeros(d_model,device=device,dtype=dtype)
        for expert in sorted(selected):
            v=contributions[expert]
            h=updates[expert]
            if v.shape!=(d_model,) or h.shape!=(d_model,):
                raise ValueError("incorrect selected expert vector width")
            if not bool(torch.isfinite(v).all()) or not bool(torch.isfinite(h).all()):
                raise AssertionError("nonfinite selected result")
            merged=merged+v*weights[expert]
        if not bool(torch.isfinite(merged).all()):
            raise AssertionError("nonfinite reduced output")
        if reference is None:
            return merged,None,None,False
        if set(expected)!=selected:
            raise AssertionError("original selected history reference mismatch")
        error=float((merged-reference).abs().max())
        herror=max(float((updates[e]-expected[e]).abs().max()) for e in selected)
        if (not torch.allclose(merged,reference,atol=atol,rtol=rtol)
            or any(not torch.allclose(updates[e],expected[e],
                                     atol=atol,rtol=rtol) for e in selected)):
            raise AssertionError("source selected output/history parity failure")
        return merged,error,herror,True

def run(args):
    if args.device not in ("cuda","cuda:0"):raise ValueError("native GPU required")
    device=need_gpu()
    if args.window<1 or args.window>64 or args.max_rows<2:
        raise ValueError("invalid bounded row window")
    if args.output.exists():raise FileExistsError("refuse overwrite receipt")
    teacher,student=load_pair(args.teacher,args.student,args.config)
    teacher=teacher.to(device).eval()
    student=student.to(device).eval()
    if not 0<=args.layer<len(teacher.blocks):raise ValueError("invalid layer")
    moe=student.blocks[args.layer].moe
    specs=read_workers(args.workers,moe.num_experts)
    model_hash=sha256_file(args.student)
    rows=np.load(args.token_ids,allow_pickle=False,mmap_mode="r")
    if rows.ndim!=2 or len(rows)<2 or rows.shape[1]<2 or not np.issubdtype(rows.dtype,np.integer):
        raise ValueError("invalid real token rows")
    rows=np.asarray(rows[:min(args.max_rows,len(rows))])
    if np.any(rows<0) or np.any(rows>=teacher.cfg.vocab_size):
        raise ValueError("out of vocabulary IDs")
    peers=connect_workers(specs,args.timeout)
    peak_inflight=0
    tasks=deque()
    completed=0
    verified=0
    output_err=history_err=0.
    task_rtt=[]
    progress=[]
    started=time.perf_counter()
    next_row=0
    try:
        while next_row<len(rows) or tasks:
            while next_row<len(rows) and len(tasks)<args.window:
                check_original=(next_row%args.source_verify_every==0) if args.source_verify_every else False
                source,reference,weights,expected=prepare_source_row(
                    teacher,moe,rows[next_row],args.layer,device,
                    verify_source=check_original)
                dispatches=[]
                for peer in peers:
                    owned=tuple(sorted(set(peer.expert_ids).intersection(weights)))
                    if not owned:continue
                    payload=JointState(source.activation,{
                        e:source.histories[e] for e in owned})
                    sequence=peer.dispatch(payload,owned,model=model_hash,
                        layer=args.layer,count=moe.num_experts)
                    dispatches.append((peer,sequence,owned))
                if not dispatches:raise AssertionError("no owners for selected experts")
                tasks.append((next_row,time.perf_counter(),source,
                              reference,weights,expected,dispatches))
                next_row+=1
                peak_inflight=max(peak_inflight,len(tasks))
            if not tasks:break
            row_no,queued_at,source,reference,weights,expected,dispatches=tasks.popleft()
            contributions={}
            updates={}
            remote_times={}
            for peer,sequence,owned in dispatches:
                output,history,exec_s=peer.receive(
                    expected_sequence=sequence,model=model_hash,
                    layer=args.layer,selected=owned,width=moe.d_model)
                remote_times[peer.name]=exec_s
                for i,e in enumerate(owned):
                    if e in contributions:raise AssertionError("duplicate expert")
                    contributions[e]=torch.from_numpy(output[i]).to(device)
                    updates[e]=torch.from_numpy(history[i]).to(device)
            if set(contributions)!=set(weights):
                raise AssertionError("missing selected expert")
            merged_output,err,h_err,parity_checked=reduce_source_selected(
                weights,contributions,updates,d_model=moe.d_model,
                device=device,dtype=next(moe.parameters()).dtype,
                reference=reference,expected=expected,
                atol=args.atol,rtol=args.rtol)
            reopened=JointState(source.activation,{
                e:encode_float(updates[e]) for e in updates})
            full=merge_selected_into_authoritative_state(
                source,reopened,selected=tuple(sorted(updates)),
                expert_count=moe.num_experts,itemsize=4,block_width=24)
            if any(full.histories[e]!=source.histories[e]
                   for e in source.histories if e not in updates):
                raise AssertionError("inactive history modified")
            # Release the source window only after BOTH output and history
            # parity and inactive-history preservation are established.
            for peer,sequence,_ in dispatches:
                if (peer.worker_device!="cuda:0" or
                    not peer.worker_gpu_name or not peer.worker_hip):
                    raise RuntimeError("remote peer did not self-report native GPU selected execution")
                peer.acknowledge(sequence)
            if parity_checked:
                verified+=1
                output_err=max(output_err,err)
                history_err=max(history_err,h_err)
            completed+=1
            latency=time.perf_counter()-queued_at
            task_rtt.append(latency)
            progress.append({"row":row_no,"elapsed_seconds":latency,
                "reference_checked":parity_checked,
                "workers_used":[p.name for p,_,_ in dispatches],
                "worker_execution_seconds":remote_times})
            if completed%args.progress_every==0 or completed==len(rows):
                print(json.dumps({"rows_completed":completed,"total":len(rows),
                    "inflight":len(tasks),"peer_requests":{p.name:p.completed for p in peers},
                    "last_row_latency_seconds":latency,
                    "max_error":output_err,"max_history_error":history_err}),flush=True)
    finally:
        for p in peers:p.sock.close()
    elapsed=time.perf_counter()-started
    samples=sorted(task_rtt)
    def pct(x):
        index=(len(samples)-1)*x/100
        lo=int(index);hi=min(lo+1,len(samples)-1)
        return samples[lo]+(samples[hi]-samples[lo])*(index-lo)
    result={"schema":1,"kind":"native_gpu_multiworker_source_selected",
        "scope":"one_teacher_normalized_position_per_real_row_zero_initial_history",
        "source_teacher_sha256":sha256_file(args.teacher),
        "student_sha256":model_hash,
        "token_source_sha256":sha256_file(args.token_ids),
        "worker_ownership":[{"worker":p.name,"experts":list(p.expert_ids),
            "self_reported_device":p.worker_device,
            "self_reported_gpu_name":p.worker_gpu_name,
            "self_reported_hip":p.worker_hip,
            "requests":p.completed,"tx_bytes":p.bytes_sent,
            "rx_bytes":p.bytes_received} for p in peers],
        "workers":len(peers),"layer":args.layer,
        "input_sequence_length":int(rows.shape[1]),"rows_completed":completed,
        "rows_with_original_source_expert_parity_checked":verified,
        "source_verify_every":args.source_verify_every,
        "source_expert_compute_skipped_on_other_rows":verified<completed,
        "reported_max_error_applies_only_to_verified_rows":True,
        "max_rows_inflight":peak_inflight,"configured_window":args.window,
        "elapsed_seconds":elapsed,"completed_rows_per_second":completed/elapsed,
        "request_bytes":sum(p.bytes_sent for p in peers),
        "response_bytes":sum(p.bytes_received for p in peers),
        "p50_row_latency_seconds":pct(50),"p95_row_latency_seconds":pct(95),
        "p99_row_latency_seconds":pct(99),
        "max_output_abs_error_on_verified_rows":output_err if verified else None,
        "max_selected_history_abs_error_on_verified_rows":history_err if verified else None,
        "source_parity_unchecked_on_other_rows":True,
        "all_configured_workers_responded":all(p.completed>0 for p in peers),
        "physical_peer_device_independence_unverified":True,
        "per_row":progress,"gpu_coordinator":torch.cuda.get_device_name(0),
        "worker_device_identity_hmac_self_reported_not_hardware_attested":True,
        "hmac_authenticated":True,"transport_encrypted":False,
        "source_history_recurrent":False,"durable_commit":False,
        "diffusion_trained":False,"end_to_end_model_speedup_proven":False,
        "release_admitted":False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="per_row"},
                     indent=2,sort_keys=True))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers",type=Path,required=True)
    p.add_argument("--teacher",type=Path,required=True)
    p.add_argument("--student",type=Path,required=True)
    p.add_argument("--config",type=Path)
    p.add_argument("--token-ids",type=Path,required=True)
    p.add_argument("--layer",type=int,default=1)
    p.add_argument("--max-rows",type=int,default=256)
    p.add_argument("--window",type=int,default=8)
    p.add_argument("--source-verify-every",type=int,default=16,
        help="full local source expert parity every N rows, 0 for throughput-only")
    p.add_argument("--progress-every",type=int,default=16)
    p.add_argument("--timeout",type=float,default=30.)
    p.add_argument("--atol",type=float,default=1e-3)
    p.add_argument("--rtol",type=float,default=1e-3)
    p.add_argument("--device",choices=("cuda","cuda:0"),default="cuda")
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.progress_every<1 or args.timeout<=0 or args.source_verify_every<0:
        raise ValueError("invalid telemetry/reference settings")
    run(args)

if __name__=="__main__":
    main()
