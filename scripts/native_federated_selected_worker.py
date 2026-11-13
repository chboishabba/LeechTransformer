"""Persistent authenticated selected-expert service for real Leech MoE checkpoints.

Actual TCP client/server and genuine native-GPU expert execution on worker.
HMAC authenticates frames but DOES NOT encrypt: use only trusted networks or
a separately authenticated encrypted tunnel. No durable exactly-once commit.
A source keeps all E histories; each selected-only request carries one row,
one model/history revision and the exact selected expert list. Worker never
re-routes or receives inactive histories. Source applies mixture weights.

Use a separate physical GPU host for the worker and coordinator whenever
possible. The same-card two-process configuration is a functional diagnostic
and is not a federation throughput demonstration.
Original Leech-LILA architecture: Anatolii Kornienko.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import socket
import struct
import statistics
import time

import numpy as np
import torch
from models.model import LeechGPT
from scripts.federated_authenticated_session import seal, SessionReceiver
from scripts.federated_joint_state_quotient import (
    JointState,pack_joint_selected,unpack_joint_selected,
    merge_selected_into_authoritative_state)
from scripts.verify_trained_federated_moe import checkpoint_parts,infer_config
from scripts.diagnose_moe_quality_collapse import load_pair
from scripts.train_signed_wave_experts import layer_inputs

MAX_FRAME=64*1024*1024
MAX_HEAD=4096
MAGIC=b"FXW1"

def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda:f.read(1<<20),b""):h.update(part)
    return h.hexdigest()

def need_gpu():
    if not torch.cuda.is_available():raise RuntimeError("HIP GPU required; no CPU fallback")
    return torch.device("cuda:0")

def encode_float(t):
    if t.dtype!=torch.float32:raise ValueError("wire requires fp32 source")
    return t.detach().contiguous().cpu().numpy().astype("<f4",copy=False).tobytes()

def decode_float(raw,width,device):
    if len(raw)!=width*4:raise ValueError("incorrect fp32 component width")
    return torch.from_numpy(np.frombuffer(raw,dtype="<f4").copy()).to(device)

def encode_message(meta,payload):
    h=json.dumps(meta,sort_keys=True,separators=(",",":")).encode()
    if len(h)>MAX_HEAD:raise ValueError("header too large")
    return MAGIC+struct.pack(">I",len(h))+h+payload

def decode_message(payload):
    if len(payload)<8 or payload[:4]!=MAGIC:raise ValueError("bad message header")
    n=struct.unpack(">I",payload[4:8])[0]
    if n>MAX_HEAD or len(payload)<8+n:raise ValueError("invalid message bounds")
    m=json.loads(payload[8:8+n])
    if not isinstance(m,dict):raise ValueError("nonobject metadata")
    return m,payload[8+n:]

def recv_exact(sock,n):
    chunks=[]
    while n:
        part=sock.recv(min(n,1<<20))
        if not part:raise EOFError("peer closed in frame")
        chunks.append(part);n-=len(part)
    return b"".join(chunks)

def recv_frame(sock):
    raw=recv_exact(sock,4)
    size=struct.unpack(">I",raw)[0]
    if not 0<size<=MAX_FRAME:raise ValueError("invalid bounded frame")
    return recv_exact(sock,size),size+4

def send_frame(sock,wire):
    if not 0<len(wire)<=MAX_FRAME:raise ValueError("outgoing frame exceeds bound")
    sock.sendall(struct.pack(">I",len(wire))+wire)
    return 4+len(wire)

def history_revision(sequence):
    return ("hist-%016x" % sequence).encode("ascii")

def session_id(prefix,session):
    return (prefix+"-"+session).encode("ascii")

def check_session(session):
    if not session or len(session)>60 or not all(
        c.isascii() and (c.isalnum() or c in "-_") for c in session):
        raise ValueError("session must be safe nonempty ASCII identifier")
    return session

def read_shared_key(path):
    # Use a secret file provisioned out of band; never transmit its bytes.
    p=Path(path)
    if not p.is_file():raise FileNotFoundError(p)
    if os.name=="posix" and p.stat().st_mode & 0o077:
        raise PermissionError("shared key file must not be readable by group or others")
    key=p.read_bytes()
    if len(key)<32:raise ValueError("shared key requires >=32 raw bytes")
    return key

def load_student(student_path,device):
    state,info=checkpoint_parts(student_path)
    cfg=infer_config(state,info,None)
    if cfg.moe_experts<2:raise ValueError("selected-expert worker requires full MoE student")
    model=LeechGPT(cfg).eval()
    model.load_state_dict(state,strict=True)
    # Only the selected-execution module is placed on worker GPU in serve().\n    # This avoids keeping attention/embeddings and other layers resident there.\n    return model

def request_wire(state,selected,*,count,model_revision,history_revision_text):
    return pack_joint_selected(
        state,selected=selected,expert_count=count,itemsize=4,block_width=24,
        model_revision=model_revision,history_revision=history_revision_text,
        permit_compression=False,use_site_actions=False,use_e8_actions=False)

def response_from_worker(moe,joint,selected,device):
    width=moe.d_model
    x=decode_float(joint.activation,width,device).view(1,width)
    h=torch.stack([decode_float(joint.histories[e],width,device)
                   for e in selected],dim=0).view(1,len(selected),width)
    ids=torch.tensor([selected],dtype=torch.long,device=device)
    with torch.no_grad():
        unweighted,updated,signs=moe.execute_selected_subset(x,ids,h)
    if device.type=="cuda":
        torch.cuda.synchronize(device)
    return (b"".join(encode_float(unweighted[0,j]) for j in range(len(selected)))
           +b"".join(encode_float(updated[0,j]) for j in range(len(selected)))
           +encode_float(signs[0])),updated

def worker_connection(sock,*,moe,key,model_rev,session,layer,max_requests,
                      device=None,owned_experts=None):
    receiver=SessionReceiver(key,session_id("request",session),history_revision(0))
    owned=set(range(moe.num_experts)) if owned_experts is None else set(owned_experts)
    if device is None:device=next(moe.orientation.parameters()).device
    count=0
    try:
        while count<max_requests:
            try: packet,wire_bytes=recv_frame(sock)
            except EOFError: break
            opened=receiver.verify(packet)
            meta,payload=decode_message(opened)
            seq=receiver.next_sequence
            selected=meta.get("selected")
            if (meta.get("kind")!="selected-expert-request" or
                meta.get("sequence")!=seq or meta.get("layer")!=layer or
                meta.get("model")!=model_rev or
                not isinstance(selected,list) or not selected or
                any(type(e)!=int for e in selected) or
                selected!=sorted(set(selected)) or
                any(e<0 or e>=moe.num_experts or e not in owned for e in selected) or
                len(selected)>moe.top_k):
                raise ValueError("unauthorised or invalid source-selected request")
            selected=tuple(selected)
            joint=unpack_joint_selected(payload,expected_model=model_rev,
                expected_history=history_revision(seq).decode("ascii"),
                expected_selected=selected)
            t0=time.perf_counter()
            data,updated=response_from_worker(moe,joint,selected,device)
            execution=time.perf_counter()-t0
            response=encode_message({
                "kind":"selected-expert-response","sequence":seq,
                "model":model_rev,"layer":layer,"selected":list(selected),
                "width":moe.d_model,"execution_seconds":execution,
                "worker_device":str(device),
                "worker_gpu_name":torch.cuda.get_device_name(device) if device.type=="cuda" else None,
                "worker_hip":torch.version.hip if device.type=="cuda" else None,
                "request_framed_bytes":wire_bytes},data)
            packed=seal(key,session_id("response",session),seq,
                        history_revision(seq),response)
            send_frame(sock,packed)
            receiver.commit(history_revision(seq+1))
            count+=1
    finally:
        sock.close()
    return count

def serve(args):
    device=need_gpu()
    key=read_shared_key(args.key_file)
    model=load_student(args.student,device)
    if args.layer<0 or args.layer>=len(model.blocks):
        raise ValueError("invalid layer")
    moe=model.blocks[args.layer].moe
    model_rev=sha256_file(args.student)
    if args.host not in ("127.0.0.1","::1","localhost") and not args.allow_remote:
        raise ValueError("non-loopback TCP requires explicit --allow-remote and an encrypted trusted tunnel")
    session=check_session(args.session)
    family=socket.AF_INET6 if ":" in args.host else socket.AF_INET
    with socket.socket(family,socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        server.bind((args.host,args.port))
        server.listen(1)
        print(json.dumps({"state":"worker_listening","host":args.host,
            "port":server.getsockname()[1],"model_sha256":model_rev,
            "layer":args.layer,"device":torch.cuda.get_device_name(0),
            "max_requests":args.max_requests,"owned_experts":sorted(owned),
            "transport_encrypted":False}),flush=True)
        conn,addr=server.accept()
        conn.settimeout(args.timeout)
        executed=worker_connection(conn,moe=moe,key=key,model_rev=model_rev,
            session=session,layer=args.layer,max_requests=args.max_requests,
            device=device,owned_experts=owned)
        print(json.dumps({"state":"worker_session_closed","completed":executed}),flush=True)

def coordinator(args):
    device=need_gpu()
    key=read_shared_key(args.key_file)
    session=check_session(args.session)
    teacher,student=load_pair(args.teacher,args.student,args.config)
    teacher=teacher.to(device).eval()
    student=student.to(device).eval()
    if args.layer<0 or args.layer>=len(teacher.blocks):
        raise ValueError("invalid source layer")
    moe=student.blocks[args.layer].moe
    model_rev=sha256_file(args.student)
    rows=np.load(args.token_ids,allow_pickle=False,mmap_mode="r")
    if (rows.ndim!=2 or len(rows)<2 or rows.shape[1]<2 or
        not np.issubdtype(rows.dtype,np.integer)):
        raise ValueError("invalid original tokenizer rows")
    rows=np.asarray(rows[:min(args.max_rows,len(rows))])
    if np.any(rows<0) or np.any(rows>=teacher.cfg.vocab_size):
        raise ValueError("out-of-vocabulary source rows")
    if args.output.exists():raise FileExistsError("refuse to overwrite evidence")
    receiver=SessionReceiver(key,session_id("response",session),history_revision(0))
    max_err=max_h_err=0.
    bytes_sent=bytes_received=0
    perf=[]
    full_start=time.perf_counter()
    with socket.create_connection((args.host,args.port),timeout=args.timeout) as sock:
        sock.settimeout(args.timeout)
        pending=[]
        seq=0
        while seq<len(rows) or pending:
            # Independent zero-history rows may be staged before their replies.
            # This is a bounded network window, not an asynchronous recurrence.
            while seq<len(rows) and len(pending)<args.window:
                row=rows[seq]
                with torch.no_grad():
                    ids=torch.as_tensor(row[None,:],device=device,dtype=torch.long)
                    x=layer_inputs(teacher,ids,args.layer)[:,0,:]
                    full_h=x.new_zeros(1,moe.num_experts,moe.d_model)
                    reference,next_h,selected,signs,weights=moe.step(x,full_h)
                    ordered=torch.argsort(selected[0])
                    sorted_ids=selected[0][ordered].tolist()
                    sorted_weights=weights[0][ordered].detach().clone()
                    old_selected=full_h[0,selected[0][ordered]]
                    expected_history=next_h[0,selected[0][ordered]].detach().clone()
                    state=JointState(encode_float(x[0]),{
                        e:encode_float(old_selected[j]) for j,e in enumerate(sorted_ids)})
                    source_full=JointState(state.activation,{
                        e:encode_float(full_h[0,e]) for e in range(moe.num_experts)})
                joint=request_wire(state,tuple(sorted_ids),count=moe.num_experts,
                    model_revision=model_rev,
                    history_revision_text=history_revision(seq).decode("ascii"))
                payload=encode_message({
                    "kind":"selected-expert-request","sequence":seq,
                    "layer":args.layer,"model":model_rev,
                    "selected":sorted_ids},joint)
                wire=seal(key,session_id("request",session),seq,history_revision(seq),payload)
                send_start=time.perf_counter()
                bytes_sent+=send_frame(sock,wire)
                pending.append((seq,send_start,sorted_ids,sorted_weights,
                                reference[0].detach().clone(),expected_history,
                                source_full))
                seq+=1
            if not pending:break
            (item_seq,send_start,sorted_ids,sorted_weights,
             expected,expected_history,source_full)=pending.pop(0)
            response_wire,received=recv_frame(sock)
            bytes_received+=received
            decrypted=receiver.verify(response_wire)
            meta,data=decode_message(decrypted)
            width=moe.d_model
            k=len(sorted_ids)
            if (meta.get("kind")!="selected-expert-response" or
                meta.get("sequence")!=item_seq or meta.get("model")!=model_rev or
                meta.get("layer")!=args.layer or meta.get("selected")!=sorted_ids or
                meta.get("width")!=width or len(data)!=(2*k*width+k)*4):
                raise ValueError("response identities or dimensions differ")
            remote=torch.from_numpy(
                np.frombuffer(data[:k*width*4],dtype="<f4").copy()
            ).to(device).view(k,width)
            updated=torch.from_numpy(
                np.frombuffer(data[k*width*4:2*k*width*4],dtype="<f4").copy()
            ).to(device).view(k,width)
            torch.cuda.synchronize(device)
            elapsed=time.perf_counter()-send_start
            combined=(remote*sorted_weights[:,None]).sum(dim=0)
            err=float((combined-expected).abs().max())
            hist_err=float((updated-expected_history).abs().max())
            if not torch.allclose(combined,expected,atol=args.atol,rtol=args.rtol) or not torch.allclose(
                    updated,expected_history,atol=args.atol,rtol=args.rtol):
                raise AssertionError("selected-source worker mismatch at request %d" % item_seq)
            # The authoritative source reopens only selected histories.
            reopened=JointState(source_full.activation,{
                e:encode_float(updated[j]) for j,e in enumerate(sorted_ids)})
            merged=merge_selected_into_authoritative_state(
                source_full,reopened,selected=tuple(sorted_ids),
                expert_count=moe.num_experts,itemsize=4,block_width=24)
            if any(merged.histories[e]!=source_full.histories[e]
                   for e in range(moe.num_experts) if e not in sorted_ids):
                raise AssertionError("inactive source history mutated")
            max_err=max(max_err,err)
            max_h_err=max(max_h_err,hist_err)
            receiver.commit(history_revision(item_seq+1))
            perf.append({"sequence":item_seq,"round_trip_seconds":elapsed,
                "worker_execution_seconds":meta.get("execution_seconds"),
                "selected_ids":sorted_ids})
            if (len(perf))%args.progress_every==0 or len(perf)==len(rows):
                print(json.dumps({"completed":len(perf),"total":len(rows),
                    "last_round_trip_seconds":elapsed,
                    "in_flight":len(pending),
                    "max_output_error":max_err,
                    "max_selected_history_error":max_h_err}),flush=True)
    duration=time.perf_counter()-full_start
    latency_samples=sorted(p["round_trip_seconds"] for p in perf)
    def pct(values,percent):
        if not values:return None
        index=(len(values)-1)*percent/100
        lower=int(index)
        upper=min(lower+1,len(values)-1)
        return values[lower]+(values[upper]-values[lower])*(index-lower)
    report={
        "schema":1,"kind":"authenticated_native_gpu_selected_expert_tcp",
        "scope":"one_position_per_real_row_independent_zero_histories",
        "original_teacher_sha256":sha256_file(args.teacher),
        "original_student_sha256":model_rev,
        "token_array_sha256":sha256_file(args.token_ids),
        "rows_completed":len(perf),"elapsed_seconds":duration,
        "max_outstanding_window":args.window,
        "p50_request_response_seconds":pct(latency_samples,50),
        "p95_request_response_seconds":pct(latency_samples,95),
        "p99_request_response_seconds":pct(latency_samples,99),
        "average_request_response_seconds":statistics.mean(latency_samples),
        "request_latency_includes_queue_wait":True,
        "elapsed_includes_source_teacher_and_routing":True,
        "completed_rows_per_second":len(perf)/duration,
        "framed_request_bytes":bytes_sent,
        "framed_response_bytes":bytes_received,
        "max_output_abs_error":max_err,
        "max_selected_history_abs_error":max_h_err,
        "per_request":perf,
        "coordinator_gpu":torch.cuda.get_device_name(0),
        "hmac_authenticated":True,"transport_encrypted":False,
        "worker_gpu_native_requested":True,
        "source_router_weights_preserved":True,
        "recurrent_history_streamed":False,
        "durable_exactly_once_commit":False,
        "multi_client_queued_work":False,
        "diffusion_model_executed":False,
        "federation_throughput_vs_local_baseline_proven":False,
        "release_admitted":False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="per_request"},indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=("serve","coordinator"),required=True)
    p.add_argument("--host",default="127.0.0.1")
    p.add_argument("--port",type=int,required=True)
    p.add_argument("--session",required=True)
    p.add_argument("--key-file",type=Path,required=True)
    p.add_argument("--teacher",type=Path)
    p.add_argument("--student",type=Path,required=True)
    p.add_argument("--config",type=Path)
    p.add_argument("--token-ids",type=Path)
    p.add_argument("--layer",type=int,default=1)
    p.add_argument("--max-rows",type=int,default=64)
    p.add_argument("--max-requests",type=int,default=256)
    p.add_argument("--owned-experts",help="comma-separated worker-owned expert IDs (default all)")
    p.add_argument("--window",type=int,default=4,
        help="bounded number of outstanding independent selected-expert rows")
    p.add_argument("--progress-every",type=int,default=8)
    p.add_argument("--atol",type=float,default=1e-3)
    p.add_argument("--rtol",type=float,default=1e-3)
    p.add_argument("--timeout",type=float,default=30.)
    p.add_argument("--allow-remote",action="store_true")
    p.add_argument("--output",type=Path)
    p.add_argument("--device",choices=("cuda","cuda:0"),default="cuda")
    args=p.parse_args()
    if args.device not in ("cuda","cuda:0") or args.port<0 or args.port>65535:
        raise ValueError("invalid GPU device or port")
    if args.max_rows<2 or args.progress_every<1 or args.max_requests<1 or args.window<1 or args.window>64 or args.timeout<=0:
        raise ValueError("invalid bounded session arguments")
    if args.mode=="serve":
        serve(args)
    else:
        if not all((args.teacher,args.token_ids,args.output)):
            p.error("coordinator requires --teacher --token-ids --output")
        coordinator(args)

if __name__=="__main__":
    main()
