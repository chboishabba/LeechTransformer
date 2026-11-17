#!/usr/bin/env python3
"""Original Leech R4 native SAME-CHECKPOINT 4-load kink source producer.

This script runs the original HistoryOrbitMoE.select_source_experts,
HistoryOrbitMoE.step and HistoryOrbitMoE.execute_selected methods.
It does not reimplement the router, train a model, establish floating-point
error bounds, or certify a checkpoint's semantic quality. No GPU required.

Usage:
 python scripts/probe_x369_r4_kink_checkpoint.py \
   --checkpoint /path/to/owned.pt --input /path/to/load_input.json \
   --output /path/to/result.json [--state-prefix layers.0.experts.]

Input JSON:
 {"model":{"d_model":384,"num_experts":4,"top_k":2,
            "route_geometry":"c2_c3_residual"},
  "base_x":["0.2",...],
  "history":[["0.0",...],...],
  "load_levels":["0","1","2","3"],
  "response":{"type":"coordinate","index":0,"scale":10},
  "history_revision":"the-authoritative-revision"}

Inputs are checked BEFORE loading. A model state_dict or a dict with
state_dict/model_state_dict is accepted (weights_only=True); strict key
matching and an optional explicit prefix ensure same-object ownership.
The output hashes the actual checkpoint bytes and input bytes. It is
a measurement, NOT an interval certificate.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from models.history_orbit_moe import HistoryOrbitMoE


def sha256(path):
    digest=hashlib.sha256()
    with open(path,"rb") as stream:
        while True:
            chunk=stream.read(1024*1024)
            if not chunk:break
            digest.update(chunk)
    return digest.hexdigest()


def source_tensor(raw,shape,dtype):
    if not isinstance(raw,list):
        raise ValueError("source tensor is not a list")
    t=torch.tensor(raw,dtype=dtype,device="cpu")
    if tuple(t.shape)!=shape or not bool(torch.isfinite(t).all()):
        raise ValueError("source tensor shape or finiteness mismatch: "+str(shape))
    return t


def load_original(checkpoint,model,prefix):
    # weights_only forbids arbitrary pickle execution on untrusted checkpoints.
    data=torch.load(str(checkpoint),map_location="cpu",weights_only=True)
    if isinstance(data,dict) and "model_state_dict" in data:
        data=data["model_state_dict"]
    elif isinstance(data,dict) and "state_dict" in data:
        data=data["state_dict"]
    if not isinstance(data,dict) or not all(isinstance(k,str) for k in data):
        raise ValueError("expected source model state_dict")
    if prefix:
        data={k[len(prefix):]:v for k,v in data.items() if k.startswith(prefix)}
    if not data or not all(isinstance(v,torch.Tensor) for v in data.values()):
        raise ValueError("no source-owned module weights at selected prefix")
    model.load_state_dict(data,strict=True)


def gather_history(history,ids):
    return history.gather(
        1,ids.unsqueeze(-1).expand(-1,-1,history.shape[-1]))


def original_probe(model,base_x,initial_history,loads,coordinate,scale):
    # This is a RESET, four-state activation-load intervention.
    # It is not an on-policy time series, a trained response theorem,
    # or a mechanism proved without the matched counterfactual.
    fixed=None
    stages=[]
    with torch.no_grad():
        for t in loads:
            x=(base_x*t).unsqueeze(0)
            h=initial_history.unsqueeze(0).clone()
            ids,weights,logits=model.select_source_experts(
                x,h,return_logits=True)
            source_y,next_h,actual_ids,signs,actual_weights=(
                model.step(x,h,train_sign=False))
            if not torch.equal(ids,actual_ids) or not torch.equal(weights,actual_weights):
                raise RuntimeError("native source route changed between evaluations")
            selected_h=gather_history(h,ids)
            replay,updated,replay_signs=model.execute_selected(
                x,ids,selected_h,selected_weights=weights,train_sign=False)
            if not torch.equal(signs,replay_signs):
                raise RuntimeError("source sign chart mismatch")
            # The original implementation uses the same update and scalar
            # accumulation order in both step() and execute_selected().
            # Do not assume bitwise equality across all kernels/architectures.
            source_replay_error=float(torch.max(torch.abs(replay-source_y)).item())
            if source_replay_error>1e-5:
                raise RuntimeError("original selected-execution replay mismatch")
            for slot,expert in enumerate(ids[0].tolist()):
                if not torch.allclose(
                   updated[0,slot],next_h[0,expert],rtol=1e-6,atol=1e-6):
                    raise RuntimeError("original selected-history replay mismatch")
            if fixed is None:
                fixed=ids.detach().clone()
            forced_weights=torch.softmax(logits.gather(1,fixed),dim=-1)
            forced_h=gather_history(h,fixed)
            forced_y,forced_updated,forced_signs=model.execute_selected(
                x,fixed,forced_h,selected_weights=forced_weights,
                train_sign=False)
            def response(tensor):
                if coordinate=="l2":return float(torch.linalg.vector_norm(tensor).item())
                return float(tensor[0,int(coordinate)].item())
            real=response(source_y)
            counterfactual=response(forced_y)
            if not math.isfinite(real) or not math.isfinite(counterfactual):
                raise RuntimeError("nonfinite original source response")
            projected=math.floor(scale*real) if real>=0 else None
            stages.append({
                "load":float(t),
                "source_ordered_ids":ids[0].tolist(),
                "source_logits":logits[0].tolist(),
                "source_selected_weights":weights[0].tolist(),
                "source_selected_signs":signs[0].tolist(),
                "source_response":real,
                "source_nat_response":projected,
                "fixed_selected_ids":fixed[0].tolist(),
                "fixed_selected_weights":forced_weights[0].tolist(),
                "fixed_route_response":counterfactual,
                "source_selected_execution_replay_error":source_replay_error,
                "source_selected_history_replay":True})
    nat=[s["source_nat_response"] for s in stages]
    shape=(all(z is not None for z in nat)
           and nat[0]<nat[1]<nat[2] and nat[3]<nat[2])
    fixed_y=[s["fixed_route_response"] for s in stages]
    fixed_shape=(fixed_y[0]<fixed_y[1]<fixed_y[2]
                 and fixed_y[3]<fixed_y[2])
    return {
        "source_runner":"HistoryOrbitMoE.step/execute_selected",
        "load_mode":"independent-reset-from-same-history",
        "stages":stages,
        "nat_kink_observed":shape,
        "fixed_route_continuous_kink_observed":fixed_shape,
        "topk_replacement_isolated_from_fixed_route":
             shape and not fixed_shape,
        "certified_float_error_intervals":False,
        "trained_heldout_quality_certified":False,
        "recurrent_hysteresis_certified":False}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint",required=True,type=Path)
    parser.add_argument("--input",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--state-prefix",default="")
    args=parser.parse_args()
    with args.input.open(encoding="utf8") as stream:
        request=json.load(stream)
    raw_model=request["model"]
    allowed={"d_model","num_experts","top_k","expert_width","temperature",
             "route_geometry","transition_rank","router_selection"}
    if set(raw_model)-allowed:raise ValueError("unknown model configuration")
    if raw_model.get("route_geometry") not in (
        "c2_c3_residual","c2_c3_mixed_null_residual","c2_c2_residual"):
        raise ValueError("the early R4 test needs an original R4 residual arm")
    model=HistoryOrbitMoE(**raw_model).eval()
    load_values=request["load_levels"]
    if not isinstance(load_values,list) or len(load_values)!=4:
        raise ValueError("exactly four load levels required")
    loads=list(map(float,load_values))
    if not all(math.isfinite(t) for t in loads) or not all(
        a<b for a,b in zip(loads,loads[1:])):
        raise ValueError("strictly increasing finite load levels required")
    dtype=torch.float64 if request.get("dtype")=="float64" else torch.float32
    base_x=source_tensor(request["base_x"],(model.d_model,),dtype)
    H=source_tensor(request["history"],
       (model.num_experts,model.d_model),dtype)
    revision=request.get("history_revision")
    if not isinstance(revision,str) or not revision:
        raise ValueError("authoritative history revision required")
    r=request["response"]
    if r["type"]=="l2":coord="l2"
    elif r["type"]=="coordinate" and type(r["index"]) is int and (
        0<=r["index"]<model.d_model):coord=r["index"]
    else:raise ValueError("response observable must be l2 or valid coordinate")
    scale=int(r["scale"])
    if not 1<=scale<=10**9:raise ValueError("invalid Nat scale")
    model=model.to(dtype=dtype)
    load_original(args.checkpoint,model,args.state_prefix)
    out=original_probe(model,base_x,H,loads,coord,scale)
    out.update({
        "original_checkpoint_sha256":sha256(args.checkpoint),
        "source_input_sha256":sha256(args.input),
        "initial_history_revision":revision,
        "original_history_values_sha256":hashlib.sha256(
            H.contiguous().numpy().tobytes()).hexdigest(),
        "model_configuration":raw_model,
        "response_observable":r,
        "dtype":str(dtype),
        "device":"cpu"})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("w",encoding="utf8") as stream:
        json.dump(out,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write("\n")
    print("X369_R4_NATIVE_SOURCE_PROBE",str(args.output),
          "kink_observed",out["nat_kink_observed"],
          "isolated_from_fixed_route",out["topk_replacement_isolated_from_fixed_route"])


if __name__=="__main__":main()
