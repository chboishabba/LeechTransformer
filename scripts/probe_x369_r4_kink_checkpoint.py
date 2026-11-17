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
import sys

# Support `python scripts/...py` from the original repository root.
# Import the original package, not a copied or substitute router.
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

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
    # JSON source accepts exact numeric *strings* for revision-controlled
    # tensor values. torch.tensor(["1",...]) cannot parse those strings.
    # Convert only finite numeric scalars, never booleans/null/containers.
    def number(v):
        if type(v) not in (str,int,float):
            raise ValueError("source tensor element is not numeric")
        try:
            x=float(v)
        except (ValueError,OverflowError) as exc:
            raise ValueError("invalid source tensor number") from exc
        if not math.isfinite(x):
            raise ValueError("source tensor element is not finite")
        return x
    def convert(v,depth):
        if depth==len(shape):
            return number(v)
        if not isinstance(v,list) or len(v)!=shape[depth]:
            raise ValueError("source tensor shape mismatch: "+str(shape))
        return [convert(t,depth+1) for t in v]
    t=torch.tensor(convert(raw,0),dtype=dtype,device="cpu")
    if tuple(t.shape)!=shape or not bool(torch.isfinite(t).all()):
        raise ValueError("source tensor dtype overflow/nonfinite: "+str(shape))
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


def strict_source_order_margin(logits, ids):
    """Require unique selected-slot order and strict top-k cutoff.

    The original torch.topk is not a stable semantic tie-break.
    Return source-observed minimum selected-slot/cutoff gaps per row.
    """
    if logits.ndim!=2 or ids.ndim!=2 or logits.shape[0]!=ids.shape[0]:
        raise ValueError("original topk logit/ID shape mismatch")
    margins=[]
    for row in range(logits.shape[0]):
        scores=logits[row]
        selected=ids[row].tolist()
        if not bool(torch.isfinite(scores).all()) or len(set(selected))!=len(selected):
            raise RuntimeError("invalid original source score/selected IDs")
        ordered=[float(scores[i].item()) for i in selected]
        excluded=[float(scores[j].item()) for j in range(scores.numel())
                  if j not in selected]
        gaps=[a-b for a,b in zip(ordered,ordered[1:])]
        if excluded:
            gaps.append(min(ordered)-max(excluded))
        if not gaps or min(gaps)<=0:
            raise RuntimeError("original source topk tie or order not certified")
        margins.append(min(gaps))
    return margins


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
            strict_gap=strict_source_order_margin(logits,ids)[0]
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
            counterfactual_nat=(math.floor(scale*counterfactual)
                                if counterfactual>=0 else None)
            stages.append({
                "load":float(t),
                "source_ordered_ids":ids[0].tolist(),
                "source_logits":logits[0].tolist(),
                "source_strict_topk_gap":strict_gap,
                "source_selected_weights":weights[0].tolist(),
                "source_selected_signs":signs[0].tolist(),
                "source_response":real,
                "source_nat_response":projected,
                "fixed_selected_ids":fixed[0].tolist(),
                "fixed_selected_weights":forced_weights[0].tolist(),
                "fixed_route_response":counterfactual,
                "fixed_route_nat_response":counterfactual_nat,
                "source_selected_execution_replay_error":source_replay_error,
                "source_selected_history_replay":True})
    nat=[s["source_nat_response"] for s in stages]
    shape=(all(z is not None for z in nat)
           and nat[0]<nat[1]<nat[2] and nat[3]<nat[2])
    fixed_y=[s["fixed_route_response"] for s in stages]
    fixed_shape=(fixed_y[0]<fixed_y[1]<fixed_y[2]
                 and fixed_y[3]<fixed_y[2])
    fixed_nat=[s["fixed_route_nat_response"] for s in stages]
    fixed_nat_shape=(all(v is not None for v in fixed_nat)
                     and fixed_nat[0]<fixed_nat[1]<fixed_nat[2]
                     and fixed_nat[3]<fixed_nat[2])
    # The intervention identifies a counterfactual difference for the
    # declared reset-state load ray only. It does NOT establish a fitted
    # causal mechanism or on-policy recurrence.
    return {
        "source_runner":"HistoryOrbitMoE.step/execute_selected",
        "load_mode":"independent-reset-from-same-history",
        "stages":stages,
        "nat_kink_observed":shape,
        "fixed_route_continuous_kink_observed":fixed_shape,
        "fixed_route_nat_kink_observed":fixed_nat_shape,
        "native_vs_fixed_route_nat_kink_difference":
             shape and not fixed_nat_shape,
        # Backwards-compatible report alias; a matched ablation result,
        # NOT a proof that learned suppression is solely caused by topk.
        "topk_replacement_isolated_from_fixed_route":
             shape and not fixed_shape,
        "fitted_topk_causal_mechanism_certified":False,
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
    # Original HistoryOrbitMoE.history_cap is configuration-owned and
    # NONPERSISTENT. A bit-identical model state_dict alone does not
    # reconstruct the original recurrent transition. Require an
    # explicit cap selection (null means original source disabled).
    if "history_cap" not in request:
        raise ValueError("explicit original history_cap (null or positive) required")
    cap=request["history_cap"]
    if cap is not None:
        if type(cap) not in (int,float) or not (
            math.isfinite(cap) and cap>0):
            raise ValueError("invalid original history_cap")
        model.history_cap=float(cap)

    load_values=request["load_levels"]
    if not isinstance(load_values,list) or len(load_values)!=4:
        raise ValueError("exactly four load levels required")
    loads=list(map(float,load_values))
    if not all(math.isfinite(t) for t in loads) or not all(
        a<b for a,b in zip(loads,loads[1:])):
        raise ValueError("strictly increasing finite load levels required")
    if request.get("dtype") not in ("float32","float64"):
        raise ValueError("explicit source dtype float32 or float64 required")
    dtype=torch.float64 if request["dtype"]=="float64" else torch.float32
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
    scale=r.get("scale")
    if type(scale) is not int or not 1<=scale<=10**9:
        raise ValueError("Nat scale must be an integer in [1,1e9]")
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
        "original_nonpersistent_history_cap":cap,
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
