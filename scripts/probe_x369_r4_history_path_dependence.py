#!/usr/bin/env python3
"""Original Leech R4 SAME-CLOCK history dependence, checkpoint-owning producer.

Two one-step prefixes from the SAME model and initial full E histories:
  A: [load_0, load_2]
  B: [load_3, load_2]
At the second source time step, x and clock agree. Only the updated
full histories differ. Run the ORIGINAL HistoryOrbitMoE.step every time.
This is a tested-source path-dependence *observation*, NOT universal
hysteresis, heldout task quality or a biological stress-response claim.

Input JSON is the original four-load checkpoint probe format.
SHA claims are computed from the actual supplied original byte files.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch

from models.history_orbit_moe import HistoryOrbitMoE
from scripts.probe_x369_r4_kink_checkpoint import (
    source_tensor, load_original, sha256)


def strict_source_order_margin(logits, ids):
    """Require unique ordered top-k and a strict selected/outside cut."""
    rows=logits.shape[0]
    if ids.shape[0]!=rows:
        raise ValueError("source topk row mismatch")
    margins=[]
    for row in range(rows):
        selected=ids[row].tolist()
        vals=logits[row]
        if not bool(torch.isfinite(vals).all()):
            raise RuntimeError("nonfinite source logits")
        ordered=[float(vals[i].item()) for i in selected]
        excluded=[float(vals[j].item()) for j in range(vals.numel())
                  if j not in selected]
        gaps=[a-b for a,b in zip(ordered,ordered[1:])]
        if excluded:
            gaps.append(min(ordered)-max(excluded))
        if not gaps or min(gaps)<=0:
            raise RuntimeError("original topk tie/ordering not strictly certified")
        margins.append(min(gaps))
    return margins

def path_evaluation(model,x0,H,levels,coordinate,scale):
    if len(levels)!=4 or not all(math.isfinite(v) for v in levels):
        raise ValueError("four finite ordered source loads required")
    if not all(a<b for a,b in zip(levels,levels[1:])):
        raise ValueError("source loads not strictly ascending")
    records=[]
    with torch.no_grad():
        for name,prefix in (("lower_prefix",levels[0]),
                            ("higher_prefix",levels[3])):
            initial=H.unsqueeze(0).clone()
            first_x=(x0*prefix).unsqueeze(0)
            first_route_ids,first_route_weights,first_logits=(
                model.select_source_experts(first_x,initial,return_logits=True))
            first_margin=strict_source_order_margin(first_logits,first_route_ids)
            first_output,after_first,first_ids,first_signs,first_weights=(
                model.step(first_x,initial,train_sign=False))
            if not torch.equal(first_route_ids,first_ids) or not torch.equal(
                first_route_weights,first_weights):
                raise RuntimeError("original prefix router/step disagreement")
            probe_x=(x0*levels[2]).unsqueeze(0)
            ordered_ids,ordered_weights,logits=model.select_source_experts(
                probe_x,after_first,return_logits=True)
            second_margin=strict_source_order_margin(logits,ordered_ids)
            second_output,after_second,ids,signs,weights=(
                model.step(probe_x,after_first,train_sign=False))
            if not torch.equal(ordered_ids,ids) or not torch.equal(
                ordered_weights,weights):
                raise RuntimeError("same-state native source route not stable")
            if coordinate=="l2":
                response=float(torch.linalg.vector_norm(second_output).item())
            else:
                response=float(second_output[0,coordinate].item())
            if not math.isfinite(response):
                raise RuntimeError("nonfinite source history response")
            records.append({
                "prefix_name":name,
                "prefix_load":prefix,
                "probe_load":levels[2],
                "probe_step_index":1,
                "first_step_selected_ids":first_ids[0].tolist(),
                "first_step_selected_signs":first_signs[0].tolist(),
                "probe_source_ids":ids[0].tolist(),
                "prefix_strict_topk_gap":first_margin[0],
                "probe_strict_topk_gap":second_margin[0],
                "probe_source_logits":logits[0].tolist(),
                "probe_source_weights":weights[0].tolist(),
                "probe_signs":signs[0].tolist(),
                "probe_response":response,
                "probe_nat_response":math.floor(scale*response)
                    if response>=0 else None,
                "first_step_full_histories_sha256":hashlib.sha256(
                    after_first.contiguous().numpy().tobytes()).hexdigest(),
                "second_step_full_histories_sha256":hashlib.sha256(
                    after_second.contiguous().numpy().tobytes()).hexdigest(),
                "_source_full_history_after_first":after_first.clone(),
            })
    first,second=records
    history_difference=float(torch.linalg.vector_norm(
        first["_source_full_history_after_first"]
        -second["_source_full_history_after_first"]).item())
    for record in records:
        record.pop("_source_full_history_after_first")
    response_difference=abs(first["probe_response"]-second["probe_response"])
    if not math.isfinite(history_difference):
        raise RuntimeError("nonfinite history difference")
    return {
       "source_runner":"HistoryOrbitMoE.step",
       "comparison":"two-step prefix intervention, same probe input/clock",
       "probe_step_index":1,
       "full_history_l2_distance_before_probe":history_difference,
       "same_load_and_clock_history_dependent_response_observed":
           response_difference>0 and history_difference>0,
       "same_load_probe_response_abs_difference":response_difference,
       "probe_nat_divergence_observed":(
           first["probe_nat_response"]!=second["probe_nat_response"]),
       "paths":records,
       "source_float_interval_certified":False,
       "general_hysteresis_certified":False,
       "trained_mechanism_certified":False,
       "biological_interpretation_certified":False}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint",required=True,type=Path)
    parser.add_argument("--input",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--state-prefix",default="")
    args=parser.parse_args()
    with args.input.open(encoding="utf8") as stream:
        request=json.load(stream)
    config=request["model"]
    if config.get("route_geometry") not in (
            "c2_c3_residual","c2_c3_mixed_null_residual","c2_c2_residual"):
        raise ValueError("original R4 residual geometry required")
    if request.get("dtype") not in ("float32","float64"):
        raise ValueError("explicit original source dtype required")
    dtype={"float32":torch.float32,"float64":torch.float64}[request["dtype"]]
    model=HistoryOrbitMoE(**config).to(dtype=dtype).eval()
    if "history_cap" not in request:
        raise ValueError("original source nonpersistent history_cap required")
    cap=request["history_cap"]
    if cap is not None:
        if type(cap) not in (int,float) or not math.isfinite(cap) or cap<=0:
            raise ValueError("invalid source history_cap")
        model.history_cap=float(cap)
    if not isinstance(request.get("history_revision"),str) or not request["history_revision"]:
        raise ValueError("original source history revision required")
    x0=source_tensor(request["base_x"],(model.d_model,),dtype)
    H=source_tensor(request["history"],(model.num_experts,model.d_model),dtype)
    levels=list(map(float,request["load_levels"]))
    response=request["response"]
    if response.get("type")=="l2":
        coordinate="l2"
    elif response.get("type")=="coordinate" and type(response.get("index")) is int and (
         0<=response["index"]<model.d_model):
        coordinate=response["index"]
    else:
        raise ValueError("unsupported original source response observable")
    scale=response.get("scale")
    if type(scale) is not int or not 1<=scale<=10**9:
        raise ValueError("invalid source Nat observation scale")
    load_original(args.checkpoint,model,args.state_prefix)
    out=path_evaluation(model,x0,H,levels,coordinate,scale)
    out.update({
       "original_checkpoint_sha256":sha256(args.checkpoint),
       "source_input_sha256":sha256(args.input),
       "history_revision":request["history_revision"],
       "original_history_values_sha256":hashlib.sha256(
           H.contiguous().numpy().tobytes()).hexdigest(),
       "source_nonpersistent_history_cap":cap,
       "model_configuration":config,
       "dtype":request["dtype"],
       "response_observable":response,
       "device":"cpu"})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("w",encoding="utf8") as stream:
        json.dump(out,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write("\n")
    print("X369_R4_NATIVE_HISTORY_PATH_PROBE",args.output,
          "same_clock_response_diff",
          out["same_load_probe_response_abs_difference"],
          "history_distance",
          out["full_history_l2_distance_before_probe"])


if __name__=="__main__":
    main()
