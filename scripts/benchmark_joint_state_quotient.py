"""Benchmark a WHOLE federated MoE state quotient against recorded captures.

NPZ with numpy numeric arrays (no pickle):
  activation: [N,d] float16/float32, little-endian
  histories:  [N,E,d] same dtype
  selected:   [N,k] integer IDs sorted and unique
Optionally:
  baseline_activation: [N,d], baseline_histories: [N,E,d]
  --ack-revision authenticated by the caller's session implementation.

This reconstructs actual sent bytes, does NOT assume independent router
packets, and models both 1/10 Gbit/s alongside measured local codec time.
No socket/WAN timing, encryption or trained model quality is claimed.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import struct
import time
import numpy as np
from scripts.federated_joint_state_quotient import (
    JointState,pack_joint_selected,unpack_joint_selected)


def _validate_capture(arrays):
    required={"activation","histories","selected"}
    if not required.issubset(arrays):
        raise ValueError("missing activation/histories/selected")
    x,h,s=(arrays[k] for k in ("activation","histories","selected"))
    if (x.ndim!=2 or h.ndim!=3 or s.ndim!=2 or
        x.shape[0]<1 or h.shape[0]!=x.shape[0] or
        s.shape[0]!=x.shape[0] or h.shape[2]!=x.shape[1] or
        s.shape[1]<1 or s.shape[1]>h.shape[1] or
        x.shape[1]%24):
        raise ValueError("invalid [N,d], [N,E,d], [N,k] shapes")
    if (x.dtype not in (np.dtype("<f4"),np.dtype("<f2")) or
        h.dtype!=x.dtype or not np.issubdtype(s.dtype,np.integer)):
        raise ValueError("only matching little-endian IEEE float16/32 captures")
    if "orbit_actions" in arrays:
        orbit=arrays["orbit_actions"]
        if orbit.shape!=(h.shape[1],) or orbit.dtype.kind not in ("U","S"):
            raise ValueError("source orbit actions must be one label per expert")
        from models.e8_orbit_actions import WEYL_ACTIONS
        if any(str(x) not in WEYL_ACTIONS for x in orbit):
            raise ValueError("unknown source E8 action")
    if "baseline_activation" in arrays or "baseline_histories" in arrays:
        if not {"baseline_activation","baseline_histories"}.issubset(arrays):
            raise ValueError("both complete baseline captures needed")
        if arrays["baseline_activation"].shape!=x.shape or (
           arrays["baseline_histories"].shape!=h.shape) or (
           arrays["baseline_activation"].dtype!=x.dtype) or (
           arrays["baseline_histories"].dtype!=x.dtype):
            raise ValueError("baseline mismatch")
    return x,h,s



def heldout_latency_policy(samples, *, rtt_ms:float, reply_bytes:int,
                           minimum_gain_ms:float=.05):
    """Select RAW or JQ1 on a prefix; report performance on later frames.

    This decides BEFORE encoding a future packet. It must not run both
    codecs and pretend the losing encoder had zero cost. The final slice
    is for observation only, never for changing the decision.
    """
    if minimum_gain_ms<0:
        raise ValueError("negative latency policy margin")
    if len(samples)<4:
        return {"status":"insufficient calibration frames",
                "raw_fallback":True,"minimum_samples":4}
    split=len(samples)//2
    calibration=samples[:split]
    validation=samples[split:]
    result={"status":"measured-local-codec/ideal-link, not WAN",
            "calibration_frames":split,
            "untouched_validation_frames":len(validation),
            "minimum_calibration_gain_ms":minimum_gain_ms,
            "links_gbps":{}}
    for rate in (1,10):
        def cost(record,mode):
            prefix="raw_" if mode=="raw" else ""
            wire=record[prefix+"wire_bytes"]
            return (rtt_ms + reply_bytes*8/(rate*1e6) +
                    wire*8/(rate*1e6) +
                    record[prefix+"encode_ms"] +
                    record[prefix+"decode_ms"])
        average=lambda rows,mode:sum(cost(row,mode) for row in rows)/len(rows)
        raw_cal=average(calibration,"raw")
        jq_cal=average(calibration,"quotient")
        # Margin prevents noise-scale size wins from triggering a costly
        # codec on a different history distribution.
        selected=("quotient" if raw_cal-jq_cal>minimum_gain_ms else "raw")
        raw_val=average(validation,"raw")
        jq_val=average(validation,"quotient")
        selected_val=raw_val if selected=="raw" else jq_val
        result["links_gbps"][str(rate)]={
            "chosen_before_encoding":selected,
            "raw_fallback":selected=="raw",
            "calibration_raw_full_ms":raw_cal,
            "calibration_quotient_full_ms":jq_cal,
            "calibration_gain_ms":raw_cal-jq_cal,
            "validation_raw_full_ms":raw_val,
            "validation_quotient_full_ms":jq_val,
            "validation_policy_full_ms":selected_val,
            "validation_policy_gain_ms":raw_val-selected_val,
            "validation_policy_wire_bytes":sum(
                row["raw_wire_bytes" if selected=="raw" else "wire_bytes"]
                for row in validation),
            "selection_scope":"single policy for this model/history layout and link; recalibrate after drift"}
    return result



def compression_admitted_for_link(receipt:dict, *,
                                  model_revision:str,
                                  history_revision:str,
                                  rate_gbps:int)->bool:
    """Choose before pack_joint_selected; never execute a loser encoder.

    The caller must bind these revisions to an authenticated session.
    Missing/under-sampled profiles fail closed to raw, but a mismatch
    in an explicitly supplied profile is an error (not a silent reuse).
    """
    if (not model_revision or not history_revision or
        rate_gbps not in (1,10)):
        raise ValueError("valid trusted revisions and supported link required")
    if (receipt.get("model_revision") != model_revision or
        receipt.get("history_revision") != history_revision):
        raise ValueError("transport profile belongs to a different revision")
    profile=receipt.get("heldout_latency_policy",{})
    if profile.get("status") == "insufficient calibration frames":
        return False
    choice=profile.get("links_gbps",{}).get(str(rate_gbps),{})
    if choice.get("chosen_before_encoding") not in ("raw","quotient"):
        raise ValueError("missing model-specific pre-encoding policy")
    return choice["chosen_before_encoding"]=="quotient"


def benchmark_capture(arrays, *, model_revision="model-v1",
                      history_revision="history-v1",
                      ack_revision=None,rtt_ms=5.,return_bytes=None,
                      history_placement="source",ack_bytes=64,
                      e8_orbit_actions:bool=False):
    x,h,s=_validate_capture(arrays)
    if ("baseline_activation" in arrays)!=(ack_revision is not None):
        raise ValueError("ACK revision required iff baseline provided")
    if (rtt_ms<0 or (return_bytes is not None and return_bytes<0) or
        history_placement not in ("source","peer") or ack_bytes<0):
        raise ValueError("invalid RTT, response size or history placement")
    count,experts,d=h.shape
    # Worker sends k UNWEIGHTED outputs. Source mixes them using its
    # retained weights. If histories are source-owned, k updated memories
    # must also return. Peer-owned histories need an ACK/revision receipt.
    response_bytes=(return_bytes if return_bytes is not None else
       s.shape[1]*d*x.dtype.itemsize*
       (2 if history_placement=="source" else 1)+
       (ack_bytes if history_placement=="peer" else 0))
    totals=Counter()
    codec_counts=Counter()
    max_error=0
    sample_rows=[]
    for i in range(count):
        selection=tuple(sorted(int(v) for v in s[i]))
        if selection!=tuple(int(v) for v in s[i]) or len(set(selection))!=len(selection):
            raise ValueError("selected IDs must be sorted and distinct")
        state=JointState(x[i].tobytes(),{int(e):h[i,int(e)].tobytes() for e in selection})
        baseline=None
        if ack_revision is not None:
            baseline=JointState(arrays["baseline_activation"][i].tobytes(),
              {int(e):arrays["baseline_histories"][i,int(e)].tobytes() for e in selection})
        options=dict(selected=selection,expert_count=experts,
            itemsize=x.dtype.itemsize,block_width=24,
            model_revision=model_revision,history_revision=history_revision)
        source_orbit=({e:str(arrays["orbit_actions"][e]) for e in selection}
                      if "orbit_actions" in arrays else None)
        raw_start=time.perf_counter_ns()
        raw_packet=pack_joint_selected(state,**options,
            acknowledged_baseline=baseline,acknowledged_revision=ack_revision,
            permit_compression=False)
        raw_encode_ns=time.perf_counter_ns()-raw_start
        raw_start=time.perf_counter_ns()
        raw_reopened=unpack_joint_selected(raw_packet,
            expected_model=model_revision,expected_history=history_revision,
            expected_selected=selection,acknowledged_baseline=baseline,
            expected_ack_revision=ack_revision)
        raw_decode_ns=time.perf_counter_ns()-raw_start
        if raw_reopened!=state:
            raise AssertionError("raw joint packet roundtrip failed")
        start=time.perf_counter_ns()
        packet=pack_joint_selected(state,**options,
            acknowledged_baseline=baseline,acknowledged_revision=ack_revision,
            use_e8_actions=e8_orbit_actions,
            orbit_actions=source_orbit)
        encode_ns=time.perf_counter_ns()-start
        start=time.perf_counter_ns()
        reopened=unpack_joint_selected(packet,
            expected_model=model_revision,expected_history=history_revision,
            expected_selected=selection,acknowledged_baseline=baseline,
            expected_ack_revision=ack_revision,
            expected_orbit_actions=source_orbit)
        decode_ns=time.perf_counter_ns()-start
        if reopened!=state:
            raise AssertionError("bytewise joint state roundtrip failed")
        hlen=struct.unpack("<I",packet[4:8])[0]
        codec=json.loads(packet[8:8+hlen])["codec"]
        sample_rows.append({"wire_bytes":len(packet),"raw_wire_bytes":len(raw_packet),
            "codec":codec,"encode_ms":encode_ns/1e6,"decode_ms":decode_ns/1e6,
            "raw_encode_ms":raw_encode_ns/1e6,"raw_decode_ms":raw_decode_ns/1e6})
        codec_counts[codec]+=1
        totals["independent_pairs_bytes"]+=len(state.activation)*2*experts
        totals["shared_all_histories_bytes"]+=len(state.activation)*(experts+1)
        totals["selected_complete_bytes"]+=len(state.activation)*(len(selection)+1)
        totals["actual_wire_bytes"]+=len(packet)
        totals["raw_framed_selected_bytes"]+=len(raw_packet)
        totals["encode_ns"]+=encode_ns
        totals["decode_ns"]+=decode_ns
    avg=lambda name:sum(z[name] for z in sample_rows)/count
    policy=heldout_latency_policy(sample_rows,rtt_ms=rtt_ms,
                                  reply_bytes=response_bytes)
    transfer={}
    for rate in (1,10):
        one_way=totals["actual_wire_bytes"]*8/(rate*1e9)*1000/count
        # Known return tensor size is separately supplied; do not assume
        # the reply has identical size or code it with the request codec.
        response=response_bytes*8/(rate*1e9)*1000
        raw_one_way=totals["raw_framed_selected_bytes"]*8/(rate*1e9)*1000/count
        quotient_total=rtt_ms+one_way+response+avg("encode_ms")+avg("decode_ms")
        raw_total=rtt_ms+raw_one_way+response+avg("raw_encode_ms")+avg("raw_decode_ms")
        transfer[str(rate)]={
            "serialization_send_ms_avg":one_way,
            "raw_serialization_send_ms_avg":raw_one_way,
            "serialization_return_ms_assumed":response,
            "estimated_full_remote_ms_avg":quotient_total,
            "estimated_raw_remote_ms_avg":raw_total,
            "faster_in_measured_codec_plus_ideal_link":
                ("quotient" if quotient_total<raw_total else "raw")}
    return {
        "source":"captured data (no synthetic fallback)",
        "model_revision":model_revision,
        "history_revision":history_revision,
        "heldout_latency_policy":policy,
        "samples":count,"d_model":d,"expert_count":experts,
        "selected_count":s.shape[1],"dtype":str(x.dtype),
        "acknowledged_baseline_supplied":ack_revision is not None,
        "e8_orbit_predictors_enabled":e8_orbit_actions,
        "trusted_shared_e8_orbit_declared":"orbit_actions" in arrays,
        "total_bytes":{k:v for k,v in totals.items() if k.endswith("_bytes")},
        "quotient_vs_selected_complete_ratio":
            (totals["selected_complete_bytes"]/totals["actual_wire_bytes"]),
        "codec_counts":dict(codec_counts),
        "observed_avg_encode_ms":avg("encode_ms"),
        "observed_avg_decode_ms":avg("decode_ms"),
        "observed_raw_encode_ms":avg("raw_encode_ms"),
        "observed_raw_decode_ms":avg("raw_decode_ms"),
        "modeled_link":transfer,"rtt_assumption_ms":rtt_ms,
        "reply_bytes_assumption":response_bytes,
        "reply_history_placement":history_placement,
        "reply_contains_unweighted_expert_outputs":True,
        "reply_includes_updated_histories":history_placement=="source",
        "peer_retained_history_requires_authenticated_ack":
            history_placement=="peer",
        "byte_exact_reconstruction":True,
        "replay_revision_checked":True,
        "not_proved":"No inference correctness, sender route legitimacy, encryption, cross-peer atomic commit or measured WAN result."}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--capture",required=True,type=Path)
    ap.add_argument("--model-revision",default="local-model")
    ap.add_argument("--history-revision",default="local-history")
    ap.add_argument("--ack-revision",help="must come from authenticated remote session")
    ap.add_argument("--rtt-ms",type=float,default=5.)
    ap.add_argument("--return-bytes",type=int,
                    help="explicit reply byte override; defaults to whole state-dependent path")
    ap.add_argument("--history-placement",choices=("source","peer"),default="source")
    ap.add_argument("--ack-bytes",type=int,default=64)
    ap.add_argument("--e8-orbit-actions",action="store_true",
                    help="enable extra E8 Weyl predictor search only when relevant")
    ap.add_argument("--output",default="joint_quotient_receipt.json",type=Path)
    args=ap.parse_args()
    with np.load(args.capture,allow_pickle=False) as f:
        capture={key:f[key] for key in f.files}
    result=benchmark_capture(capture,model_revision=args.model_revision,
        history_revision=args.history_revision,ack_revision=args.ack_revision,
        rtt_ms=args.rtt_ms,return_bytes=args.return_bytes,
        history_placement=args.history_placement,ack_bytes=args.ack_bytes,
        e8_orbit_actions=args.e8_orbit_actions)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
