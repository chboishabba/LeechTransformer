"""Cost boundary for the 384=16*3*(9-1) router projection.

Use decimal link rates. Measurements here are arithmetic alone.
Do not call a router-only quotient a full MoE activation codec:
the selected expert still needs full activation and historical state
unless an independent consumer-relative reopen theorem is supplied.
"""
from __future__ import annotations
import argparse
import json


def compare(d_model=384,experts=4,top_k=2,gbps=(1,10)):
    if d_model%24 or experts<2 or not 1<=top_k<=experts:
        raise ValueError("invalid punctured-nine routing architecture")
    scalar_bytes=4
    raw_pair=2*d_model*scalar_bytes
    compact_pair=(3*d_model//2)*scalar_bytes
    raw_all_unshared=experts*raw_pair
    raw_all_shared=(experts+1)*d_model*scalar_bytes
    projected_all=experts*compact_pair
    selected_payload=(top_k+1)*d_model*scalar_bytes
    result={
        "d_model":d_model,"experts":experts,"top_k":top_k,
        "one_expert_raw_pair_bytes":raw_pair,
        "one_expert_router_only_quotient_bytes":compact_pair,
        "projected_reduction_fraction_vs_unshared_pair":
            (raw_pair-compact_pair)/raw_pair,
        "all_experts_unshared_raw_bytes":raw_all_unshared,
        "all_experts_shared_raw_bytes":raw_all_shared,
        "all_experts_router_only_features_bytes":projected_all,
        "selected_activation_and_histories_bytes":selected_payload,
        "complete_model_compute_available_from_router_quotient":False,
        "link_serialization_ms":{
            str(rate):{
                "shared_full_inputs":raw_all_shared*8/(rate*1e9)*1e3,
                "all_router_quotients":projected_all*8/(rate*1e9)*1e3,
                "selected_full_inputs":selected_payload*8/(rate*1e9)*1e3}
            for rate in gbps},
        "caveat":"Route metadata, true ACK and cryptography, model/state placement, CPU codec cost, and return tensors excluded. Shared raw can beat projected all-expert features."
    }
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--width",type=int,default=384)
    parser.add_argument("--experts",type=int,default=4)
    parser.add_argument("--top-k",type=int,default=2)
    args=parser.parse_args()
    print(json.dumps(compare(args.width,args.experts,args.top_k),indent=2))


if __name__=="__main__":
    main()
