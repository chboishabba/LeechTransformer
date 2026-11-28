"""Offline exact rational replay of a native quantization *observation* JSON.
Verifies internal arithmetic of the claimed receipt, not source authenticity,
source code SHA, actual HIP execution, or prospective native error bounds.
"""
import argparse,json
from fractions import Fraction as Q
from pathlib import Path

def certificate(observed,candidate,order):
    n=len(observed); selected=set(order)
    edges=[(i,j) for i in order for j in range(n) if j not in selected]
    edges.extend(zip(order,order[1:]))
    a=[observed[i]-observed[j] for i,j in edges]
    b=[candidate[i]-candidate[j] for i,j in edges]
    return dict(observed_strict=all(x>0 for x in a),candidate_strict=all(x>0 for x in b),
        observed_order_certified=all(x>0 for x in a),
        both_orders_certified=all(x>0 for x in a) and all(x>0 for x in b),
        source_min_pair_gap=str(min(a)) if a else None,
        candidate_min_pair_gap=str(min(b)) if b else None,
        max_observed_pair_perturbation=str(max(abs(x-y) for x,y in zip(a,b))) if a else None,
        native_future_error_bound_available=False)

def verify(d):
    if d.get("kind")!="x369_native_two_channel_quantization_observation" or d.get("retrospective_exact_dyadic") is not True or d.get("prospective_native_rounding_certified") is not False or d.get("nat_cell_certified") is not False or d.get("future_history_sufficient") is not False:
        raise ValueError("invalid scope")
    if Q(d["quantum"])<=0:raise ValueError("invalid quantum")
    for key in ("source_x_sha256","source_h_sha256","observed_model_state_sha256"):
        v=d.get(key)
        if not isinstance(v,str) or len(v)!=64 or any(c not in "0123456789abcdef" for c in v):
            raise ValueError("missing bound hash")
    conf=d["model_runtime_config"]
    if conf.get("route_geometry")!="c2_c3_residual" or conf.get("router_selection")!="topk":
        raise ValueError("unsupported source")
    count=0
    for row in d["records"]:
        scores=row["expert_scores"]; n=len(scores)
        order=row["ids"]; k=conf["top_k"]
        if n<2 or not isinstance(k,int) or not 1<=k<=n or len(order)!=k or len(set(order))!=k or any(not isinstance(i,int) or not 0<=i<n for i in order):
            raise ValueError("invalid selection")
        if len(row["native_selected_weight"])!=k:
            raise ValueError("selected weights mismatch")
        observed=[]; candidate=[]
        for j,item in enumerate(scores):
            if item["expert"]!=j:raise ValueError("expert index mismatch")
            c=Q(item["candidate"]); err=Q(item["quantization_error"])
            full=Q(item["exact_dyadic_directed"])
            residual=Q(item["native_directed_residual"])
            native=Q(item["observed_logit"])
            if c+err!=full:raise ValueError("original directed score not reconstructed")
            # Reconstruct the rounded baseline from stored differences.
            baseline=native-full-residual
            observed.append(native);candidate.append(c+baseline)
        rank=sorted(range(n),key=lambda j:(-candidate[j],j))
        if rank[:k]!=row["quantized_selected_ids"]:
            raise ValueError("candidate order mismatch")
        selected=set(order)
        margins=[candidate[i]-candidate[j] for i in order for j in range(n) if j not in selected]
        expected=str(min(margins)) if margins else None
        if row["strict_candidate_membership_margin"]!=expected:
            raise ValueError("candidate membership margin mismatch")
        if row.get("retrospective_pair_certificate")!=certificate(observed,candidate,order):
            raise ValueError("retrospective pair certificate mismatch")
        count+=1
    if not count:raise ValueError("empty receipt")
    return count

def main():
    p=argparse.ArgumentParser()
    p.add_argument("receipt",type=Path)
    a=p.parse_args()
    with a.receipt.open(encoding="utf-8") as fp:d=json.load(fp)
    print(json.dumps(dict(status="RETROSPECTIVE_REPLAY_PASS",samples=verify(d),
        source_authenticity_independently_verified=False,prospective_native_error_certified=False)))
if __name__=="__main__":main()
