"""X369 F3/F4 original-router confound audit: shared head and feature metric.

Actual HistoryOrbitMoE source:
  C2 route  = (abs(x), abs(h_e))
  C3 route  = (trialectic_invariants(x), directed_invariants(x,h_e)).
ONE nn.Linear(2d,1) is SHARED by all experts. Consequently its x-only
contribution is constant across experts and cancels in softmax/top-k
comparisons, although the history-dependent routing contributions need
not agree. The two feature maps are NOT a function-preserving clone.

The F3 matched null preserves ||x|| and ||x'-x|| but does not preserve
||F(x,h)-F(x',h)|| for this nonlinear router. This module measures both
feature-space and logit-space disparities. No final lockbox or model
training and no fabricated original-state capture.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from models.trialectic_wave_prior import (
    rotate3,trialectic_router_invariants,directed_trialectic_invariants)
from models.c3_matched_router_control import matched_displacement_random_x


def original_router_features(x,history,geometry):
    if (x.ndim!=2 or history.ndim!=3 or history.shape[0]!=x.shape[0]
            or history.shape[2]!=x.shape[1] or x.shape[1]%3
            or history.shape[1]<2 or not torch.isfinite(x).all()
            or not torch.isfinite(history).all()):
        raise ValueError("finite original [batch,3k] and [batch,experts,3k] required")
    if geometry not in ("c2_invariant","c2c3_wave"):
        raise ValueError("source C2/C2xC3 geometry only")
    xr=x[:,None,:].expand(-1,history.shape[1],-1)
    if geometry=="c2_invariant":
        return torch.cat((xr.abs(),history.abs()),dim=-1)
    return torch.cat((trialectic_router_invariants(xr),
                      directed_trialectic_invariants(xr,history)),dim=-1)


def source_f3_f4_router_geometry_audit(x,history,weight,*,seed=41):
    if (x.ndim!=2 or not torch.is_floating_point(x) or
            history.dtype!=x.dtype or history.device!=x.device):
        raise ValueError("original source activation/history dtype/device mismatch")
    if weight.ndim==2 and weight.shape[0]==1:
        weight=weight.squeeze(0)
    if (weight.ndim!=1 or weight.numel()!=2*x.shape[1]
            or not torch.isfinite(weight).all()):
        raise ValueError("ONE shared original router [2d] weight required")
    if weight.dtype!=x.dtype or weight.device!=x.device:
        raise ValueError("router weight dtype/device mismatch")
    d=x.shape[1];e=history.shape[1];n=len(x)
    with torch.no_grad():
        c2=original_router_features(x,history,"c2_invariant")
        c3=original_router_features(x,history,"c2c3_wave")
        w_x=weight[:d];w_h=weight[d:]
        c2_unary=c2[...,:d]@w_x
        c3_unary=c3[...,:d]@w_x
        unary_spread_c2=(c2_unary-c2_unary[:,:1]).abs().max()
        unary_spread_c3=(c3_unary-c3_unary[:,:1]).abs().max()
        c2_logits=c2@weight
        c3_logits=c3@weight
        centered=lambda scores:scores-scores.mean(dim=1,keepdim=True)
        relative_logit_shift=centered(c3_logits)-centered(c2_logits)
        per_expert_history_score_gap=(c3[...,d:]-c2[...,d:])@w_h
        relative_feature_before=c3
        rotated=rotate3(x)
        random_x=matched_displacement_random_x(x,seed=seed)
        f_rel=original_router_features(rotated,history,"c2c3_wave")
        f_null=original_router_features(random_x,history,"c2c3_wave")
        def norms(z):
            return torch.linalg.vector_norm(z.reshape(n,-1).double(),dim=-1)
        rel_delta=norms(f_rel-relative_feature_before)
        null_delta=norms(f_null-relative_feature_before)
        source_rel=torch.linalg.vector_norm((rotated-x).double(),dim=-1)
        source_null=torch.linalg.vector_norm((random_x-x).double(),dim=-1)
        eligible=rel_delta>1e-10
        if not eligible.any():
            raise ValueError("relative C3 source feature delta collapsed; comparison undefined")
        ratio=null_delta[eligible]/rel_delta[eligible]
        feature_matched=(null_delta-rel_delta).abs() <= (
            .05*torch.maximum(rel_delta,null_delta).clamp_min(1e-12))
        nonlinear_invariant_delta=norms(
            (f_rel-relative_feature_before)[...,:d])
        nonlinear_random_delta=norms(
            (f_null-relative_feature_before)[...,:d])
        cosine=torch.nn.functional.cosine_similarity(
            centered(c2_logits).double(),centered(c3_logits).double(),
            dim=-1,eps=1e-12)
        return {
            "schema":1,"kind":"x369_source_router_clone_and_f3_feature_metric_audit",
            "original_source_samples":n,"expert_count":e,
            "original_hidden_dimension":d,
            "same_2d_router_head_across_all_experts":True,
            "common_x_unary_logit_spread_c2_max":float(unary_spread_c2),
            "common_x_unary_logit_spread_c3_max":float(unary_spread_c3),
            "original_history_routing_score_gap_rms":float(
                per_expert_history_score_gap.square().mean().sqrt()),
            "expert_relative_logits_geometry_gap_rms":float(
                relative_logit_shift.square().mean().sqrt()),
            "expert_centered_logit_cosine_mean":float(cosine.mean()),
            "source_x_displacement_rel_vs_random_max_abs_difference":float(
                (source_rel-source_null).abs().max()),
            "feature_delta_rel_mean":float(rel_delta.mean()),
            "feature_delta_random_mean":float(null_delta.mean()),
            "feature_delta_random_to_rel_ratio_median":float(
                ratio.median()),
            "fraction_original_source_rows_feature_displacement_matched_5pct":
                float(feature_matched.float().mean()),
            "relative_C3_unary_feature_delta_max":float(
                nonlinear_invariant_delta.max()),
            "random_unary_feature_delta_mean":float(
                nonlinear_random_delta.mean()),
            "source_activation_matched_is_not_router_feature_matched":True,
            "geometry_comparison_same_initial_function_established":False,
            "source_model_C3_learning_or_phase_proved":False,
            "no_F5_rows_loaded":True,
        }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-x",type=Path,required=True,
                   help="original captured route activations [N,d]; NOT a test lockbox")
    p.add_argument("--source-history",type=Path,required=True,
                   help="same captured original expert histories [N,experts,d]")
    p.add_argument("--original-router-weight",type=Path,required=True,
                   help="original shared nn.Linear weight [2d] or [1,2d]")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--seed",type=int,default=41)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError("refuse overwrite F3/F4 router evidence")
    data=[np.load(q,allow_pickle=False) for q in
          (a.source_x,a.source_history,a.original_router_weight)]
    if not all(np.issubdtype(x.dtype,np.floating) for x in data):
        raise ValueError("original saved route tensors must be floating")
    dtype=torch.float64
    report=source_f3_f4_router_geometry_audit(
        *(torch.as_tensor(x,dtype=dtype) for x in data),seed=a.seed)
    report["inputs"]=[{"path":str(q),
                      "sha256":hashlib.sha256(q.read_bytes()).hexdigest()}
                     for q in (a.source_x,a.source_history,
                               a.original_router_weight)]
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,sort_keys=True,indent=2,
                                   allow_nan=False)+"\n")
    print(json.dumps(report,sort_keys=True,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
