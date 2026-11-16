"""X369 exact-feature -> learned-logit -> top-k bridge for ORIGINAL C3 routers.

The Z-polynomial identity is already represented by DASHI's
X369OriginalDirectedTriadC3PolynomialCandidate.agda. This file
implements a source-semantics refinement at one frozen Leech MoE
router site. It adds NO semantic task action, neural theorem or
frozen-model result. F5 must not be supplied or opened.

For the directed triple T_j and P=I-(11^T/3):
 delta T_j lies in 1^perp, so
 delta score_e = sum_j (P w_j) dot delta T_{ej},
 |delta score_e| <= sum_j ||P w_j|| sqrt(2 q(x_j)q(h_ej)).
Also a global Cauchy bound holds. Exact feature energy therefore
does NOT imply nonzero learned score change: P w=0 is a counterexample.
"""
from __future__ import annotations

import torch

from models.trialectic_wave_prior import (
    rotate3, trialectic_router_invariants, directed_trialectic_invariants)


def _q(z):
    triads=z.reshape(*z.shape[:-1],z.shape[-1]//3,3)
    a,b,c=triads.unbind(-1)
    return 0.5*((a-b).square()+(b-c).square()+(c-a).square())


def projected_original_c3_router_weld(
        x: torch.Tensor, history: torch.Tensor, weight: torch.Tensor,
        *, expert_bias: torch.Tensor | None=None, top_k: int=1) -> dict:
    """Return a report on measured frozen source vectors and actual head.

    Input tensors: x [N,3m], history [N,E,3m], single shared route
    weight [2*3m] or [1,2*3m], optional expert bias [E].
    This function has NO model mutation, data I/O or training.
    """
    if (x.ndim!=2 or x.shape[0]<1 or x.shape[1]<3 or x.shape[1]%3 or
            history.ndim!=3 or history.shape[0]!=len(x) or
            history.shape[2]!=x.shape[1] or history.shape[1]<2 or
            not torch.is_floating_point(x) or
            history.dtype!=x.dtype or history.device!=x.device or
            not bool(torch.isfinite(x).all()) or
            not bool(torch.isfinite(history).all())):
        raise ValueError("finite original source x[N,3m], h[N,E,3m] required")
    d=x.shape[1]
    n,e=history.shape[:2]
    if weight.ndim==2 and weight.shape[0]==1:
        weight=weight[0]
    if (weight.ndim!=1 or weight.numel()!=2*d or
            weight.dtype!=x.dtype or weight.device!=x.device or
            not bool(torch.isfinite(weight).all()) or
            type(top_k) is not int or not 1<=top_k<e):
        raise ValueError("one finite shared original 2d route head and 1<=top_k<E required")
    if expert_bias is None:
        expert_bias=torch.zeros(e,dtype=x.dtype,device=x.device)
    if (expert_bias.shape!=(e,) or expert_bias.dtype!=x.dtype or
            expert_bias.device!=x.device or
            not bool(torch.isfinite(expert_bias).all())):
        raise ValueError("source expert bias [E] mismatches router")
    xr=x[:,None,:].expand(-1,e,-1)
    rot=rotate3(x)[:,None,:].expand(-1,e,-1)
    before=torch.cat((
        trialectic_router_invariants(xr),
        directed_trialectic_invariants(xr,history)),dim=-1)
    after=torch.cat((
        trialectic_router_invariants(rot),
        directed_trialectic_invariants(rot,history)),dim=-1)
    delta=(after-before)[...,d:].reshape(n,e,d//3,3)
    w=weight[d:].reshape(d//3,3)
    projected=w-w.mean(dim=-1,keepdim=True)
    # Exact zero-sum property of each rotated directed triad; only
    # w's 2D rotating projection is visible to source score changes.
    delta_score=(delta*w).sum(dim=(-1,-2))
    projected_score=(delta*projected).sum(dim=(-1,-2))
    qx=_q(x)[:,None,:]
    qh=_q(history)
    feature_sq=(delta.square()).sum(dim=-1)
    analytic_sq=2*qx*qh
    per_triad_bound=(
        projected.square().sum(-1).sqrt()[None,None,:]*
        analytic_sq.clamp_min(0).sqrt()).sum(-1)
    global_bound=(
        projected.square().sum().sqrt()*
        analytic_sq.clamp_min(0).sum(-1).sqrt())
    bound=torch.minimum(per_triad_bound,global_bound)
    before_logits=(before@weight)+expert_bias
    after_logits=(after@weight)+expert_bias
    actual_logit_delta=after_logits-before_logits
    source_ids=before_logits.topk(top_k,dim=-1).indices
    target_ids=after_logits.topk(top_k,dim=-1).indices
    selected=torch.zeros_like(before_logits,dtype=torch.bool)
    selected.scatter_(1,source_ids,True)
    worst_selected=(before_logits-bound).masked_fill(~selected,float("inf")).min(-1).values
    best_unselected=(before_logits+bound).masked_fill(selected,-float("inf")).max(-1).values
    # Strict separation is a SUFFICIENT route-set preservation
    # certificate; it does not promise unchanged mixture weights.
    margin=worst_selected-best_unselected
    predicted_safe=margin>1e-9
    same_set=(source_ids.sort(-1).values==target_ids.sort(-1).values).all(-1)
    atol=1e-10 if x.dtype==torch.float64 else 2e-4
    rtol=1e-10 if x.dtype==torch.float64 else 3e-5
    torch.testing.assert_close(
        delta_score,projected_score,atol=atol,rtol=rtol)
    torch.testing.assert_close(
        actual_logit_delta,delta_score,atol=atol,rtol=rtol)
    torch.testing.assert_close(
        feature_sq,analytic_sq,atol=atol,rtol=rtol)
    if bool((delta_score.abs()>bound+atol+rtol*bound.abs()).any()):
        raise AssertionError("original learned-logit C3 projection bound violated")
    if bool((predicted_safe & ~same_set).any()):
        raise AssertionError("strict route-margin certificate contradicted by source")
    unary_shift=(after-before)[...,:d]
    if float(unary_shift.abs().max())>atol+rtol*float(before[...,:d].abs().max()):
        raise AssertionError("unary invariant feature unexpectedly moved")
    result={
        "schema":1,
        "kind":"x369_original_triad_feature_learned_logit_topk_bridge",
        "source_rows":n,"expert_count":e,"hidden_width":d,"top_k":top_k,
        "relative_c3_directed_feature_energy_max_abs_error":
            float((feature_sq-analytic_sq).abs().max()),
        "directed_feature_zero_sum_max_abs_error":
            float(delta.sum(-1).abs().max()),
        "weight_projection_score_max_abs_error":
            float((delta_score-projected_score).abs().max()),
        "full_source_logit_projection_max_abs_error":
            float((actual_logit_delta-delta_score).abs().max()),
        "source_phase_projected_logit_delta_mean_abs":
            float(delta_score.abs().mean()),
        "source_phase_feature_displacement_mean":
            float(feature_sq.sum(-1).sqrt().mean()),
        "source_score_bound_max_violation":
            float((delta_score.abs()-bound).clamp_min(0).max()),
        "fraction_strict_topk_set_stability_certified":
            float(predicted_safe.float().mean()),
        "fraction_original_topk_set_actually_changed":
            float((~same_set).float().mean()),
        "strict_topk_margin_min":float(margin.min()),
        "constant_weight_triad_DC_projection_is_invisible":True,
        "same_topk_set_does_not_imply_equal_mixture_weights":True,
        "grokking_semantic_action_verified":False,
        "sophontic_intervention_verified":False,
        "dialectic_semantic_opposition_inferred":False,
        "triadic_face_inferred":False,
        "F5_raw_or_reserved_read":False,
    }
    return result
