"""Trainable paired semantic + conditional geometric objective.

Source contrast: Sophontic's paired perturbation evaluation is externally
authored. DASHI's argument-response non-antipode and trialectic face
nonfactorability require independent labels and prevent auto-sign mapping.

The user/dataset provides TWO correct token targets, response category,
AND (if legitimately known) a separate latent action certificate.
Unknown action = no geometry penalty. This is a testable training prior,
not a proof of reasoning or semantics.
"""
import torch
import torch.nn.functional as F
from models.trialectic_wave_prior import rotate3

ACTION_LABELS={"none","identity","C2_inversion","C3_cyclic"}


def paired_training_objective(
        logits_before: torch.Tensor,logits_after: torch.Tensor,
        target_before: torch.Tensor,target_after: torch.Tensor,
        hidden_before: torch.Tensor,hidden_after: torch.Tensor,
        *,known_actions: list[str] | None=None,
        geometry_weight:float=0.0):
    if (logits_before.ndim!=2 or logits_before.shape!=logits_after.shape
        or hidden_before.ndim!=2 or hidden_before.shape!=hidden_after.shape
        or logits_before.shape[0]!=hidden_before.shape[0]):
        raise ValueError("paired logits/hidden shape mismatch")
    n=hidden_before.shape[0]
    if target_before.shape!=(n,) or target_after.shape!=(n,):
        raise ValueError("correct labels required for BOTH members")
    if geometry_weight<0:
        raise ValueError("geometry_weight cannot be negative")
    if known_actions is None:
        known_actions=["none"]*n
    if len(known_actions)!=n or any(a not in ACTION_LABELS for a in known_actions):
        raise ValueError("invalid independently supplied geometric actions")
    correctness=(F.cross_entropy(logits_before,target_before)+
                 F.cross_entropy(logits_after,target_after))*0.5
    action_cost=[]
    for i,act in enumerate(known_actions):
        if act=="none":
            continue
        expected={"identity":lambda x:x,
                  "C2_inversion":lambda x:-x,
                  "C3_cyclic":rotate3}[act](hidden_before[i:i+1])
        action_cost.append(F.mse_loss(hidden_after[i:i+1],expected))
    geometric=(torch.stack(action_cost).mean() if action_cost else
               correctness.new_zeros(()))
    return {"loss":correctness+geometry_weight*geometric,
            "paired_cross_entropy":correctness,
            "licensed_action_loss":geometric,
            "licensed_action_count":len(action_cost),
            "missing_action_is_not_assumed_inversion":True}
