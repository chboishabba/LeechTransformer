"""Experimental signed-conjugate, history-conditioned expert with exact routing witness.

Independent extension, not a modification of Kornienko's Leech-LILA model.
No claim that sign equals benevolence/harmfulness or is a Leech automorphism.
"""
import torch
from torch import nn


class SignedHistoryExpert(nn.Module):
    """One base expert shared by two orientation charts.

    State is a per-token history summary maintained by the caller; the model
    returns the updated state. Sign routing is a *learned* latent relation.
    y = s * F(s*x, s*h); next_h = s * U(s*h, s*x, F(...)).
    Thus switching the chart sign consistently on both x and history
    induces the corresponding signed orientation on the output. The
    learned router can still choose different signs and need not itself
    be equivariant; its route witness must be transported.
    """

    def __init__(self, d_model: int = 384, d_ff: int | None = None):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.router = nn.Linear(2 * d_model, 1)
        self.expert = nn.Sequential(
            nn.Linear(2 * d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.update = nn.Linear(3 * d_model, d_model)

    def forward(self, x: torch.Tensor, history: torch.Tensor,
                *, forced_sign: torch.Tensor | None = None):
        if x.shape != history.shape or x.shape[-1] * 2 != self.router.in_features:
            raise ValueError("x/history must have identical [...,d_model] shapes")
        scores = self.router(torch.cat((x, history), -1))
        sign = torch.where(scores >= 0, torch.ones_like(scores), -torch.ones_like(scores))
        if forced_sign is not None:
            if forced_sign.shape != scores.shape or not bool(torch.all((forced_sign == 1) | (forced_sign == -1))):
                raise ValueError("forced_sign must be an exact ±1 tensor shaped [...,1]")
            sign = forced_sign
        x_c, h_c = sign * x, sign * history
        y_c = self.expert(torch.cat((x_c, h_c), -1))
        next_c = self.update(torch.cat((h_c, x_c, y_c), -1))
        return sign * y_c, sign * next_c, sign, scores

    @staticmethod
    def recanonicalize(y: torch.Tensor, history: torch.Tensor, sign: torch.Tensor):
        """Inverse sign chart, involutive; exact negation on finite floats."""
        return sign * y, sign * history


def signed_training_objective(module, x, history, target, history_target,
                              *, route_sign=None):
    """Gradients reach the canonical expert through the signed inverse.

    Router's hard threshold is nondifferentiable: provide supervision for
    its logits or replace this baseline with a differentiable relaxation.
    This loss trains the expert and history transition, NOT router quality.
    """
    y, new_history, sign, scores = module(x, history, forced_sign=route_sign)
    return ((y - target)**2).mean() + ((new_history-history_target)**2).mean(), sign, scores
