"""Lossless dense-to-signed-MoE learning bridge for a *real* trained FFN.

For ANY fixed deterministic dense function F:
    F(x) = E_F(x) + O_F(x)
    E_F(x) = (F(x)+F(-x))/2, O_F(x) = (F(x)-F(-x))/2.
This identity is a genuine transfer curriculum: the old checkpoint remains
the alpha=0 reference and a learned history-bearing odd expert is distilled
towards O_F before it replaces it. No claim that semantic disagreement
equals antipodal inversion is made. An independent face-context channel
is optional and cannot be inferred merely from three dyadic edge values.
"""
import torch
import torch.nn.functional as F
from torch import nn
from models.history_orbit_moe import HistoryOrbitMoE


class DenseToSignedExperts(nn.Module):
    def __init__(self,dense_ffn: nn.Module, d_model: int,
                 num_experts: int=4, top_k: int=2,
                 expert_width: int | None=None,
                 route_geometry: str="c2c3_wave"):
        super().__init__()
        self.teacher=dense_ffn
        # Preserve existing trained source; do not modify its parameters.
        self.teacher.eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        self.student=HistoryOrbitMoE(
            d_model,num_experts,top_k,expert_width,
            route_geometry=route_geometry)
        # Explicit triadic face payload: not predicted from boundary alone.
        self.face=nn.Sequential(nn.Linear(d_model,d_model),
                                nn.GELU(),nn.Linear(d_model,d_model))
        nn.init.zeros_(self.face[-1].weight)
        nn.init.zeros_(self.face[-1].bias)

    def train(self,mode=True):
        super().train(mode)
        # Frozen teacher must not use stochastic dropout on parity pairs.
        self.teacher.eval()
        return self

    def forward(self,x: torch.Tensor,history: torch.Tensor | None=None,
                blend: float=0.0,face_context: torch.Tensor | None=None):
        if not 0 <= blend <= 1:
            raise ValueError("blend must be between 0 and 1")
        with torch.no_grad():
            y_plus=self.teacher(x)
            y_minus=self.teacher(-x)
            y_even=(y_plus+y_minus)*0.5
            y_odd=(y_plus-y_minus)*0.5
        # The sign-canonical history-bearing learner carries the odd
        # oriented part; the source even part is independently preserved.
        student,updated,ids,signs,weights=self.student(
            x,history,train_sign=self.training)
        if face_context is not None:
            if face_context.shape != x.shape:
                raise ValueError("face_context must match x shape")
            independent_face=self.face(face_context)
        else:
            independent_face=torch.zeros_like(x)
        mixture=y_even+(1-blend)*y_odd+blend*student+independent_face
        return {
            "output":mixture,
            "reference":y_plus,
            "even":y_even,
            "odd":y_odd,
            "odd_student":student,
            "odd_distillation_loss":F.mse_loss(student,y_odd),
            "history":updated,
            "expert_ids":ids,
            "signs":signs,
            "weights":weights,
            "face_supplied":face_context is not None,
        }
