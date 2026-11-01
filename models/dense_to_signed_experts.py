"""Dense Leech FFN -> deployable signed odd + invariant even + independent face.

For any fixed deterministic F:
 E(x)=(F(x)+F(-x))/2, O(x)=(F(x)-F(-x))/2, F=E+O.
The trained F provides targets; students learn BOTH parity channels.
A sign-canonical even learner preserves relative sign information (unlike
elementwise |x|). The odd learner is a history-bearing routed MoE.
The face channel requires independent context: no dyadic edge inference.

At blend=0, dense reference is exactly retained to float precision.
At blend=1, predict_only requires NO dense teacher; compare its output
against F and preserve an exact residual whenever not equivalent.
"""
import torch
import torch.nn.functional as F
from torch import nn
from models.history_orbit_moe import HistoryOrbitMoE


def canonical_even_input(x: torch.Tensor) -> torch.Tensor:
    """Choose one representative of {x,-x}; zero is its own centre."""
    first=(x != 0).to(torch.int64).argmax(dim=-1,keepdim=True)
    sign=torch.sign(x.gather(-1,first))
    return sign*x


class DenseToSignedExperts(nn.Module):
    def __init__(self,dense_ffn: nn.Module, d_model: int,
                 num_experts: int=4, top_k: int=2,
                 expert_width: int | None=None,
                 route_geometry: str="punctured9_wave",
                 transition_rank: int | None=24,
                 even_width: int | None=None):
        super().__init__()
        self.teacher=dense_ffn
        self.teacher.eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        self.student=HistoryOrbitMoE(
            d_model,num_experts,top_k,expert_width,
            route_geometry=route_geometry,
            transition_rank=transition_rank)
        width=even_width or max(24,d_model//2)
        self.even_student=nn.Sequential(
            nn.Linear(d_model,width),nn.GELU(),nn.Linear(width,d_model))
        # Explicit triadic face payload: not predicted from boundary alone.
        self.face=nn.Sequential(nn.Linear(d_model,d_model),
                                nn.GELU(),nn.Linear(d_model,d_model))
        nn.init.zeros_(self.face[-1].weight)
        nn.init.zeros_(self.face[-1].bias)

    def train(self,mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    def predict_only(self,x: torch.Tensor,history: torch.Tensor | None=None,
                     face_context: torch.Tensor | None=None):
        """Deployment path never evaluates or depends on teacher weights."""
        if face_context is not None and face_context.shape != x.shape:
            raise ValueError("face_context must match x shape")
        odd,updated,ids,signs,weights=self.student(
            x,history,train_sign=self.training)
        even=self.even_student(canonical_even_input(x))
        face=self.face(face_context) if face_context is not None else torch.zeros_like(x)
        return {"output":even+odd+face,"even_student":even,
                "odd_student":odd,"history":updated,
                "expert_ids":ids,"signs":signs,"weights":weights,
                "face_supplied":face_context is not None}

    def forward(self,x: torch.Tensor,history: torch.Tensor | None=None,
                blend: float=0.0,face_context: torch.Tensor | None=None):
        if not 0 <= blend <= 1:
            raise ValueError("blend must be between 0 and 1")
        with torch.no_grad():
            y_plus=self.teacher(x)
            y_minus=self.teacher(-x)
            y_even=(y_plus+y_minus)*0.5
            y_odd=(y_plus-y_minus)*0.5
        predicted=self.predict_only(x,history,face_context)
        residual=y_plus-predicted["output"]
        result=(1-blend)*y_plus+blend*predicted["output"]
        return {
            "output":result,"reference":y_plus,
            "even":y_even,"odd":y_odd,
            "even_student":predicted["even_student"],
            "odd_student":predicted["odd_student"],
            "odd_distillation_loss":F.mse_loss(predicted["odd_student"],y_odd),
            "even_distillation_loss":F.mse_loss(predicted["even_student"],y_even),
            "student_distillation_loss":(
                F.mse_loss(predicted["odd_student"],y_odd)+
                F.mse_loss(predicted["even_student"],y_even)),
            "exact_residual":residual,
            "student_output":predicted["output"],
            "history":predicted["history"],
            "expert_ids":predicted["expert_ids"],
            "signs":predicted["signs"],
            "weights":predicted["weights"],
            "face_supplied":predicted["face_supplied"],
        }
