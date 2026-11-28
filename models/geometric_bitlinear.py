"""Opt-in signed C3/S3 BitLinear candidate for LILA.

Original Leech-LILA: Anatolii Kornienko. DASHI geometric extension.
BitNet ternary alphabet inspiration; not a port of Microsoft's CUDA kernel.
The base operator is a sum of group actions. The quadratic lift must be
generated AFTER the sum; it is not an independent learned 6x6 matrix.
"""
import torch
from torch import nn


def apply_group_triads(x: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
    """Apply one 3- or 6-coefficient group-algebra block per input triad.

    x: [..., 3*n]; coefficients: [n,3] or [n,6].
    Generator R matches trialectic_wave_prior.rotate3; J reverses a triad.
    Coefficients can be real; ternary enforcement belongs to the caller.
    """
    if x.ndim < 1 or x.shape[-1] % 3:
        raise ValueError("complete triads required")
    n = x.shape[-1] // 3
    if coefficients.shape not in ((n, 3), (n, 6)):
        raise ValueError("expected one 3- or 6-coefficient row per triad")
    t = x.reshape(*x.shape[:-1], n, 3)
    result = sum(coefficients[:, k, None] * torch.roll(t, -k, -1)
                 for k in range(3))
    if coefficients.shape[1] == 6:
        jt = t.flip(-1)
        result = result + sum(coefficients[:, 3 + k, None] *
                              torch.roll(jt, -k, -1) for k in range(3))
    return result.reshape_as(x)


def generated_369(x: torch.Tensor) -> torch.Tensor:
    """Per-triad graph embedding v |-> (v, x²,y²,z²,xy,yz,zx).

    Nine coordinates, only three independent real DOF on the graph.
    This is NOT a replacement for the source ordered pair observer.
    """
    if x.ndim < 1 or x.shape[-1] % 3:
        raise ValueError("complete triads required")
    t = x.reshape(*x.shape[:-1], x.shape[-1] // 3, 3)
    a, b, c = t.unbind(-1)
    q = torch.stack((a*a,b*b,c*c,a*b,b*c,c*a),dim=-1)
    return torch.cat((t,q),dim=-1).flatten(-2)


class GeometricBitLinear(nn.Module):
    """Block-diagonal candidate; independent group coefficients per triad.

    fixed-unit ternary STE is an experimental surrogate, not a claim of
    BitNet absmean QAT equivalence or a trained-model improvement.
    scale is fixed, positive and applied before the nonlinear lift.
    S3 coefficients are ternary; expanded matrix entries need not be.
    """
    def __init__(self, d_model: int, reflections: bool = False,
                 scale: float = 1.0):
        super().__init__()
        if d_model < 3 or d_model % 3:
            raise ValueError("positive width divisible by three required")
        if not 0 < scale < float("inf"):
            raise ValueError("finite positive scale required")
        self.d_model = d_model
        self.scale = float(scale)
        coeff = torch.zeros(d_model // 3, 6 if reflections else 3)
        coeff[:, 0] = 1
        self.coefficients = nn.Parameter(coeff)

    def quantized_coefficients(self) -> torch.Tensor:
        q = self.coefficients.round().clamp(-1, 1)
        if self.training:
            return q.detach() + (self.coefficients - self.coefficients.detach())
        return q

    def forward(self, x: torch.Tensor, lift: bool = False) -> torch.Tensor:
        if x.shape[-1] != self.d_model:
            raise ValueError("input width must match module")
        y = self.scale * apply_group_triads(x, self.quantized_coefficients())
        return generated_369(y) if lift else y
