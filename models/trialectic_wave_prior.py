"""DASHI 3/6/9 trialectic-wave coordinates for *learning priors*.

A hidden vector of dimension 384 has 16 source 24D blocks, each of which
contains eight ordered real triads. Orthogonal three-point Fourier axes
separate DC and the two-dimensional rotating C3 phase plane.

This is a data-independent REAL 3-point Fourier rechart. It is not a proof
of Leech automorphism, semantic opposition, physical wave dynamics or a
neural model's C3 equivariance. Face semantics require independent data.
"""
from __future__ import annotations
import math
import torch

_INV_SQRT3 = 1.0 / math.sqrt(3.0)
_INV_SQRT6 = 1.0 / math.sqrt(6.0)
_INV_SQRT2 = 1.0 / math.sqrt(2.0)


def _triads(x: torch.Tensor) -> torch.Tensor:
    if x.ndim < 1 or x.shape[-1] % 3:
        raise ValueError("trialectic-wave coordinates require width divisible by three")
    return x.reshape(*x.shape[:-1], x.shape[-1] // 3, 3)


def wave3(x: torch.Tensor) -> torch.Tensor:
    """Orthogonal 3-point real Fourier coordinates [DC, Re, Im]."""
    a,b,c=_triads(x).unbind(dim=-1)
    dc=(a+b+c)*_INV_SQRT3
    re=(2*a-b-c)*_INV_SQRT6
    im=(b-c)*_INV_SQRT2
    return torch.stack((dc,re,im),dim=-1).reshape(x.shape)


def unwave3(phase: torch.Tensor) -> torch.Tensor:
    """Explicit inverse, including its C3 rotating phase plane."""
    dc,re,im=_triads(phase).unbind(dim=-1)
    a=dc*_INV_SQRT3+2*re*_INV_SQRT6
    b=dc*_INV_SQRT3-re*_INV_SQRT6+im*_INV_SQRT2
    c=dc*_INV_SQRT3-re*_INV_SQRT6-im*_INV_SQRT2
    return torch.stack((a,b,c),dim=-1).reshape(phase.shape)


def rotate3(x: torch.Tensor) -> torch.Tensor:
    """Cyclically permute each ordered trialectic triple."""
    return _triads(x).roll(shifts=-1,dims=-1).reshape(x.shape)


def trialectic_router_invariants(x: torch.Tensor) -> torch.Tensor:
    """C2-invariant, C3-cyclic-invariant coordinates, 3 per triplet.

    Invariants: |DC|, squared rotating-wave amplitude, |a*b*c|.
    The last is an interaction magnitude, not a semantic/causal face.
    Floating sums are approximately, not bitwise, C3 invariant.
    """
    t=_triads(x)
    phase=_triads(wave3(x))
    dc,re,im=phase.unbind(dim=-1)
    a,b,c=t.unbind(dim=-1)
    return torch.stack((dc.abs(),re.square()+im.square(),
                        (a*b*c).abs()),dim=-1).reshape(x.shape)


def phase_rotation_residual(x: torch.Tensor) -> torch.Tensor:
    """Diagnostic: how precisely the C3 generator conjugates to wave3."""
    phase=_triads(wave3(x))
    transformed=_triads(wave3(rotate3(x)))
    dc,re,im=phase.unbind(-1)
    # (a,b,c)->(b,c,a) acts on this real phase basis by -120 degrees.
    c120=-0.5
    s120=-math.sqrt(3.0)/2.0
    expected=torch.stack((dc,c120*re-s120*im,s120*re+c120*im),dim=-1)
    return (transformed-expected).reshape(x.shape)


def geometry_shape(d_model: int, n_heads: int, layers: int) -> dict:
    if d_model % 24 or d_model % n_heads or (d_model//n_heads)%24:
        raise ValueError("learner requires 24D heads/blocks")
    return {"width":d_model,"layers":layers,"heads":n_heads,
            "head_width":d_model//n_heads,"leech24_blocks":d_model//24,
            "head24_blocks":(d_model//n_heads)//24,
            "trialectic_triples_per_layer":d_model//3,
            "triples_per_24D_block":8,
            "ternary_cube_states":27,"antipodal_orbit_classes":14,
            "explicit_semantic_face_required":True}
