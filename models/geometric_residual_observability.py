"""Geometry-forward residual reference. Original Leech-LILA: Anatolii Kornienko.
DASHI extension. Real-arithmetic contracts; no fitted or native GPU claim.
"""
import math
import torch


def triad_probes(*, dtype=None, device=None):
    """Six ternary rows: singles and sums in complement-index order."""
    i=torch.eye(3,dtype=dtype,device=device)
    return torch.cat((i,torch.ones_like(i)-i),dim=0)


def squared_probe_packet(d):
    return (d @ triad_probes(dtype=d.dtype,device=d.device).T).square()


def gram_from_squared_packet(packet):
    """Linear polarization inverse; input must be squared probes for rank-one interpretation."""
    if packet.shape[-1]!=6: raise ValueError('six squared probes required')
    a,b,c,bc,ca,ab=packet.unbind(-1)
    xy=(ab-a-b)/2; yz=(bc-b-c)/2; zx=(ca-c-a)/2
    return torch.stack((a,xy,zx,xy,b,yz,zx,yz,c),-1).reshape(*packet.shape[:-1],3,3)


def gelu_curvature_bound(preactivation_delta):
    """Euclidean bound for midpoint defect of exact coordinate GELU."""
    return math.sqrt(2/math.pi)/2*preactivation_delta.square().norm(dim=-1)


def propagate_envelope(lipschitz, local_defects, initial=0.):
    """Nonnegative declared local bounds, not an automatic guard certificate."""
    if len(lipschitz)!=len(local_defects):raise ValueError('length mismatch')
    e=float(initial);out=[e]
    if not math.isfinite(e) or e<0:raise ValueError('invalid initial bound')
    for L,d in zip(lipschitz,local_defects):
        if not all(math.isfinite(v) and v>=0 for v in (L,d)):raise ValueError('invalid bound')
        e=L*e+d;out.append(e)
    return out
