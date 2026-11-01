"""Explicit 8D E8 root-system geometric prior for small Leech-LILA learners.

24D = 3 × E8-coordinate blocks, 384D = 48 × 8D, and each 48D head
contains six 8D blocks. These are ambient E8 *charts*; three copies
of E8 are not the Leech lattice without a gluing/coset receipt.

The E8 lattice roots are 112 two-sparse integer roots ±ei±ej and 128
half-integer roots with even sign parity. All 240 have norm²=2.
A reflection in any root is an exact Weyl-group isometry mathematically.
Float Torch errors are measured, not proof.

The invariant router features below are invariant to a simultaneous
E8 Weyl reflection of activation+expert-history 8D blocks:
  ||x||², ||h||², x·h, root max |x·r|, root max |h·r|,
  root max |(x+h)·r|, fourth root moment x, fourth root moment h.
This is a candidate INDUCTIVE BIAS on router selection, not a
guarantee of whole-network E8 equivariance nor semantic disagreement.

DASHI owners: Algebra/Trit/E8RootEnumeration.agda, Biology/
ExceptionalLatticeGrokkingProtocolExact.agda, Moonshine/
MonsterThreeLocalE8LeechBridgeExact.agda.
"""
from __future__ import annotations
from functools import lru_cache
from itertools import combinations,product
import math
import torch


@lru_cache(maxsize=1)
def _e8_roots_cpu()->torch.Tensor:
    vectors=[]
    for i,j in combinations(range(8),2):
        for a,b in product((-1,1),repeat=2):
            row=[0.0]*8;row[i]=float(a);row[j]=float(b)
            vectors.append(row)
    for signs in product((-1,1),repeat=8):
        if sum(sign<0 for sign in signs)%2==0:
            vectors.append([s*0.5 for s in signs])
    assert len(vectors)==240
    return torch.tensor(vectors,dtype=torch.float64)


def e8_roots(*,device=None,dtype=torch.float32)->torch.Tensor:
    if dtype not in (torch.float32,torch.float64):
        raise ValueError("real float32/64 E8 carrier required")
    return _e8_roots_cpu().to(device=device,dtype=dtype)


def reflect_e8(x:torch.Tensor,root_index:int)->torch.Tensor:
    """Reflect all contiguous 8D charts using one legitimate E8 root."""
    if x.shape[-1]%8 or not isinstance(root_index,int) or not 0<=root_index<240:
        raise ValueError("E8 chart width/index invalid")
    root=e8_roots(device=x.device,dtype=x.dtype)[root_index]
    blocks=x.reshape(*x.shape[:-1],-1,8)
    # Root norm squared=2; Weyl reflection v->v-2(v.r)/(r.r)*r.
    result=blocks-(blocks*root).sum(-1,keepdim=True)*root
    return result.reshape(x.shape)


def e8_router_invariants(x:torch.Tensor,h:torch.Tensor)->torch.Tensor:
    """Eight invariant scalar features per pair of 8D coordinate charts.

    Shape [...,d] rather than [...,2d], where d is divisible by 8.
    Jointly sign-invariant. E8 Weyl-invariant under simultaneous
    reflection in a root when exact arithmetic is assumed. In floating
    arithmetic root orbits are compared within tolerance.
    """
    if x.shape!=h.shape or x.shape[-1]%8 or x.dtype not in (torch.float32,torch.float64):
        raise ValueError("E8 requires paired equal-shaped real 8D charts")
    xb=x.reshape(*x.shape[:-1],-1,8)
    hb=h.reshape(*h.shape[:-1],-1,8)
    roots=e8_roots(device=x.device,dtype=x.dtype)
    xdot=xb@roots.T
    hdot=hb@roots.T
    x2=(xb*xb).sum(-1)
    h2=(hb*hb).sum(-1)
    cross=(xb*hb).sum(-1)
    # Use normalized (by 240) fourth moments; computed per 8D block.
    x4=xdot.square().square().mean(-1)
    h4=hdot.square().square().mean(-1)
    values=torch.stack((
        x2,h2,cross,
        xdot.abs().amax(-1),hdot.abs().amax(-1),
        (xdot+hdot).abs().amax(-1),
        x4,h4),dim=-1)
    return values.reshape(x.shape)


def e8_geometry_receipt(width:int=384,heads:int=8)->dict:
    if width%24 or heads<1 or width%heads or (width//heads)%8:
        raise ValueError("E8/Leech24 head alignment required")
    return {"hidden_width":width,"head_count":heads,
            "e8_rank":8,"e8_roots":240,
            "e8_charts":width//8,"e8_charts_per_head":(width//heads)//8,
            "e8_charts_per_24d_block":3,
            "leech24_blocks":width//24,
            "Leech_requires_independent_gluing_proof":True,
            "punctured_nine_eight_is_not_e8_identity":True}
