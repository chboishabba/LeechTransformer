"""Signed S3 reference geometry; original Leech-LILA: Anatolii Kornienko.
DASHI extension. Exact tuple reference, not a packed GPU implementation.
"""
from dataclasses import dataclass
from itertools import product

# R^k J^b times global sign (-1)^a; R rolls left, J reverses.
GROUP = tuple(product(range(3), range(2), range(2)))
IDENTITY = (0, 0, 0)


def mul(g, h):
    k,b,a = g; l,d,c = h
    return ((k + (-1)**b*l)%3, b^d, a^c)


def inv(g):
    k,b,a = g
    return ((-(-1)**b*k)%3,b,a)


def act(g, v):
    if not v or len(v)%3:
        raise ValueError('nonempty complete triads required')
    k,b,a = g
    out=[]
    for i in range(0,len(v),3):
        t=tuple(v[i:i+3]); t=t[::-1] if b else t
        t=t[k:]+t[:k]
        out.extend((-1)**a*x for x in t)
    return tuple(out)


def stabilizer(v):
    return tuple(g for g in GROUP if act(g,v)==v)


@dataclass(frozen=True)
class MarkedOrbit:
    canonical: tuple
    mark: tuple  # deterministic representative of a left coset g K (right stabilizer action)

    def decode(self):
        return act(self.mark,self.canonical)

    def transport(self, g):
        return MarkedOrbit(self.canonical, min(mul(mul(g,self.mark),h) for h in stabilizer(self.canonical)))


def encode(v):
    v=tuple(v)
    c=max(act(g,v) for g in GROUP)
    return MarkedOrbit(c,min(g for g in GROUP if act(g,c)==v))


def indexed_step(state, x, transition):
    """Frame-local update for a jointly G-equivariant transition only.

    Input in world frame; transition returns (output, next_state) in local
    frame. State mark is re-canonicalized at each time. No equivariance
    checker here: caller must establish the transition contract.
    """
    g=state.mark
    y,w=transition(act(inv(g),tuple(x)),state.canonical)
    local=encode(w)
    return act(g,tuple(y)),local.transport(g)


def reflection_chart(z):
    """Two signed scalar charts; R invariant, J swaps, global sign negates."""
    from .geometric_history_adapter import c3_orientation
    import torch
    j=z.reshape(*z.shape[:-1],-1,3).flip(-1).reshape_as(z)
    return torch.cat((c3_orientation(z),c3_orientation(j)),dim=-1)


def signed_pair_apply(z, canonical_fn):
    """New opt-in branch-symmetrized map. canonical_fn must intertwine S3.

    Not source-identical at a reflection-opposed chart; returns both marks.
    canonical_fn maps a stack of triads to a stack of triads, preserving
    leading dimensions. No training-equivalence claim.
    """
    s=reflection_chart(z)
    a,b=s[...,0:1],s[...,1:2]
    return (a*canonical_fn(a*z)+b*canonical_fn(b*z))/2,s


def signed_pair_residual(z, canonical_fn):
    """Mean plus reflection-odd residual: both branches recoverable."""
    s=reflection_chart(z)
    a,b=s[...,0:1],s[...,1:2]
    u,v=a*canonical_fn(a*z),b*canonical_fn(b*z)
    return (u+v)/2,(u-v)/2,s
