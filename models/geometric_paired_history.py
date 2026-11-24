"""Opt-in paired-history LILA reference. Original Leech-LILA: Anatolii Kornienko.
DASHI extension; original source not changed. No native packed kernel claim.
"""
from dataclasses import dataclass
import torch
from torch import nn
from .geometric_reflection_fibre import GROUP, act, mul, inv, reflection_chart


def pair_action(g, pair):
    """Tuple reference on two complete, equally sized branch vectors."""
    a,b=pair
    if len(a)!=len(b):
        raise ValueError('unequal branch widths')
    return (act(g,b),act(g,a)) if g[1] else (act(g,a),act(g,b))


@dataclass(frozen=True)
class PairedMark:
    canonical: tuple
    mark: tuple

    def decode(self):
        return pair_action(self.mark,self.canonical)

    def transport(self,g):
        k=tuple(h for h in GROUP if pair_action(h,self.canonical)==self.canonical)
        return PairedMark(self.canonical,min(mul(mul(g,self.mark),h) for h in k))


def encode_pair(pair):
    pair=tuple(tuple(v) for v in pair)
    c=max(pair_action(g,pair) for g in GROUP)
    return PairedMark(c,min(g for g in GROUP if pair_action(g,c)==pair))


def indexed_pair_step(state,x,transition):
    """For jointly equivariant tuple transition with paired output and history."""
    g=state.mark
    y,w=transition(act(inv(g),tuple(x)),state.canonical)
    return pair_action(g,y),encode_pair(w).transport(g)


@dataclass
class PairedStep:
    branches: torch.Tensor  # [batch,2,d]
    history: torch.Tensor   # [batch,experts,2,d], authoritative branch storage
    ids: torch.Tensor
    weights: torch.Tensor
    signs: torch.Tensor     # [batch,topk,2]

    @property
    def mean(self): return (self.branches[:,0]+self.branches[:,1])/2
    @property
    def residual(self): return (self.branches[:,0]-self.branches[:,1])/2
    @property
    def history_mean(self): return (self.history[:,:,0]+self.history[:,:,1])/2
    @property
    def history_residual(self): return (self.history[:,:,0]-self.history[:,:,1])/2


class PairedHistoryMoE(nn.Module):
    """Use original router on mean; execute original experts on both histories.

    New semantics: chart fallback is cyclic/reflection-paired, cap each branch
    BEFORE merging. Authoritative storage is branch pair (avoid floating-point
    mean/difference roundtrip). Full G-equivariance requires the separately
    proved source parameter contract; not enforced by this wrapper.
    """
    def __init__(self, source):
        super().__init__()
        if source.d_model<3 or source.d_model%3 or source.route_geometry!='c2_c3_residual':
            raise ValueError('requires coherent R4 source with complete triads')
        self.source=source

    def step(self,x,history,train_sign=False):
        s=self.source; batch=x.shape[0]; d=s.d_model
        if x.ndim!=2 or x.shape[-1]!=d or history.shape!=(batch,s.num_experts,2,d):
            raise ValueError('expected activation [B,d] and paired history [B,E,2,d]')
        mean=(history[:,:,0]+history[:,:,1])/2
        ids,weights=s.select_source_experts(x,mean)
        outputs=[x.new_zeros(batch,d),x.new_zeros(batch,d)]
        next_h=history.clone(); signs=x.new_zeros(batch,s.top_k,2)
        for e in range(s.num_experts):
            pos=(ids==e).nonzero(as_tuple=False)
            if not pos.numel():continue
            rows,slots=pos.unbind(1); raw=x.index_select(0,rows)
            for b in (0,1):
                h=history[rows,e,b]; z=torch.cat((raw,h),-1)
                score=2*torch.nn.functional.linear(z,s.orientation.weight,None)
                fallback=reflection_chart(z)[...,b:b+1]
                sign=torch.where(score==0,fallback,torch.sign(score))
                if train_sign:
                    soft=torch.tanh(score/s.temperature)
                    sign=sign.detach()+(soft-soft.detach())
                cx,ch=sign*raw,sign*h
                yc=s.experts[e](torch.cat((cx,ch),-1))
                hc=s.transitions[e](torch.cat((ch,cx,yc),-1))
                outputs[b]=outputs[b].index_add(0,rows,sign*yc*weights[rows,slots,None])
                next_h[rows,e,b]=s.cap_history(sign*hc)
                signs[rows,slots,b]=sign.squeeze(-1).detach()
        return PairedStep(torch.stack(outputs,1),next_h,ids,weights,signs)
