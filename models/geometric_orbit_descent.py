"""Opt-in whole-source signed orbit descent, original Leech-LILA: Anatolii Kornienko.

Canonicalize the joint activation/history, execute the ORIGINAL sparse step once,
project continuous results to the stabilizer-fixed space, and decode the frame.
Ideal-real equivariance by construction; not same-function checkpoint migration,
native packed performance, learned recognition, or a gradient/rounding theorem.
"""
from dataclasses import dataclass
import torch
from torch import nn
from .geometric_reflection_fibre import GROUP, inv, mul
from .geometric_paired_history import PairedHistoryMoE


def act_tensor(g, x, paired=False):
    """Original reflection-fibre convention: R rolls left, J reverses triad."""
    if x.shape[-1]<3 or x.shape[-1]%3:
        raise ValueError("complete triads required")
    k,b,a=g
    if paired:
        if x.shape[-2]!=2:raise ValueError("paired branch axis required")
        if b:x=x.flip(-2)
    z=x.reshape(*x.shape[:-1],-1,3)
    if b:z=z.flip(-1)
    z=z.roll(-k,-1).reshape_as(x)
    return -z if a else z


def flat_joint(x,h):
    return torch.cat((x.flatten(1),h.flatten(1)),dim=1)


def batch_where(mask,a,b):
    return torch.where(mask.reshape(-1,*([1]*(a.ndim-1))),a,b)


def canonical_joint(x,h,paired):
    """Return canonical input, world->local mark, and full stabilizer mask."""
    if not torch.isfinite(x).all() or not torch.isfinite(h).all():
        raise ValueError("finite orbit carrier required")
    cx,ch=x,h
    mark=torch.zeros(len(x),device=x.device,dtype=torch.long)
    for i,g in enumerate(GROUP[1:],1):
        u,v=act_tensor(g,x),act_tensor(g,h,paired)
        p,q=flat_joint(u,v),flat_joint(cx,ch)
        different=p!=q
        first=different.to(torch.int64).argmax(1,keepdim=True)
        greater=different.any(1)&(p.gather(1,first)>q.gather(1,first)).squeeze(1)
        cx,ch=batch_where(greater,u,cx),batch_where(greater,v,ch)
        mark=torch.where(greater,i,mark)
    stabilizer=torch.stack([((act_tensor(g,cx)==cx).flatten(1).all(1)&
        (act_tensor(g,ch,paired)==ch).flatten(1).all(1)) for g in GROUP],dim=1)
    return cx,ch,mark,stabilizer


def project_stabilizer(x,mask,paired=False):
    """Reynolds projection in local frame; mask is from the exact input orbit."""
    total=torch.zeros_like(x)
    for i,g in enumerate(GROUP):
        total=total+batch_where(mask[:,i],act_tensor(g,x,paired),torch.zeros_like(x))
    count=mask.sum(1).to(x.dtype).reshape(-1,*([1]*(x.ndim-1)))
    return total/count


def apply_group_indices(x,indices,paired=False,inverse=False):
    if indices.shape!=(len(x),) or indices.dtype!=torch.long or not ((indices>=0)&(indices<len(GROUP))).all():
        raise ValueError("valid per-row group indices required")
    out=torch.zeros_like(x)
    for i,g in enumerate(GROUP):
        out=batch_where(indices==i,act_tensor(inv(g) if inverse else g,x,paired),out)
    return out


def decode_frame(x,mark,paired=False):
    return apply_group_indices(x,mark,paired,inverse=True)


def conjugate_control(g,control):
    """Operations are a conjugation carrier, not a polar activation."""
    table=torch.tensor([GROUP.index(mul(mul(g,h),inv(g))) for h in GROUP],
        device=control.device,dtype=torch.long)
    return table[control]


def canonical_controlled(control,h,paired):
    if control.dtype!=torch.long or control.shape!=(len(h),) or not ((control>=0)&(control<len(GROUP))).all() or not torch.isfinite(h).all():
        raise ValueError("finite history and valid operation required")
    cc,ch=control,h
    mark=torch.zeros_like(control)
    for i,g in enumerate(GROUP[1:],1):
        u,v=conjugate_control(g,control),act_tensor(g,h,paired)
        p,q=v.flatten(1),ch.flatten(1);different=p!=q
        first=different.to(torch.int64).argmax(1,keepdim=True)
        greater=(u>cc)|((u==cc)&different.any(1)&(p.gather(1,first)>q.gather(1,first)).squeeze(1))
        cc,ch=torch.where(greater,u,cc),batch_where(greater,v,ch)
        mark=torch.where(greater,i,mark)
    stabilizer=torch.stack([(conjugate_control(g,cc)==cc)&
        (act_tensor(g,ch,paired)==ch).flatten(1).all(1) for g in GROUP],1)
    return cc,ch,mark,stabilizer


@dataclass
class OrbitStep:
    output: torch.Tensor
    history: torch.Tensor
    ids: torch.Tensor
    weights: torch.Tensor
    local_signs: torch.Tensor
    world_to_local: torch.Tensor
    stabilizer: torch.Tensor
    raw_local_output: torch.Tensor
    raw_local_history: torch.Tensor
    local_output: torch.Tensor
    local_history: torch.Tensor
    local_control: torch.Tensor | None = None


class OrbitDescentHistoryMoE(nn.Module):
    """One original source step, shared parameters, stabilizer-safe descent.

    Input activation and every expert history transform together. Paired history
    also swaps branches under reflection. IDs/weights and signs are canonical
    source observations, not invented world-frame orientation scores. Raw and
    projected local results remain explicit. The projected world history is
    the recurrence of this NEW architecture, not the original raw recurrence.
    """
    def __init__(self,source,paired=False):
        super().__init__()
        if source.d_model<3 or source.d_model%3 or source.route_geometry!='c2_c3_residual':
            raise ValueError("coherent R4 source and complete triads required")
        self.paired=bool(paired)
        self.stepper=PairedHistoryMoE(source) if self.paired else source

    @property
    def source(self):
        return self.stepper.source if self.paired else self.stepper

    def step(self,x,history,train_sign=False):
        s=self.source
        if getattr(s,'route_probe_action',None) is not None:
            raise ValueError("unintervened original source required")
        shape=(len(x),s.num_experts,2,s.d_model) if self.paired else (len(x),s.num_experts,s.d_model)
        if x.ndim!=2 or not len(x) or x.shape[1]!=s.d_model or history.shape!=shape:
            raise ValueError("incorrect activation/history shape")
        cx,ch,mark,stab=canonical_joint(x,history,self.paired)
        return self._execute(cx,ch,mark,stab,history,train_sign)

    def step_controlled(self,control,history,embed,train_sign=False):
        """Canonicalize (operation,history), then embed operation locally.
        An invariant literal tag may be captured by embed; it must not read
        the world mark or untransformed control.
        """
        s=self.source
        shape=(len(history),s.num_experts,2,s.d_model) if self.paired else (len(history),s.num_experts,s.d_model)
        if not len(history) or history.shape!=shape or getattr(s,'route_probe_action',None) is not None:
            raise ValueError("unintervened original history required")
        cc,ch,mark,stab=canonical_controlled(control,history,self.paired)
        cx=embed(cc)
        if cx.shape!=(len(history),s.d_model) or not torch.isfinite(cx).all():
            raise ValueError("finite local embedding required")
        return self._execute(cx,ch,mark,stab,history,train_sign,cc)

    def _execute(self,cx,ch,mark,stab,history,train_sign,control=None):
        s=self.source
        if self.paired:
            o=self.stepper.step(cx,ch,train_sign=train_sign)
            y,h,ids,signs,weights=o.branches,o.history,o.ids,o.signs,o.weights
        else:
            y,h,ids,signs,weights=self.stepper.step(cx,ch,train_sign=train_sign)
        py,ph=project_stabilizer(y,stab,self.paired),project_stabilizer(h,stab,self.paired)
        wy,wh=decode_frame(py,mark,self.paired),decode_frame(ph,mark,self.paired)
        # Original inactive expert identity, including floating representation.
        selected=torch.zeros((len(history),s.num_experts),device=history.device,dtype=torch.bool).scatter_(1,ids,True)
        wh=torch.where(selected.reshape(len(history),s.num_experts,*([1]*(history.ndim-2))),wh,history)
        return OrbitStep(wy,wh,ids,weights,signs,mark,stab,y,h,py,ph,control)
