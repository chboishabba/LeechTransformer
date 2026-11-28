"""Explicit supplied-action reference, not independent learned task recognition.
Original Leech-LILA: Anatolii Kornienko; DASHI extension.
Preserve original sparse residual histories; no cap repair or forced routing.
"""
from dataclasses import dataclass
import torch
from torch import nn
from models.geometric_reflection_fibre import GROUP
from models.geometric_orbit_descent import apply_group_indices
from scripts.x369_source_geometry import SourceGeometryOrderedLearner,literal_controls,signed_seed
from scripts.x369_indexed_residual_closure import semantic_adjoint
from scripts.x369_constructive_finite_source import decode_history
from scripts.x369_typed_prototype_readout import TypedPrototypeReadout

def semantic_action(z,u):
    if z.ndim!=2 or z.shape[-1]!=6:raise ValueError("six coordinates required")
    m,d=z[:,:3],z[:,3:]
    return torch.cat((apply_group_indices(m,u),apply_group_indices(d,u)*torch.where((u//2)%2==1,-1,1)[:,None]),-1)

@dataclass
class SplitState:
    semantic: torch.Tensor
    sparse: torch.Tensor

@dataclass
class SplitStep:
    state: SplitState
    raw: object
    residual_logits: tuple

class SplitCarrierLearner(nn.Module):
    """The authoritative semantic register is outside sparse expert storage.
    Sparse history is retained whole, not asserted to lie in ker Q.
    Its kernel chart is an observer, not a destructive projection.
    """
    def __init__(self,source=None):
        super().__init__()
        self.source=source if source is not None else SourceGeometryOrderedLearner()
        self.prototype=TypedPrototypeReadout(persistent=False)
    def advance_semantic(self,z,u,tag):
        return semantic_action(z,u)
    def initial(self,seed):
        h=signed_seed(seed,self.source.experts,self.source.width)
        return SplitState(seed[:,:6],h)
    def step(self,state,u,tag):
        raw=self.source.history.step_controlled(u,state.sparse,lambda c:self.source.control_embed(tag*len(GROUP)+c),train_sign=self.training)
        z=self.advance_semantic(state.semantic,u,tag)
        return SplitStep(SplitState(z,raw.history),raw,self.source.raw_affine_consume(raw))
    def chart(self,state):
        h=state.sparse;q=torch.cat(decode_history(h),-1)
        r=h-semantic_adjoint(q,h)
        return state.semantic,q-state.semantic,r
    def reconstruct(self,z,e,r):
        return semantic_adjoint(z+e,r)+r
    def rollout(self,seed,ops):
        controls,tags=literal_controls(ops);state=self.initial(seed);trace=[]
        for t in range(controls.shape[1]):
            item=self.step(state,controls[:,t],tags[:,t]);trace.append(item);state=item.state
        if not trace:raise ValueError("empty word")
        return self.prototype(state.semantic[:,:3],state.semantic[:,3:]),trace
    def forward(self,seed,ops):
        return self.rollout(seed,ops)[0]


class LearnedGeneratorSplitCarrier(SplitCarrierLearner):
    """The semantic register uses only admitted elementary fitted coefficients.
    Signed permutations use gather/sign, not a dense float reduction.
    Sparse source remains an independent, fully retained learned carrier.
    """
    def __init__(self,matrices,source=None):
        from scripts.x369_learned_generator_weld import certify
        certify(matrices)
        super().__init__(source)
        index=torch.zeros(4,len(GROUP),6,dtype=torch.long)
        sign=torch.zeros_like(index);valid=torch.zeros(4,len(GROUP),dtype=torch.bool)
        for (letter,guard),W in matrices.items():
            t='abjn'.index(letter);u=4*((-guard)%3) if t<2 else 10*guard if t==2 else guard
            for j,row in enumerate(W):
                nz=[i for i,x in enumerate(row) if x]
                if len(nz)!=1 or abs(row[nz[0]])!=1:raise ValueError('signed permutation required')
                index[t,u,j]=nz[0];sign[t,u,j]=row[nz[0]]
            valid[t,u]=True
        self.register_buffer('learned_index',index)
        self.register_buffer('learned_sign',sign)
        self.register_buffer('learned_valid',valid)
    def advance_semantic(self,z,u,tag):
        if u.shape!=tag.shape or u.shape!=(len(z),) or u.dtype!=torch.long or tag.dtype!=torch.long or not ((u>=0)&(u<len(GROUP))&(tag>=0)&(tag<4)).all():
            raise ValueError('invalid literal')
        if not self.learned_valid[tag,u].all():raise ValueError('unidentified literal')
        return z.gather(1,self.learned_index[tag,u])*self.learned_sign[tag,u]
