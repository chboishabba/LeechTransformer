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
    def initial(self,seed):
        h=signed_seed(seed,self.source.experts,self.source.width)
        return SplitState(seed[:,:6],h)
    def step(self,state,u,tag):
        raw=self.source.history.step_controlled(u,state.sparse,lambda c:self.source.control_embed(tag*len(GROUP)+c),train_sign=self.training)
        z=semantic_action(state.semantic,u)
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
