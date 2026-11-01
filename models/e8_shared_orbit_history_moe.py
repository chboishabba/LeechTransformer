"""Genuine shared canonical E8-history family for learned MoE ablations.

All E expert memories are explicitly derived from ONE shared canonical
history using signed-permutation E8 Weyl actions. Therefore cross-expert
history reuse is true *by construction*, not an inferred consequence of
per-expert sign equivariance. For every selected expert, its candidate
new history is inverted to the canonical chart; the next common state
is a deterministic route-weighted mean of selected candidate states.
The full E history family is regenerated from this one source state.

This changes model semantics, so use it as a separate trainable ablation.
It is NOT a lossless conversion of old independent expert histories and
does not prove Leech lattice / semantic argument equivariance.
"""
from __future__ import annotations
import torch
from torch import nn
from models.history_orbit_moe import HistoryOrbitMoE
from models.e8_orbit_actions import e8_orbit_transform,e8_inverse_action


class E8SharedOrbitHistoryMoE(nn.Module):
    def __init__(self,d_model:int=384,num_experts:int=4,top_k:int=2,
                 expert_width:int|None=None,
                 transition_rank:int|None=24,
                 route_geometry:str="e8_root_invariant",
                 share_expert_weights:bool=False):
        super().__init__()
        if d_model%8 or num_experts<2 or num_experts>4:
            raise ValueError("shared E8 orbit version supports 2–4 experts")
        self.d_model,self.num_experts,self.top_k=d_model,num_experts,top_k
        self.route_geometry=route_geometry
        self.transition_rank=transition_rank
        self.share_expert_weights=share_expert_weights
        self.core=HistoryOrbitMoE(d_model,num_experts,top_k,
            expert_width,route_geometry=route_geometry,
            transition_rank=transition_rank)
        self.action_names=("identity","swap01","flip01","negative8")[:num_experts]
        if share_expert_weights:
            # One learned canonical FFN/transition shared by all four
            # orientation/point-action charts. Expert-specific routing bias
            # and historical charts are still independent inputs.
            self.core.experts=nn.ModuleList([self.core.experts[0]]*num_experts)
            self.core.transitions=nn.ModuleList([self.core.transitions[0]]*num_experts)

    def expand_history(self,base:torch.Tensor)->torch.Tensor:
        if base.ndim!=2 or base.shape[-1]!=self.d_model:
            raise ValueError("canonical E8 memory must be [B,d]")
        return torch.stack([
            e8_orbit_transform(base,name) for name in self.action_names],dim=1)

    def extract_base(self,histories:torch.Tensor)->torch.Tensor:
        if histories.ndim!=3 or histories.shape[1:]!=(self.num_experts,self.d_model):
            raise ValueError("historical E8 family must be [B,E,d]")
        base=histories[:,0,:]
        if not torch.equal(histories,self.expand_history(base)):
            raise ValueError("histories are not the admitted canonical E8 orbit family")
        return base

    def step(self,x:torch.Tensor,histories:torch.Tensor,train_sign:bool=False):
        base=self.extract_base(histories)
        y,updated,ids,signs,weights=self.core.step(x,histories,train_sign)
        # Route-weighted canonical state combining the actually selected
        # update proposals. It is one deterministic causal recurrence.
        b=x.shape[0]
        candidates=[]
        for e,name in enumerate(self.action_names):
            inverse=e8_inverse_action(name)
            candidates.append(e8_orbit_transform(updated[:,e,:],inverse))
        proposals=torch.stack(candidates,dim=1)
        # Only selected expert updates contribute. Never average stale
        # unselected histories in place of a selected update.
        selected=proposals.gather(1,ids[:,:,None].expand(-1,-1,self.d_model))
        next_base=(selected*weights[:,:,None]).sum(dim=1)
        next_histories=self.expand_history(next_base)
        return y,next_histories,ids,signs,weights

    def execute_selected(self,x,ids,selected_histories,
                         selected_weights=None,train_sign=False):
        return self.core.execute_selected(
            x,ids,selected_histories,selected_weights,train_sign)

    def forward(self,tokens:torch.Tensor,history:torch.Tensor|None=None,
                train_sign:bool=False):
        if tokens.ndim!=3 or tokens.shape[-1]!=self.d_model or tokens.shape[1]<1:
            raise ValueError("expected [B,T,d]")
        b=tokens.shape[0]
        if history is None:
            history=self.expand_history(tokens.new_zeros(b,self.d_model))
        results=[];routes=[];signs=[];weights=[]
        for t in range(tokens.shape[1]):
            y,history,ids,sgn,w=self.step(tokens[:,t],history,train_sign)
            results.append(y);routes.append(ids);signs.append(sgn);weights.append(w)
        return (torch.stack(results,dim=1),history,
                torch.stack(routes,dim=1),torch.stack(signs,dim=1),
                torch.stack(weights,dim=1))
