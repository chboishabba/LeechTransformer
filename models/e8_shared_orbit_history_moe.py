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
        # Only the selected updates and original weights are needed to
        # reconstruct the canonical next state; inactive histories are
        # not copied or guessed from a remote worker.
        chosen=updated.gather(1,ids[:,:,None].expand(-1,-1,self.d_model))
        next_histories=self.reconcile_selected(ids,chosen,weights)
        return y,next_histories,ids,signs,weights

    def reconcile_selected(self,ids:torch.Tensor,selected_updates:torch.Tensor,
                           weights:torch.Tensor)->torch.Tensor:
        """Source-owned exact canonical merger after remote execution.

        Selected expert updates are pulled through their inverse E8
        charts, then pooled in canonical space using retained router
        weights. The merged canonical history is the SOLE authority.
        """
        b,k=ids.shape
        if (ids.dtype!=torch.long or k!=self.top_k or
            selected_updates.shape!=(b,k,self.d_model) or
            weights.shape!=(b,k)):
            raise ValueError("selected canonical merger layout mismatch")
        if (bool((ids<0).any()) or bool((ids>=self.num_experts).any()) or
            bool((ids.sort(-1).values[:,1:]==ids.sort(-1).values[:,:-1]).any())):
            raise ValueError("invalid selected orbit IDs")
        # Selected-only remote output is untrusted until the session
        # authenticates its revision. At this local boundary also reject
        # non-probability weights and nonfinite updates rather than
        # silently corrupting the authoritative canonical memory.
        if (not bool(torch.isfinite(weights).all()) or
            bool((weights<0).any()) or
            not bool(torch.allclose(weights.sum(-1),
                                    torch.ones_like(weights[:,0]),
                                    atol=1e-5,rtol=1e-5))):
            raise ValueError("invalid canonical merger mixture weights")
        if not bool(torch.isfinite(selected_updates).all()):
            raise ValueError("nonfinite selected canonical history update")
        canon=selected_updates.new_zeros(b,k,self.d_model)
        for e,name in enumerate(self.action_names):
            positions=(ids==e).nonzero(as_tuple=False)
            if positions.numel():
                rows,slots=positions.unbind(1)
                canon[rows,slots]=e8_orbit_transform(
                    selected_updates[rows,slots],e8_inverse_action(name))
        base=(canon*weights[:,:,None]).sum(1)
        return self.expand_history(base)

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
