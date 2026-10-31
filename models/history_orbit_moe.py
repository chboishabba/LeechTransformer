"""Learned sparse mixture of history-bearing signed-orbit expert families.

Each expert has independent learned FFN and history transition.
Orientation is an involutive chart *within* an expert family, not an expert ID.
The router reads each expert's retained state. Selected expert histories
are updated independently, unselected histories remain unchanged.
This prototype is additive: it does not replace the original Leech FFN.
"""
import torch
from torch import nn


class HistoryOrbitMoE(nn.Module):
    def __init__(self, d_model: int = 384, num_experts: int = 4,
                 top_k: int = 2, expert_width: int | None = None,
                 temperature: float = 1.0):
        super().__init__()
        if d_model < 1 or num_experts < 2 or not 1 <= top_k <= num_experts:
            raise ValueError("invalid expert configuration")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.d_model, self.num_experts, self.top_k = d_model, num_experts, top_k
        self.temperature = temperature
        width = expert_width or (4 * d_model // top_k)
        self.router = nn.Linear(2*d_model, 1)
        self.orientation = nn.Linear(2*d_model, 1)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(2*d_model, width), nn.GELU(),
                          nn.Linear(width, d_model))
            for _ in range(num_experts)
        ])
        self.transitions = nn.ModuleList([
            nn.Linear(3*d_model, d_model) for _ in range(num_experts)
        ])
        self.expert_bias = nn.Parameter(torch.zeros(num_experts))

    def step(self, x: torch.Tensor, history: torch.Tensor,
             train_sign: bool = False):
        if x.ndim != 2 or x.shape[1] != self.d_model:
            raise ValueError("x must have shape [batch,d_model]")
        batch = x.shape[0]
        if history.shape != (batch,self.num_experts,self.d_model):
            raise ValueError("history must have shape [batch,experts,d_model]")
        features = torch.cat((x[:,None,:].expand(-1,self.num_experts,-1),
                              history), dim=-1)
        logits = self.router(features).squeeze(-1) + self.expert_bias
        values, ids = torch.topk(logits,self.top_k,dim=-1)
        weights = torch.softmax(values,dim=-1)
        result = x.new_zeros(x.shape)
        next_hist = history.clone()
        signs = x.new_zeros(batch,self.top_k)
        for e in range(self.num_experts):
            positions = (ids == e).nonzero(as_tuple=False)
            if positions.numel() == 0:
                continue
            b_idx, slot = positions.unbind(1)
            raw_x = x.index_select(0,b_idx)
            old_h = history[b_idx,e]
            z = torch.cat((raw_x,old_h),dim=-1)
            score = self.orientation(z)-self.orientation(-z)
            sign = torch.where(score >= 0,torch.ones_like(score),-torch.ones_like(score))
            if train_sign:
                soft = torch.tanh(score/self.temperature)
                sign = sign.detach()+soft-soft.detach()
            canon_x,canon_h = sign*raw_x,sign*old_h
            y_c = self.experts[e](torch.cat((canon_x,canon_h),dim=-1))
            h_c = self.transitions[e](torch.cat((canon_h,canon_x,y_c),dim=-1))
            y = sign*y_c
            updated_h = sign*h_c
            contribution = y*weights[b_idx,slot,None]
            result = result.index_add(0,b_idx,contribution)
            next_hist[b_idx,e] = updated_h
            signs[b_idx,slot] = sign.squeeze(-1).detach()
        return result,next_hist,ids,signs,weights

    def forward(self, tokens: torch.Tensor, history: torch.Tensor | None = None,
                train_sign: bool = False):
        if tokens.ndim != 3 or tokens.shape[2] != self.d_model or tokens.shape[1] < 1:
            raise ValueError("tokens must have shape [batch,time,d_model]")
        batch = tokens.shape[0]
        if history is None:
            history = tokens.new_zeros(batch,self.num_experts,self.d_model)
        outputs,expert_ids,signs,weights = [],[],[],[]
        for time in range(tokens.shape[1]):
            y,history,ids,sgn,w = self.step(tokens[:,time],history,train_sign)
            outputs.append(y)
            expert_ids.append(ids)
            signs.append(sgn)
            weights.append(w)
        return (torch.stack(outputs,1),history,torch.stack(expert_ids,1),
                torch.stack(signs,1),torch.stack(weights,1))
