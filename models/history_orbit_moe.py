"""Learned sparse mixture of history-bearing signed-orbit expert families.

Each expert has independent learned FFN and history transition.
Orientation is an involutive chart *within* an expert family, not an expert ID.
The router reads each expert's retained state. Selected expert histories
are updated independently, unselected histories remain unchanged.
This prototype is additive: it does not replace the original Leech FFN.
"""
import torch
from torch import nn
from models.trialectic_wave_prior import (
    trialectic_router_invariants, directed_trialectic_invariants)
from models.punctured_nine_wave_prior import (
    pair_wave_router_features,punctured_blocks_to_wave_blocks)


class HistoryOrbitMoE(nn.Module):
    def __init__(self, d_model: int = 384, num_experts: int = 4,
                 top_k: int = 2, expert_width: int | None = None,
                 temperature: float = 1.0, route_geometry: str = "raw",
                 transition_rank: int | None = None):
        super().__init__()
        if d_model < 1 or num_experts < 2 or not 1 <= top_k <= num_experts:
            raise ValueError("invalid expert configuration")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if transition_rank is not None and not 1 <= transition_rank <= d_model:
            raise ValueError("transition_rank must be between 1 and d_model")
        if route_geometry not in ("raw", "c2_invariant", "c2c3_wave", "punctured9_wave"):
            raise ValueError("unknown route_geometry: raw, c2_invariant, c2c3_wave, punctured9_wave")
        if route_geometry == "c2c3_wave" and d_model % 3:
            raise ValueError("C3 wave router requires hidden width divisible by three")
        if route_geometry == "punctured9_wave" and d_model % 24:
            raise ValueError("punctured nine-wave router requires complete 24D blocks")
        self.d_model, self.num_experts, self.top_k = d_model, num_experts, top_k
        self.temperature = temperature
        self.route_geometry = route_geometry
        self.transition_rank = transition_rank
        width = expert_width or (4 * d_model // top_k)
        self.router = nn.Linear(
            3*d_model//2 if route_geometry == "punctured9_wave" else 2*d_model, 1)
        self.orientation = nn.Linear(2*d_model, 1)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(2*d_model, width), nn.GELU(),
                          nn.Linear(width, d_model))
            for _ in range(num_experts)
        ])
        def transition():
            if transition_rank is None:
                return nn.Linear(3*d_model, d_model)
            # 24D aligned rank is an *ablation*. This is a low-rank
            # factorisation, not a proven Leech automorphism.
            return nn.Sequential(
                nn.Linear(3*d_model, transition_rank, bias=False),
                nn.Linear(transition_rank, d_model))
        self.transitions = nn.ModuleList([
            transition() for _ in range(num_experts)
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
        # The opt-in C2 route depends on invariant features.  Geometry
        # modulates the learning/routing architecture, not only bytes on wire.
        # This preserves expert IDs and mixture weights under (x,H)->(-x,-H).
        if self.route_geometry == "c2_invariant":
            route_features = features.abs()
        elif self.route_geometry == "c2c3_wave":
            # The learned route sees both unary wave magnitudes and
            # *ordered* interaction channels (self/cyclic/anti-cyclic).
            # Argumentative direction is not represented by a sign flip.
            route_features = torch.cat((
                trialectic_router_invariants(features[..., :self.d_model]),
                directed_trialectic_invariants(
                    features[..., :self.d_model],
                    features[..., self.d_model:])), dim=-1)
        elif self.route_geometry == "punctured9_wave":
            # 24=3*(9-1), four real Fourier character pairs per lane.
            # Include signed ordered dot/wedge relations to each expert's
            # history without pretending disagreement means negation.
            route_features = pair_wave_router_features(
                punctured_blocks_to_wave_blocks(features[..., :self.d_model]),
                punctured_blocks_to_wave_blocks(features[..., self.d_model:]),
                compact=True)
        else:
            route_features = features
        logits = self.router(route_features).squeeze(-1) + self.expert_bias
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
            if self.route_geometry != "raw":
                # This is the bias-free odd part of the learned orientation
                # scorer. An exact zero score on a nonzero state needs an
                # antipodally consistent tie-break (first nonzero component).
                score = 2 * torch.nn.functional.linear(z, self.orientation.weight, None)
                first = (z != 0).to(torch.int64).argmax(dim=-1, keepdim=True)
                fallback = torch.sign(z.gather(-1, first))
                sign = torch.where(score == 0, fallback, torch.sign(score))
                # At the unique fixed point z=0, sign=0 and the selected
                # output/history remain zero, as required by oddness.
            else:
                score = self.orientation(z)-self.orientation(-z)
                sign = torch.where(score >= 0,torch.ones_like(score),-torch.ones_like(score))
            if train_sign:
                soft = torch.tanh(score/self.temperature)
                sign = sign.detach()+(soft-soft.detach())
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

    def execute_selected(self, x: torch.Tensor, selected_ids: torch.Tensor,
                         selected_histories: torch.Tensor,
                         selected_weights: torch.Tensor | None = None,
                         train_sign: bool = False):
        """Remote selected-only expert execution: NO routing re-evaluation.

        The source has already computed the top-k route and weights using
        all E histories. This consumer only needs x and the k selected
        histories. A receiver MUST verify source routing authority, model
        and history revisions before accepting these externally supplied
        IDs/weights; this method is not authentication.

        Returns weighted total, selected updated histories and sign charts.
        It never reads, changes or invents inactive expert histories.
        """
        if x.ndim!=2 or x.shape[-1]!=self.d_model:
            raise ValueError("selected execution requires [batch,d] activation")
        b=x.shape[0]
        k=selected_ids.shape[-1] if selected_ids.ndim==2 else 0
        if (selected_ids.shape!=(b,self.top_k) or
            selected_histories.shape!=(b,self.top_k,self.d_model) or
            (selected_weights is not None and
             selected_weights.shape!=(b,self.top_k)) or
            selected_ids.dtype!=torch.long):
            raise ValueError("selected IDs/histories/weights shape mismatch")
        if (bool((selected_ids<0).any()) or
            bool((selected_ids>=self.num_experts).any()) or
            bool((selected_ids.sort(dim=1).values[:,1:] ==
                  selected_ids.sort(dim=1).values[:,:-1]).any())):
            raise ValueError("invalid duplicate or out-of-range selected expert")
        result=x.new_zeros(x.shape)
        unweighted=x.new_zeros(b,k,self.d_model)
        updated=selected_histories.clone()
        signs=x.new_zeros(b,k)
        # Match original step: iterate by expert ID rather than top-k slot.
        for e in range(self.num_experts):
            positions=(selected_ids==e).nonzero(as_tuple=False)
            if positions.numel()==0:
                continue
            rows,slots=positions.unbind(1)
            raw_x=x.index_select(0,rows)
            h=selected_histories[rows,slots]
            z=torch.cat((raw_x,h),dim=-1)
            if self.route_geometry!="raw":
                score=2*torch.nn.functional.linear(z,self.orientation.weight,None)
                first=(z!=0).to(torch.int64).argmax(dim=-1,keepdim=True)
                fallback=torch.sign(z.gather(-1,first))
                sign=torch.where(score==0,fallback,torch.sign(score))
            else:
                score=self.orientation(z)-self.orientation(-z)
                sign=torch.where(score>=0,torch.ones_like(score),-torch.ones_like(score))
            if train_sign:
                soft=torch.tanh(score/self.temperature)
                sign=sign.detach()+(soft-soft.detach())
            cx,ch=sign*raw_x,sign*h
            yc=self.experts[e](torch.cat((cx,ch),dim=-1))
            hc=self.transitions[e](torch.cat((ch,cx,yc),dim=-1))
            y=sign*yc
            new_h=sign*hc
            unweighted[rows,slots]=y
            if selected_weights is not None:
                result=result.index_add(0,rows,y*selected_weights[rows,slots,None])
            updated[rows,slots]=new_h
            signs[rows,slots]=sign.squeeze(-1).detach()
        if selected_weights is None:
            # Correct remote consumer without routing weights: source owns
            # the mixture, peer returns each selected unweighted output.
            return unweighted,updated,signs
        return result,updated,signs

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
