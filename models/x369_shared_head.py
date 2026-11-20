"""Opt-in R4 directed score fusion; original fingerprint owners untouched."""
import torch
from torch.nn.functional import linear


def fused_directed_score(x, history, weight):
    if x.ndim != 2 or history.ndim != 3 or history.shape[0] != x.shape[0] or history.shape[-1] != x.shape[-1] or x.shape[-1] % 3 or weight.shape != (1, x.shape[-1]):
        raise ValueError('expected common [B,D], [B,E,D], [1,D]')
    a,b,c=x.reshape(x.shape[0],-1,3).unbind(-1)
    s,f,r=weight.reshape(-1,3).unbind(-1)
    z=torch.stack((s*a+r*b+f*c,f*a+s*b+r*c,r*a+f*b+s*c),-1).reshape_as(x)
    return torch.bmm(history,z.unsqueeze(-1)).squeeze(-1)


def fused_residual_score(x,history,weight):
    from models.trialectic_wave_prior import trialectic_router_invariants
    unary,directed=weight.split(x.shape[-1],dim=-1)
    return linear(trialectic_router_invariants(x),unary)+fused_directed_score(x,history,directed)


def select_fused_r4(module,x,history):
    """Explicit alternate consumer, never silently replaces source routing."""
    from models.trialectic_wave_prior import rotate3
    if module.route_geometry != 'c2_c3_residual' or getattr(module,'route_probe_action',None) is not None:
        return module.select_source_experts(x,history)
    if x.ndim != 2 or x.shape[-1]!=module.d_model or history.shape!=(x.shape[0],module.num_experts,module.d_model):
        raise ValueError('invalid source input shape')
    features=torch.cat((x[:,None,:].expand(-1,module.num_experts,-1),history),dim=-1)
    logits=module.router(features.abs()).squeeze(-1)+module.expert_bias
    logits=logits+fused_residual_score(rotate3(x),history,module.c3_residual_weight)
    if module.router_selection=='topk':
        values,ids=torch.topk(logits,module.top_k,dim=-1)
    else:
        values,ids=torch.sort(logits,dim=-1,descending=True)
        values,ids=values[...,:module.top_k],ids[...,:module.top_k]
    return ids,torch.softmax(values,dim=-1)
