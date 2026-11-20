"""Original directed two-channel codec, separate from fingerprint owners.

DASHI #1526 X369DirectedRouterConditionalCodecExact / CenteredDirectedConsumerMDLExact.
Anatolii Kornienko attribution retained; no F5 access.
Exact-real point admission is not native arithmetic or training certification.
"""
from fractions import Fraction
import torch
from torch.nn.functional import linear

def components(x,h,w,*,compute_dc=True):
    if x.ndim!=2 or h.ndim!=3 or h.shape[0]!=x.shape[0] or h.shape[-1]!=x.shape[-1] or x.shape[-1]%3 or w.shape!=(1,x.shape[-1]):
        raise ValueError("expected [B,3T], [B,E,3T], [1,3T]")
    a,b,c=x.reshape(x.shape[0],1,-1,3).unbind(-1)
    A,B,C=h.reshape(*h.shape[:2],-1,3).unbind(-1)
    ws,wf,wr=w.reshape(-1,3).unbind(-1)
    s=a*A+b*B+c*C
    f=a*B+b*C+c*A
    retained=((ws-wr)*s+(wf-wr)*f).sum(-1)
    if not compute_dc:
        return retained,None
    dc=(a+b+c)*(A+B+C)
    return retained,(wr*dc).sum(-1)

def exact_common_dc(x,h,w):
    """Audit actual stored float values as exact dyadics, not rounded GPU sums."""
    # Host-side reference audit only; deliberately outside timed GPU paths.
    def q(v): return Fraction(float(v))
    if not (torch.isfinite(x).all() and torch.isfinite(h).all() and torch.isfinite(w).all()):
        return [False]*x.shape[0]
    xs=x.detach().cpu().tolist(); hs=h.detach().cpu().tolist()
    ws=list(map(q,w.detach().cpu().tolist()[0]))
    out=[]
    for row,experts in zip(xs,hs):
        xr=list(map(q,row))
        coeff=[sum(xr[t:t+3])*ws[t+2] for t in range(0,len(xr),3)]
        totals=[]
        for expert in experts:
            he=list(map(q,expert))
            totals.append(sum(k*sum(he[3*t:3*t+3]) for t,k in enumerate(coeff)))
        out.append(all(v==totals[0] for v in totals))
    return out

def codec_logits(module,x,h,omit=False):
    from models.trialectic_wave_prior import rotate3,trialectic_router_invariants
    if module.route_geometry!="c2_c3_residual" or getattr(module,"route_probe_action",None) is not None:
        raise ValueError("codec only owns unmodified coherent R4")
    if omit and torch.is_grad_enabled():
        raise RuntimeError("pointwise DC equality does not certify gradients")
    phase=rotate3(x)
    unary,directed=module.c3_residual_weight.split(module.d_model,dim=-1)
    retained,dc=components(phase,h,directed,compute_dc=not omit)
    features=torch.cat((x[:,None,:].expand(-1,module.num_experts,-1),h),dim=-1)
    base=module.router(features.abs()).squeeze(-1)+module.expert_bias
    u=linear(trialectic_router_invariants(phase),unary)
    return base+(u+retained if omit else u+(retained+dc))

def route(module,logits):
    if module.router_selection=="topk":
        vals,ids=torch.topk(logits,module.top_k,dim=-1)
    elif module.router_selection=="sort":
        vals,ids=torch.sort(logits,dim=-1,descending=True)
        vals,ids=vals[...,:module.top_k],ids[...,:module.top_k]
    else: raise ValueError("unknown selection")
    return ids,torch.softmax(vals,dim=-1)

def select_two_with_dc(module,x,h):
    if module.route_geometry!="c2_c3_residual" or getattr(module,"route_probe_action",None) is not None:
        return module.select_source_experts(x,h)
    return route(module,codec_logits(module,x,h))

@torch.no_grad()
def audit_omission(module,x,h):
    """Shadow-mode D: exact DC premise + observed native output equality.
    Returns original route on rejection; never certifies native rounding.
    This is not a production compression path or a timed speedup claim.
    """
    from models.trialectic_wave_prior import rotate3
    source=module.select_source_experts(x,h)
    if module.route_geometry!="c2_c3_residual" or getattr(module,"route_probe_action",None) is not None:
        return source,{"admitted":False,"reason":"unsupported"}
    _,w=module.c3_residual_weight.split(module.d_model,dim=-1)
    exact=exact_common_dc(rotate3(x),h,w)
    if not all(exact):
        return source,{"admitted":False,"reason":"unequal exact weighted DC","exact_rows":exact}
    candidate=route(module,codec_logits(module,x,h,omit=True))
    ok=all(torch.equal(a,b) for a,b in zip(source,candidate))
    return (candidate if ok else source),{"admitted":ok,"exact_rows":exact,
        "reason":"observed route and weights equal" if ok else "native rounding discrepancy"}
