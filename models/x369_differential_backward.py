"""Exact-real shared-head VJP with differential DC; first-order only."""
import torch
from torch.autograd.function import once_differentiable

class DirectedVJP(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,h,w,relative):
        b,d=x.shape
        a,c0,c=x.reshape(b,-1,3).unbind(-1)
        s,f,r=w.reshape(-1,3).unbind(-1)
        z=torch.stack((s*a+r*c0+f*c,f*a+s*c0+r*c,r*a+f*c0+s*c),-1).reshape_as(x)
        y=torch.bmm(h,z.unsqueeze(-1)).squeeze(-1)
        ctx.save_for_backward(x,h,w)
        ctx.relative=relative
        return y-y[:,:1] if relative else y

    @staticmethod
    @once_differentiable
    def backward(ctx,g):
        x,h,w=ctx.saved_tensors
        if ctx.relative:
            # No assumption that the native softmax VJP sums to zero.
            g=torch.cat((-g[:,1:].sum(-1,keepdim=True),g[:,1:]),-1)
        b,d=x.shape
        a,b0,c=x.reshape(b,-1,3).unbind(-1)
        s,f,r=w.reshape(-1,3).unbind(-1)
        z=torch.stack((s*a+r*b0+f*c,f*a+s*b0+r*c,r*a+f*b0+s*c),-1).reshape_as(x)
        H=torch.bmm(g.unsqueeze(1),h).squeeze(1)
        A,B,C=H.reshape(b,-1,3).unbind(-1)
        dx=torch.stack((s*A+f*B+r*C,r*A+s*B+f*C,f*A+r*B+s*C),-1).reshape_as(x)
        dh=g.unsqueeze(-1)*z.unsqueeze(1)
        # Explicit retained and DC contributions to reverse-head VJP.
        self_term=a*A+b0*B+c*C
        forward_term=a*B+b0*C+c*A
        dc=(a+b0+c)*(A+B+C)
        dw=torch.stack((self_term,forward_term,dc-self_term-forward_term),-1).sum(0).reshape_as(w)
        return dx,dh,dw,None

def directed_vjp(x,h,w,*,relative=False):
    if x.ndim!=2 or x.shape[0]<1 or x.shape[-1]<3 or x.shape[-1]%3 or h.ndim!=3 or h.shape[1]<1 or h.shape[0]!=x.shape[0] or h.shape[-1]!=x.shape[-1] or w.shape!=(1,x.shape[-1]):
        raise ValueError('expected [B,3T], [B,E,3T], [1,3T]')
    return DirectedVJP.apply(x,h,w,relative)

def select_vjp_r4(module,x,h):
    """Opt-in full-score learner; intervention fallback; no DC omission."""
    from models.trialectic_wave_prior import rotate3,trialectic_router_invariants
    from models.x369_two_channel_codec import route
    from torch.nn.functional import linear
    if module.route_geometry!="c2_c3_residual" or getattr(module,"route_probe_action",None) is not None:
        return module.select_source_experts(x,h)
    phase=rotate3(x)
    unary,w=module.c3_residual_weight.split(module.d_model,dim=-1)
    features=torch.cat((x[:,None,:].expand(-1,module.num_experts,-1),h),dim=-1)
    base=module.router(features.abs()).squeeze(-1)+module.expert_bias
    return route(module,base+(linear(trialectic_router_invariants(phase),unary)+directed_vjp(phase,h,w)))
