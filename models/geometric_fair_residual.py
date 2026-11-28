"""Common-function residual arms; original Leech-LILA: Anatolii Kornienko.
DASHI extension. Ternary coefficients in real orthonormal bases, not dense trit weights.
"""
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F
from .geometric_cross_triads import basis


def block_frame(mode, null=False, seed=41):
    """Real Frobenius-orthonormal frame with matched independent rank."""
    if mode=='entry':
        g=torch.eye(9,dtype=torch.float64)
    else:
        g=basis(mode,dtype=torch.float64).reshape(-1,9).T
    rank=g.shape[1]
    if null:
        g=torch.randn(9,rank,dtype=torch.float64,generator=torch.Generator().manual_seed(seed))
    q,_=torch.linalg.qr(g,mode='reduced')
    return q.T.reshape(rank,3,3)


class FairResidualLinear(nn.Module):
    def __init__(self,source,frame,ternary=True,scale=.25):
        super().__init__()
        o,i=source.weight.shape
        if o%3 or i%3:raise ValueError('complete triads required')
        self.register_buffer('base',source.weight.detach().clone())
        self.register_buffer('frame',frame.to(source.weight))
        self.coefficients=nn.Parameter(source.weight.new_zeros(o//3,i//3,len(frame)))
        self.bias=nn.Parameter(source.bias.detach().clone()) if source.bias is not None else None
        self.ternary=ternary;self.scale=scale
    def effective(self):
        c=self.coefficients
        if not self.ternary:return c
        q=c.round().clamp(-1,1)
        return q.detach()+(c-c.detach()) if self.training else q
    def delta(self):
        return self.scale*torch.einsum('oik,kab->oaib',self.effective(),self.frame).reshape_as(self.base)
    def forward(self,x):return F.linear(x,self.base+self.delta(),self.bias)


def fair_arm(source,mode,null=False,ternary=True,seed=41):
    """Copy source, replace only expert/transition linears; shared dense base."""
    model=deepcopy(source);frame=block_frame(mode,null,seed)
    for name,layer in list(model.named_modules()):
        if isinstance(layer,nn.Linear) and name.split('.')[0] in ('experts','transitions'):
            parent,key=name.rsplit('.',1)
            setattr(model.get_submodule(parent),key,FairResidualLinear(layer,frame,ternary))
    return model
