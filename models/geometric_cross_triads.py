"""Opt-in cross-triad group algebra; original Leech-LILA: Anatolii Kornienko.
DASHI extension, BitNet alphabet donor. No upstream CUDA port.
"""
import torch
from torch import nn
from torch.nn import functional as F


def basis(mode, *, device=None, dtype=None):
    I = torch.eye(3, device=device, dtype=dtype)
    R = I.roll(-1, 0)
    J = I.flip(0)
    C = [I, R, R @ R]
    if mode == 'c3':
        return torch.stack(C)
    S = C + [J, R @ J]  # omit R^2 J: exact gauge
    if mode == 's3':
        return torch.stack(S)
    if mode != 'full':
        raise ValueError('mode must be c3, s3 or full')
    u = I.new_tensor([1, 1, 1]); p = I.new_tensor([1, -1, 0]); q = I.new_tensor([0, 1, -1])
    return torch.stack(S + [u[:,None]*p, u[:,None]*q, p[:,None]*u, q[:,None]*u])


class CrossTriadBitLinear(nn.Module):
    """All input/output triad pairs; fixed-scale ternary STE or real reference.
    Five S3 coordinates fix one group-algebra gauge, not all six-trit words.
    Full mode adds four explicit DC/phase bridges; expanded entries can exceed 1.
    """
    def __init__(self, in_features, out_features, mode='c3', scale=1., bias=True, ternary=True, *, device=None, dtype=None):
        super().__init__()
        if min(in_features, out_features) < 3 or in_features % 3 or out_features % 3:
            raise ValueError('complete positive triads required')
        if not 0 < scale < float('inf'):
            raise ValueError('finite positive scale required')
        self.in_features, self.out_features = in_features, out_features
        self.mode, self.scale, self.ternary = mode, float(scale), bool(ternary)
        g = basis(mode, device=device, dtype=dtype)
        self.register_buffer('generators', g, persistent=False)
        self.coefficients = nn.Parameter(g.new_zeros(out_features//3, in_features//3, len(g)))
        with torch.no_grad():
            k = min(in_features, out_features)//3
            i = torch.arange(k, device=g.device)
            self.coefficients[i,i,0] = 1
        self.bias = nn.Parameter(g.new_zeros(out_features)) if bias else None

    def effective_coefficients(self):
        c = self.coefficients
        if not self.ternary:
            return c
        q = c.round().clamp(-1,1)
        return q.detach()+(c-c.detach()) if self.training else q

    def dense_weight(self):
        return self.scale * torch.einsum('oik,kab->oaib', self.effective_coefficients(), self.generators).reshape(self.out_features,self.in_features)

    def forward(self,x):
        if x.shape[-1] != self.in_features:
            raise ValueError('wrong input width')
        t=x.reshape(*x.shape[:-1],self.in_features//3,3)
        z=torch.einsum('oik,kab,...ib->...oa',self.effective_coefficients(),self.generators,t)
        y=(self.scale*z).flatten(-2)
        return y if self.bias is None else y+self.bias

    @classmethod
    def from_linear(cls, source, coefficients, mode='c3', scale=1., ternary=True):
        """Explicit witness adapter; exact stored dense equality or reject.
        Not a search for coefficients, no silent projection of dense checkpoint.
        Parity in real arithmetic; accumulation ordering may differ in hardware.
        """
        if not isinstance(source,nn.Linear):
            raise TypeError('expected nn.Linear')
        target=cls(source.in_features,source.out_features,mode,scale,source.bias is not None,ternary,device=source.weight.device,dtype=source.weight.dtype)
        if coefficients.shape != target.coefficients.shape:
            raise ValueError('coefficient witness shape mismatch')
        with torch.no_grad():
            target.coefficients.copy_(coefficients)
            if target.bias is not None:
                target.bias.copy_(source.bias)
            if not torch.equal(target.dense_weight(),source.weight):
                raise ValueError('source weight does not equal declared geometric witness')
        target.train(source.training)
        return target
