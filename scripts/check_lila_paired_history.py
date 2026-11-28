from itertools import product
import torch
from torch import nn
import sympy as sp
from models.geometric_paired_history import (GROUP, mul, pair_action, encode_pair, indexed_pair_step, PairedHistoryMoE)

# Source-interface fixture, not imported original HistoryOrbitMoE. It implements
# the coherent directed routing formula and the original expert/cap interface.
class Fixture(nn.Module):
    def __init__(self,cap=None):
        super().__init__(); self.d_model=3; self.num_experts=4; self.top_k=2
        self.route_geometry='c2_c3_residual'; self.history_cap=cap
        self.orientation=nn.Linear(6,1,bias=False,dtype=torch.float64)
        self.experts=nn.ModuleList([nn.Sequential(nn.Linear(6,3,dtype=torch.float64),nn.GELU(),nn.Linear(3,3,dtype=torch.float64)) for _ in range(4)])
        self.transitions=nn.ModuleList([nn.Linear(9,3,dtype=torch.float64) for _ in range(4)])
        with torch.no_grad():
            self.orientation.weight.copy_(torch.tensor([[1,1,1,0,0,0]],dtype=torch.float64))
            for e in range(4):
                a,_,b=self.experts[e]; a.weight.zero_(); a.bias.zero_(); b.weight.copy_(torch.eye(3,dtype=torch.float64)); b.bias.zero_()
                if e==0:a.weight[:,3:]=torch.eye(3,dtype=torch.float64)
                t=self.transitions[e];t.weight.zero_();t.bias.zero_();t.weight[:,:3]=torch.eye(3,dtype=torch.float64);t.weight[:,6:]=torch.eye(3,dtype=torch.float64)
    def select_source_experts(self,x,h):
        # shifted head a=c=0,b=1, so K=I; strict bias keeps selected set for probes.
        logits=(x[:,None,:]*h).sum(-1)+x.new_tensor([12,4,-4,-12])
        val,ids=torch.topk(logits,2,dim=-1);return ids,torch.softmax(val,-1)
    def cap_history(self,h):
        if self.history_cap is None:return h
        return h*(self.history_cap/h.norm(dim=-1,keepdim=True).clamp_min(1e-30)).clamp(max=1)

def transform(g,x,paired=False):
    k,b,a=g
    if paired and b:x=x.flip(-2)
    if b:x=x.flip(-1)
    return (-1)**a*x.roll(-k,-1)

def main(factory=Fixture):
    # Full 729 signed pair states, stabilizer transport and compositions.
    count=0; classes=set()
    for v in product((-1,0,1),repeat=6):
        pair=(v[:3],v[3:]); e=encode_pair(pair);classes.add(e.canonical);assert e.decode()==pair
        for g in GROUP:
            assert e.transport(g)==encode_pair(pair_action(g,pair))
            for h in GROUP:
                assert e.transport(h).transport(g)==e.transport(mul(g,h));count+=1
    print('paired orbit classes',len(classes),'compositions',count)
    def T(x,p):
        y=tuple(tuple(a+b for a,b in zip(x,h)) for h in p)
        return y,tuple(tuple(a**3-b for a,b in zip(z,h)) for z,h in zip(y,p))
    indexed=0
    for v in product((-1,0,1),repeat=6):
        p=(v[:3],v[3:]); st=encode_pair(p)
        for x in ((1,0,-1),(0,0,0),(-1,1,0)):
            y,p=T(x,p);yy,st=indexed_pair_step(st,x,T);assert yy==y and st.decode()==p;indexed+=1
    print('indexed steps',indexed)
    torch.manual_seed(42);torch.set_num_threads(1)
    x=torch.randn(3,3,dtype=torch.float64)/4
    h=torch.randn(3,4,2,3,dtype=torch.float64)/4
    x[0]=0;h[0]=0 # exact zero
    x[1]=torch.tensor([1.,0.,-1.]);h[1]=0 # opposed chart fallback
    checks=0
    for cap in (None,.4):
        model=PairedHistoryMoE(factory(cap)); out=model.step(x,h)
        for g in GROUP:
            hh=transform(g,h,True);xx=transform(g,x);state=hh;base=h
            for t in range(3):
                a=model.step(x,base);b=model.step(xx,state)
                assert torch.equal(a.ids,b.ids)
                assert torch.allclose(a.weights,b.weights,atol=1e-13,rtol=1e-13)
                assert torch.allclose(transform(g,a.branches,True),b.branches,atol=1e-13,rtol=1e-13)
                assert torch.allclose(transform(g,a.history,True),b.history,atol=1e-13,rtol=1e-13)
                assert torch.equal(((-1)**g[2])*(a.signs.flip(-1) if g[1] else a.signs),b.signs)
                assert torch.allclose(transform(g,a.mean),b.mean,atol=1e-13,rtol=1e-13)
                assert torch.allclose((-1)**g[1]*transform(g,a.residual),b.residual,atol=1e-13,rtol=1e-13)
                mask=torch.ones(3,4,dtype=torch.bool);mask.scatter_(1,a.ids,False)
                assert torch.equal(a.history[mask],base[mask]);base=a.history;state=b.history;checks+=1
        assert torch.allclose(out.history_mean+out.history_residual,out.history[:,:,0])
        assert torch.allclose(out.history_mean-out.history_residual,out.history[:,:,1])
    # Exact mean collision before step; current mean output separates under GELU.
    model=PairedHistoryMoE(factory());x=torch.ones(1,3,dtype=torch.float64)
    h=torch.zeros(1,4,2,3,dtype=torch.float64);hh=h.clone();hh[0,0,0]=torch.tensor([1.,-1.,0.]);hh[0,0,1]=-hh[0,0,0]
    a,b=model.step(x,h),model.step(x,hh)
    assert torch.equal(a.ids,b.ids) and torch.equal(a.weights,b.weights)
    assert torch.equal(a.mean,torch.zeros_like(a.mean)) and b.mean.sum()>0
    # Cap-before-mean not equal to cap-after-mean: 2,0 -> mean .5 vs 1.
    cap=factory(1).cap_history
    p=torch.tensor([[2.,0.,0.]],dtype=torch.float64);q=torch.zeros_like(p)
    assert torch.equal((cap(p)+cap(q))/2,p/4) and torch.equal(cap((p+q)/2),p/2)
    # Shared affine branch map: exact mean/residual descent.
    A=sp.Matrix([[1,2],[0,-1]]);m=sp.Matrix(sp.symbols('m:2'));d=sp.Matrix(sp.symbols('d:2'));c=sp.Matrix([3,4]);u=A*(m+d)+c;v=A*(m-d)+c
    assert (u+v)/2==A*m+c and (u-v)/2==A*d
    print('PASS',torch.__version__,sp.__version__,'FP64 equivariant steps',checks,'GELU collision mean',b.mean.tolist())
def original_factory(cap=None):
    from models.history_orbit_moe import HistoryOrbitMoE
    fixture=Fixture(cap)
    model=HistoryOrbitMoE(d_model=3,num_experts=4,top_k=2,expert_width=3,route_geometry='c2_c3_residual').double().eval()
    model.history_cap=cap
    with torch.no_grad():
        model.router.weight.zero_();model.router.bias.zero_()
        model.c3_residual_weight.zero_();model.c3_residual_weight[0,4]=1
        model.expert_bias.copy_(torch.tensor([12.,4.,-4.,-12.],dtype=torch.float64))
        model.orientation.weight.copy_(fixture.orientation.weight)
        model.experts.load_state_dict(fixture.experts.state_dict())
        model.transitions.load_state_dict(fixture.transitions.state_dict())
    return model

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--original',action='store_true');args=parser.parse_args()
    print('source mode:', 'original HistoryOrbitMoE' if args.original else 'source-interface fixture')
    main(original_factory if args.original else Fixture)
