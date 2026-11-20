"""Compare original R4 consumer with explicit fusion on synthetic states."""
import copy
import math
import torch
from models.history_orbit_moe import HistoryOrbitMoE
from models.x369_shared_head import select_fused_r4

def fused_step(m,x,h,train_sign=False):
    ids,w=select_fused_r4(m,x,h)
    idx=ids.unsqueeze(-1).expand(-1,-1,m.d_model)
    y,updated,signs=m.execute_selected(x,ids,h.gather(1,idx),w,train_sign=train_sign)
    return y,h.scatter(1,idx,updated),ids,signs,w

def compare(m,x,h):
    a=m.step(x,h)
    b=fused_step(m,x,h)
    for i,(u,v) in enumerate(zip(a,b)):
        if i in (2,3):
            assert torch.equal(u,v),i
        else:
            torch.testing.assert_close(u,v,atol=1e-10,rtol=1e-10)
    return a,b

def fixture():
    m=HistoryOrbitMoE(d_model=6,num_experts=4,top_k=2,route_geometry="c2_c3_residual").double().eval()
    with torch.no_grad():
        for p in m.parameters(): p.zero_()
        m.c3_residual_weight[0,6]=1
        m.expert_bias.copy_(torch.tensor([8,3,-11,-16],dtype=torch.double))
        m.orientation.weight[0,6]=1
        m.experts[1][2].bias[0]=1
    h=torch.zeros(1,4,6,dtype=torch.double)
    h[0,:,0]=1
    h[0,:,2]=torch.tensor([0,4,9,11])
    def x(t):
        v=torch.zeros(1,6,dtype=torch.double); v[0,0]=t
        return v
    with torch.no_grad():
        expected=[[0,1],[0,1],[1,0],[3,2]]
        nat=[]
        for t in range(4):
            a,b=compare(m,x(t),h)
            assert b[2][0].tolist()==expected[t]
            nat.append(math.floor(10*float(b[0][0,0])))
        assert nat==[0,2,9,0]
        recurrent=[]
        for prefix in (0,3):
            a,b=compare(m,x(prefix),h)
            aa=m.step(x(2),a[1]); bb=fused_step(m,x(2),b[1])
            assert torch.equal(aa[2],bb[2])
            torch.testing.assert_close(aa[0],bb[0])
            torch.testing.assert_close(aa[1],bb[1])
            recurrent.append(math.floor(10*float(bb[0][0,0])))
        assert recurrent==[0,9]
        for cutoff in (5/4,19/9,5/2,19/7,14/5):
            for eps in (-1e-4,1e-4): compare(m,x(cutoff+eps),h)
        m.route_probe_action="c2_diagonal_negation"
        a=m.select_source_experts(x(2),h); b=select_fused_r4(m,x(2),h)
        assert all(torch.equal(u,v) for u,v in zip(a,b))
    print("native synthetic reset, recurrent, boundary, fallback controls PASS")

def gradients():
    torch.manual_seed(42)
    m=HistoryOrbitMoE(d_model=24,num_experts=4,top_k=2,route_geometry="c2_c3_residual").double()
    with torch.no_grad(): m.c3_residual_weight.normal_(std=.03)
    n=copy.deepcopy(m)
    x=torch.randn(3,24,dtype=torch.double,requires_grad=True)
    h=torch.randn(3,4,24,dtype=torch.double,requires_grad=True)
    xx=x.detach().clone().requires_grad_(); hh=h.detach().clone().requires_grad_()
    a=m.step(x,h,train_sign=True); b=fused_step(n,xx,hh,train_sign=True)
    assert torch.equal(a[2],b[2])
    for i in (0,1,4): torch.testing.assert_close(a[i],b[i],atol=1e-10,rtol=1e-10)
    for out in (a,b): (out[0].square().sum()+out[1].square().sum()+out[4].square().sum()).backward()
    for u,v in ((x,xx),(h,hh),*zip(m.parameters(),n.parameters())):
        assert (u.grad is None)==(v.grad is None)
        if u.grad is not None: torch.testing.assert_close(u.grad,v.grad,atol=1e-9,rtol=1e-9)
    print("native synthetic full consumer gradients PASS")

if __name__=="__main__":
    fixture()
    gradients()
