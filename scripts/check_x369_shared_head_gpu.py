"""GPU-first X369 acceptance. No CPU fallback; HIP uses torch.cuda API."""
import argparse, copy, json, math, time
import torch
from models.history_orbit_moe import HistoryOrbitMoE
from scripts.check_x369_shared_head_consumer import fused_step

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--dtype",choices=["float32","float64"],default="float32")
    p.add_argument("--selection",choices=["topk","sort"],default="topk")
    p.add_argument("--steps",type=int,default=20)
    p.add_argument("--require-gfx803",action="store_true")
    a=p.parse_args()
    assert a.steps>0
    dev=torch.device(a.device)
    if dev.type!="cuda" or not torch.cuda.is_available():
        raise RuntimeError("GPU required; no CPU fallback")
    dev=torch.device("cuda",torch.cuda.current_device() if dev.index is None else dev.index)
    props=torch.cuda.get_device_properties(dev)
    arch=getattr(props,"gcnArchName","")
    if a.require_gfx803 and (not torch.version.hip or not arch.startswith("gfx803")):
        raise RuntimeError("gfx803 HIP device required")
    dt=getattr(torch,a.dtype)
    tol=1e-5 if dt==torch.float32 else 1e-10
    print(json.dumps(dict(device=str(dev),name=props.name,arch=arch,hip=torch.version.hip,torch=torch.__version__,dtype=a.dtype,selection=a.selection)),flush=True)
    torch.manual_seed(42)
    def check(u,v):
        assert u.device==dev and v.device==dev
        assert torch.isfinite(u).all() and torch.isfinite(v).all()
        torch.testing.assert_close(u,v,atol=tol,rtol=tol)
    def compare(m,n,x,h,hh,train=False,xx=None):
        A=m.step(x,h,train_sign=train)
        B=fused_step(n,x if xx is None else xx,hh,train_sign=train)
        for i,(u,v) in enumerate(zip(A,B)):
            if i in (2,3):
                assert u.device==dev and v.device==dev and torch.equal(u,v),i
            else: check(u,v)
        return A,B
    m=HistoryOrbitMoE(d_model=6,num_experts=4,top_k=2,route_geometry="c2_c3_residual",router_selection=a.selection).to(dev,dt)
    with torch.no_grad():
        for v in m.parameters(): v.zero_()
        m.c3_residual_weight[0,6]=1
        m.expert_bias.copy_(torch.tensor([8,3,-11,-16],device=dev,dtype=dt))
        m.orientation.weight[0,6]=1
        m.experts[1][2].bias[0]=1
    h=torch.zeros(1,4,6,device=dev,dtype=dt)
    h[0,:,0]=1; h[0,:,2]=torch.tensor([0,4,9,11],device=dev,dtype=dt)
    def x(t):
        v=torch.zeros(1,6,device=dev,dtype=dt); v[0,0]=t; return v
    with torch.no_grad():
        nat=[]
        for t,ids in enumerate(([0,1],[0,1],[1,0],[3,2])):
            A,B=compare(m,m,x(t),h,h)
            assert B[2][0].tolist()==ids
            assert math.floor(10*float(A[0][0,0]))==math.floor(10*float(B[0][0,0]))
            nat.append(math.floor(10*float(B[0][0,0])))
        assert nat==[0,2,9,0],nat
        recurrent=[]
        for t in (0,3):
            A,B=compare(m,m,x(t),h,h)
            A,B=compare(m,m,x(2),A[1],B[1])
            assert math.floor(10*float(A[0][0,0]))==math.floor(10*float(B[0][0,0]))
            recurrent.append(math.floor(10*float(B[0][0,0])))
        assert recurrent==[0,9],recurrent
        for c in (5/4,19/9,5/2,19/7,14/5):
            for e in (-1e-4,1e-4): compare(m,m,x(c+e),h,h)
        m.route_probe_action="c2_diagonal_negation"
        compare(m,m,x(2),h,h)
        m.route_probe_action=None
    print("GPU fixture PASS",nat,recurrent,flush=True)
    m=HistoryOrbitMoE(d_model=384,num_experts=4,top_k=2,route_geometry="c2_c3_residual",router_selection=a.selection).to(dev,dt)
    with torch.no_grad(): m.c3_residual_weight.normal_(std=.003)
    n=copy.deepcopy(m)
    opt=[torch.optim.AdamW(v.parameters(),lr=1e-4) for v in (m,n)]
    h=torch.randn(3,4,384,device=dev,dtype=dt); hh=h.clone()
    torch.cuda.synchronize(dev); start=time.perf_counter()
    for t in range(a.steps):
        x=torch.randn(3,384,device=dev,dtype=dt,requires_grad=True)
        xx=x.detach().clone().requires_grad_()
        h=h.detach().requires_grad_(); hh=hh.detach().requires_grad_()
        for o in opt:o.zero_grad(set_to_none=True)
        A,B=compare(m,n,x,h,hh,True,xx)
        for out in (A,B): (out[0].square().mean()+out[1].square().mean()+out[4].square().mean()).backward()
        for u,v in ((x,xx),(h,hh),*zip(m.parameters(),n.parameters())):
            assert (u.grad is None)==(v.grad is None)
            if u.grad is not None:check(u.grad,v.grad)
        for o in opt:o.step()
        for u,v in zip(m.parameters(),n.parameters()):check(u,v)
        h,hh=A[1].detach(),B[1].detach()
        torch.cuda.synchronize(dev)
        print("GPU matched update",t+1,flush=True)
    print(json.dumps(dict(status="PASS",steps=a.steps,combined_seconds=time.perf_counter()-start)),flush=True)
if __name__=="__main__":main()
