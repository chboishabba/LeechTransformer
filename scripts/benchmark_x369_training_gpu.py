"""GPU-only training allocation and timing for original/shared/two/DC-VJP arms."""
import argparse,json,statistics,time
import torch
from models.history_orbit_moe import HistoryOrbitMoE
from models.x369_shared_head import select_fused_r4
from models.x369_two_channel_codec import select_two_with_dc
from models.x369_differential_backward import select_vjp_r4
from scripts.check_x369_two_channel_gpu import step

def main():
    p=argparse.ArgumentParser()
    for name,default in (("batch",8),("width",384),("experts",4),("repeats",7),("iterations",10)):
        p.add_argument("--"+name,type=int,default=default)
    p.add_argument("--require-gfx803",action="store_true")
    p.add_argument("--dtype",choices=("float32","float64"),default="float32")
    a=p.parse_args()
    if min(a.batch,a.width,a.repeats,a.iterations)<1 or a.width%3 or a.experts<2:p.error("bad shape/count")
    if not torch.cuda.is_available():raise RuntimeError("GPU required")
    dev=torch.device("cuda",torch.cuda.current_device())
    prop=torch.cuda.get_device_properties(dev); arch=getattr(prop,"gcnArchName","")
    if a.require_gfx803 and (not torch.version.hip or not arch.startswith("gfx803")):raise RuntimeError("gfx803 required")
    dt=getattr(torch,a.dtype); tol=1e-4 if dt==torch.float32 else 1e-10
    torch.manual_seed(42)
    m=HistoryOrbitMoE(d_model=a.width,num_experts=a.experts,top_k=2,route_geometry="c2_c3_residual").to(dev,dt)
    with torch.no_grad():m.c3_residual_weight.normal_(std=.003)
    x=torch.randn(a.batch,a.width,device=dev,dtype=dt,requires_grad=True)
    h=torch.randn(a.batch,a.experts,a.width,device=dev,dtype=dt,requires_grad=True)
    selectors={"original":lambda m,x,h:m.select_source_experts(x,h),"shared":select_fused_r4,"two_dc":select_two_with_dc,"vjp":select_vjp_r4}
    def clear():
        m.zero_grad(set_to_none=True); x.grad=None; h.grad=None
    def run(sel,scope):
        if scope=="routing":
            ids,w=sel(m,x,h); loss=w.square().mean()
        else:
            y,nh,ids,signs,w=step(m,x,h,sel)
            loss=y.square().mean()+nh.square().mean()+w.square().mean()
        return ids,loss
    def audit(sel,scope):
        clear(); saved=[]
        def pack(t):
            saved.append((t.numel()*t.element_size(),t.untyped_storage().data_ptr(),t.untyped_storage().nbytes()))
            return t
        with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):
            ids,loss=run(sel,scope)
        loss.backward()
        pointers={ptr:n for _,ptr,n in saved}
        grads=[None if v.grad is None else v.grad.detach().clone() for v in (x,h,*m.parameters())]
        result=dict(saved_calls=len(saved),logical_saved_bytes=sum(n for n,_,_ in saved),unique_referenced_storage_bytes=sum(pointers.values()))
        return ids.detach().clone(),loss.detach().clone(),grads,result
    def measure(sel,scope):
        clear(); torch.cuda.synchronize(dev)
        base=torch.cuda.memory_allocated(dev); torch.cuda.reset_peak_memory_stats(dev)
        fwd=bwd=0.
        for _ in range(a.iterations):
            clear(); torch.cuda.synchronize(dev)
            t=time.perf_counter(); ids,loss=run(sel,scope); torch.cuda.synchronize(dev)
            fwd+=time.perf_counter()-t
            t=time.perf_counter(); loss.backward(); torch.cuda.synchronize(dev)
            bwd+=time.perf_counter()-t
            del ids,loss
        return dict(forward_ms=1000*fwd/a.iterations,backward_ms=1000*bwd/a.iterations,peak_increment_bytes=torch.cuda.max_memory_allocated(dev)-base)
    for scope in ("routing","full_step"):
        ref_ids,ref_loss,ref_grads,_=audit(selectors["original"],scope)
        audits={}
        for name,sel in selectors.items():
            ids,loss,grads,receipt=audit(sel,scope)
            assert torch.equal(ids,ref_ids)
            torch.testing.assert_close(loss,ref_loss,atol=tol,rtol=tol)
            for u,v in zip(grads,ref_grads):
                assert (u is None)==(v is None)
                if u is not None:torch.testing.assert_close(u,v,atol=tol,rtol=tol)
            audits[name]=receipt
        del ref_ids,ref_loss,ref_grads,ids,loss,grads
        for sel in selectors.values():measure(sel,scope)
        samples={name:[] for name in selectors}
        for i in range(a.repeats):
            order=list(selectors)
            order=order[i%len(order):]+order[:i%len(order)]
            for name in order:samples[name].append(measure(selectors[name],scope))
        summary={name:{key:statistics.median(v[key] for v in rows) for key in rows[0]} for name,rows in samples.items()}
        print(json.dumps(dict(scope=scope,arch=arch,device=prop.name,hip=torch.version.hip,torch=torch.__version__,args=vars(a),saved_tensor_audit=audits,samples=samples,summary=summary,optimizer_included=False)),flush=True)
if __name__=="__main__":main()
