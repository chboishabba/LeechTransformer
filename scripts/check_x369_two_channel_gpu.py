"""GPU-only four-arm learner codec experiment (synthetic, no fitted/F5 assets)."""
import argparse,copy,json,time
import torch
from models.history_orbit_moe import HistoryOrbitMoE
from models.x369_shared_head import select_fused_r4
from models.x369_two_channel_codec import select_two_with_dc,audit_omission
from models.trialectic_wave_prior import rotate3
from models.x369_two_channel_codec import exact_common_dc

def step(m,x,h,select):
    ids,w=select(m,x,h)
    idx=ids.unsqueeze(-1).expand(-1,-1,m.d_model)
    y,updated,signs=m.execute_selected(x,ids,h.gather(1,idx),w,train_sign=True)
    return y,h.scatter(1,idx,updated),ids,signs,w

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--require-gfx803",action="store_true")
    p.add_argument("--dtype",choices=("float32","float64"),default="float64")
    p.add_argument("--steps",type=int,default=20)
    p.add_argument("--width",type=int,default=384)
    p.add_argument("--experts",type=int,default=4)
    p.add_argument("--batch",type=int,default=3)
    a=p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("GPU required; no CPU fallback")
    if a.width<6 or a.width%3 or a.experts<2 or min(a.batch,a.steps)<1:
        p.error("invalid shape/count")
    dev=torch.device("cuda",torch.cuda.current_device())
    prop=torch.cuda.get_device_properties(dev)
    arch=getattr(prop,"gcnArchName","")
    if a.require_gfx803 and (not torch.version.hip or not arch.startswith("gfx803")):
        raise RuntimeError("gfx803 HIP required")
    dt=getattr(torch,a.dtype)
    tol=1e-4 if dt==torch.float32 else 1e-10
    torch.manual_seed(42)
    m=HistoryOrbitMoE(d_model=a.width,num_experts=a.experts,top_k=2,route_geometry="c2_c3_residual").to(dev,dt)
    with torch.no_grad(): m.c3_residual_weight.normal_(std=.003)
    models=[m,copy.deepcopy(m),copy.deepcopy(m)]
    selectors=[lambda m,x,h:m.select_source_experts(x,h),select_fused_r4,select_two_with_dc]
    names=["original","shared_head","two_with_dc"]
    opts=[torch.optim.AdamW(v.parameters(),lr=1e-4) for v in models]
    h=torch.randn(a.batch,a.experts,a.width,device=dev,dtype=dt)
    hs=[h.clone() for _ in models]
    print(json.dumps(dict(arch=arch,device=prop.name,hip=torch.version.hip,torch=torch.__version__,args=vars(a),tolerance=tol)),flush=True)
    # Construct exact aggregate equality with unequal per-triad DC, and a rejection.
    with torch.no_grad():
        test=copy.deepcopy(m)
        test.c3_residual_weight.zero_()
        test.c3_residual_weight[0,a.width+2]=2
        test.c3_residual_weight[0,a.width+5]=3
        x=torch.zeros(1,a.width,device=dev,dtype=dt)
        x[0,0]=x[0,3]=1
        h0=torch.zeros(1,a.experts,a.width,device=dev,dtype=dt)
        h0[:,:,0]=1;h0[:,:,3]=4
        h0[:,1,0]=4;h0[:,1,3]=2
        w=test.c3_residual_weight[:,a.width:]
        assert exact_common_dc(rotate3(x),h0,w)==[True]
        _,positive=audit_omission(test,x,h0)
        h0[:,1,3]=3
        assert exact_common_dc(rotate3(x),h0,w)==[False]
        _,negative=audit_omission(test,x,h0)
        assert not negative["admitted"]
        print(json.dumps(dict(aggregate_control=positive,unequal_control=negative)),flush=True)
    def diff(u,v):
        return float((u-v).abs().max())
    for t in range(a.steps):
        x=torch.randn(a.batch,a.width,device=dev,dtype=dt)
        xs=[x.clone().requires_grad_() for _ in models]
        hs=[v.detach().requires_grad_() for v in hs]
        for opt in opts:opt.zero_grad(set_to_none=True)
        out=[step(v,xx,hh,s) for v,xx,hh,s in zip(models,xs,hs,selectors)]
        for v in out:
            (v[0].square().mean()+v[1].square().mean()+v[4].square().mean()).backward()
        rec=[]
        for j in (1,2):
            same_ids=torch.equal(out[0][2],out[j][2])
            same_signs=torch.equal(out[0][3],out[j][3])
            pairs=[(xs[0],xs[j]),(hs[0],hs[j]),*zip(models[0].parameters(),models[j].parameters())]
            grads=[diff(u.grad,v.grad) for u,v in pairs if u.grad is not None and v.grad is not None]
            grad_presence=all((u.grad is None)==(v.grad is None) for u,v in pairs)
            rec.append(dict(arm=names[j],ids_equal=same_ids,signs_equal=same_signs,
                response_error=diff(out[0][0],out[j][0]),history_error=diff(out[0][1],out[j][1]),
                weight_error=diff(out[0][4],out[j][4]),gradient_presence_equal=grad_presence,
                max_gradient_error=max(grads,default=0)))
            for u,v in pairs:
                if u.grad is not None and v.grad is not None:
                    assert torch.isfinite(u.grad).all() and torch.isfinite(v.grad).all()
        for opt in opts:opt.step()
        for j,r in zip((1,2),rec):
            r["max_parameter_error"]=max(diff(u,v) for u,v in zip(models[0].parameters(),models[j].parameters()))
        torch.cuda.synchronize(dev)
        print(json.dumps(dict(step=t+1,comparisons=rec)),flush=True)
        # Float32 is a measured drift experiment, not a silently loosened PASS.
        if dt==torch.float64:
            for j,r in zip((1,2),rec):
                assert r["ids_equal"] and r["signs_equal"] and r["gradient_presence_equal"]
                for i in (0,1,4):torch.testing.assert_close(out[0][i],out[j][i],atol=tol,rtol=tol)
                for u,v in [(xs[0],xs[j]),(hs[0],hs[j]),*zip(models[0].parameters(),models[j].parameters())]:
                    torch.testing.assert_close(u,v,atol=tol,rtol=tol)
                    if u.grad is not None:torch.testing.assert_close(u.grad,v.grad,atol=tol,rtol=tol)
        hs=[v[1].detach() for v in out]
    with torch.no_grad():
        _,audit=audit_omission(models[0],x,hs[0])
    print(json.dumps(dict(status="PASS_FP64" if dt==torch.float64 else "FP32_MEASURED_NOT_CERTIFIED",steps=a.steps,final_shadow=audit)),flush=True)
if __name__=="__main__":main()
