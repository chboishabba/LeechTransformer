"""One narrowly scoped native integration acceptance, not a learning experiment."""
import json
import torch
from models.geometric_reflection_fibre import GROUP
from models.geometric_orbit_descent import act_tensor,conjugate_control
from scripts.x369_source_geometry import SourceGeometryOrderedLearner,literal_controls,known_action_register
from scripts.audit_x369_ordered_signed_task import endpoint
from scripts.train_x369_ordered_action import encode_inputs


def main():
    if not torch.cuda.is_available() or not torch.version.hip or not getattr(torch.cuda.get_device_properties(0),"gcnArchName","").startswith("gfx803"):
        raise RuntimeError("native gfx803 HIP required")
    torch.manual_seed(41);dev=torch.device("cuda",0)
    rows=[dict(word="abjn",guards=(1,2,1,1),mean=(1,0,0),residual=(1,-1,0)),
          dict(word="abjn",guards=(0,0,0,0),mean=(0,0,0),residual=(0,0,0)),
          dict(word="abjn",guards=(1,0,1,0),mean=(0,0,0),residual=(1,1,1))]
    ss,oo=zip(*(encode_inputs(r,"signed") for r in rows))
    s=torch.tensor(ss,device=dev,dtype=torch.float32);ops=torch.tensor(oo,device=dev,dtype=s.dtype)
    c,t=literal_controls(ops);model=SourceGeometryOrderedLearner().to(dev).eval()
    with torch.no_grad():
        expected=[endpoint(r["word"],r["guards"],r["mean"],r["residual"]) for r in rows]
        values=known_action_register(s,c)
        # endpoint returns mean,signed,Gram; register returns mean,Gram,signed.
        for v,i in zip(values,(0,2,1)):
            assert torch.equal(v,torch.tensor([r[i] for r in expected],device=dev,dtype=s.dtype))
        y,trace=model.rollout_group(s,c,t)
        assert torch.equal(trace[-1].history[1],torch.zeros_like(trace[-1].history[1]))
        defects={}
        for g in ((2,0,0),(2,1,0),(0,0,1)):
            x=s.clone();x[:,:3]=act_tensor(g,s[:,:3]);x[:,3:6]=(-1)**g[1]*act_tensor(g,s[:,3:6])
            z,tr=model.rollout_group(x,conjugate_control(g,c),t)
            errs=[]
            for a,b in zip(trace,tr):
                for u,v in ((a.output,b.output),(a.history,b.history)):
                    q=act_tensor(g,u,True);errs.append(float((q-v).abs().max()))
                    assert torch.allclose(q,v,rtol=2e-5,atol=2e-6)
                assert torch.equal(a.ids,b.ids) and torch.equal(a.local_signs,b.local_signs)
                assert torch.allclose(a.weights,b.weights,rtol=2e-5,atol=2e-6)
            for j,(a,b) in enumerate(zip(y,z)):
                expected=a[:,getattr(model,f"label_perm_{j}")[GROUP.index(g)]]
                errs.append(float((expected-b).abs().max()))
                assert torch.allclose(expected,b,rtol=2e-5,atol=2e-6)
            defects[str(g)]=max(errs)
    # Single backward plumbing, no optimizer steps or claims of training-law parity.
    model.train();loss=sum(v.square().mean() for v in model(s,ops));loss.backward()
    assert any(p.grad is not None and p.grad.abs().max()>0 for p in model.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    print(json.dumps(dict(kind="source_geometry_native_integration",rows=len(rows),steps=4,defects=defects,torch=str(torch.__version__),hip=torch.version.hip,training_steps=0,universal_float_certified=False,release_admitted=False)))
if __name__=="__main__":main()
