"""Signed-control factorization of the finite three-lane recurrent witness.

Original Leech-LILA: Anatolii Kornienko; DASHI extension. E6 incidence
fibre is an analogy, not an identification. Two explicitly separate
architectures: dense72 (unchanged equations) and group12 (new expert).
"""
import torch
from torch import nn
from models.geometric_reflection_fibre import GROUP, act
from models.geometric_bitlinear import apply_group_triads
from scripts.x369_source_geometry import SourceGeometryOrderedLearner
from scripts.x369_constructive_finite_source import representatives,audit_candidate

SPATIAL=((0,0,0),(2,0,0),(2,1,0))
OFFSETS=(1,3)


def scalar_coefficients():
    v=torch.tensor([[1.],[2.]],dtype=torch.float64)
    t=torch.tensor([OFFSETS],dtype=v.dtype)
    g=lambda z:torch.nn.functional.gelu(8*z,approximate="none")/8
    B=g(v+t)-g(-v+t)-v
    return torch.linalg.solve(B,v).flatten()


@torch.no_grad()
def common(width,device,dtype):
    model=SourceGeometryOrderedLearner(width=width,experts=4).to(device=device,dtype=dtype).eval()
    for p in model.parameters():p.zero_()
    src=model.history.source
    src.history_cap=None
    src.expert_bias.copy_(src.expert_bias.new_tensor([2,1,0,0]))
    return model


def action_matrix(g,lane):
    # Column form of the original signed spatial action.
    return tuple(tuple(((-1)**(g[1]*lane))*act(g,tuple(int(i==j) for i in range(3)))[k]
                       for j in range(3)) for k in range(3))


@torch.no_grad()
def dense72(device="cpu",dtype=torch.float64):
    """72 hidden/recurrent width. Original dense GELU equations unchanged.
    Exact-real coefficients exist; stored floating coefficients approximate.
    Expert output includes a baseline removed by the actual history transition.
    """
    width=72;model=common(width,device,dtype);src=model.history.source
    coeff=scalar_coefficients();A=src.experts[0][0];W=src.experts[0][2]
    T=src.transitions[0].weight
    src.orientation.weight[0,6]=1
    for tag in range(4):
        for u in representatives():
            k,b,a=GROUP[u];i=SPATIAL.index((k,b,0))
            x=model.control_embed.weight[tag*len(GROUP)+u]
            x[6]=1
            for j,t in enumerate(OFFSETS):x[2*i+j]=(-1)**a*t
    for lane in range(2):
        for i,g in enumerate(SPATIAL):
            P=W.weight.new_tensor(action_matrix(g,lane))
            start=3*(lane+1)
            # Cancel the shared f(v,0)=v baseline through original A_h.
            T[start:start+3,:3]-=coeff.sum().to(T)*P
            T[start:start+3,start:start+3]-=coeff.sum().to(T)*P
            for j in range(3):
                for t in range(2):
                    for b in range(2):
                        slot=(((lane*3+j)*3+i)*2+t)*2+b
                        s=(-1)**b
                        A.weight[slot,2*i+t]=8
                        A.weight[slot,width+j]=8*s
                        A.weight[slot,width+start+j]=8*s
                        W.weight[start:start+3,slot]=s*coeff[t].to(W.weight)*P[:,j]/8
        for j in range(start,start+3):T[j,2*width+j]=1
    return model


class SignedControlExpert(nn.Module):
    """Explicit bilinear expert, not a parameterization of the old GELU FFN.
    Reads canonical control embedding and actual branch history only.
    """
    def __init__(self,width):
        super().__init__();self.width=width
        basis=torch.zeros(3,2,6)
        for i,(k,b,_) in enumerate(SPATIAL):
            basis[i,0,k+3*b]=1
            basis[i,1,k+3*b]=(-1)**b
        self.register_buffer("basis",basis,persistent=False)

    def forward(self,z):
        x,h=z.split(self.width,-1)
        lanes=torch.cat((h[...,:3]+h[...,3:6],h[...,:3]+h[...,6:9]),-1)
        y=sum(x[...,i:i+1]*apply_group_triads(lanes,self.basis[i].to(lanes))
              for i in range(3))
        return torch.cat((torch.zeros_like(h[...,:3]),y,torch.zeros_like(h[...,9:])), -1)


@torch.no_grad()
def group12(device="cpu",dtype=torch.float64):
    """Width-12 opt-in group-algebra expert; preserve other source equations."""
    model=common(12,device,dtype);src=model.history.source
    src.experts[0]=SignedControlExpert(12).to(device=device,dtype=dtype).eval()
    src.orientation.weight[0,3]=1
    for tag in range(4):
        for u in representatives():
            k,b,a=GROUP[u];i=SPATIAL.index((k,b,0))
            x=model.control_embed.weight[tag*len(GROUP)+u]
            x[i]=(-1)**a;x[3]=1
    for j in range(3,9):src.transitions[0].weight[j,24+j]=1
    return model


def main():
    import argparse,json
    from hashlib import sha256
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=("dense72","group12"),required=True)
    p.add_argument("--device",choices=("cpu","cuda"),required=True)
    p.add_argument("--dtype",choices=("float32","float64"),default="float64")
    p.add_argument("--audit-rows",type=int,default=12)
    p.add_argument("--audit-depth",type=int,default=4)
    p.add_argument("--checkpoint",required=True)
    a=p.parse_args()
    if a.device=="cuda" and (not torch.cuda.is_available() or not torch.version.hip or
        not getattr(torch.cuda.get_device_properties(0),"gcnArchName","").startswith("gfx803")):
        raise RuntimeError("native gfx803 required")
    model={"dense72":dense72,"group12":group12}[a.mode](a.device,getattr(torch,a.dtype))
    result=audit_candidate(model,a.audit_rows,a.audit_depth)
    if any(result["head_errors"]) or result["route_errors"]:
        raise RuntimeError("candidate did not pass bounded semantic replay")
    # Mode is essential: group12 has a different expert class/state schema.
    with open(a.checkpoint,"xb") as f:torch.save(model.state_dict(),f)
    with open(a.checkpoint,"rb") as f:digest=sha256(f.read()).hexdigest()
    print(json.dumps(dict(mode=a.mode,width=model.width,sha256=digest,dtype=a.dtype,
        device=a.device,audit=result,training_steps=0,original_affine_readout_admitted=False,
        original_dense_expert_equations=(a.mode=="dense72"),native_float_universal=False)))


if __name__=="__main__":main()
