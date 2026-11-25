"""Constructive finite task parameter witness for the NEW source architecture.
Original Leech-LILA: Anatolii Kornienko; DASHI extension.
Exact-real existence via GELU interpolation; floating materialization is
a numerical candidate, NOT exact/native acceptance or fitted migration.
"""
from itertools import product
import torch
from models.geometric_reflection_fibre import GROUP,act,mul,inv
from scripts.x369_source_geometry import SourceGeometryOrderedLearner
from scripts.x369_typed_prototype_readout import TypedPrototypeReadout


def representatives():
    return tuple(sorted({max(GROUP.index(mul(mul(g,u),inv(g))) for g in GROUP)
                         for u in GROUP}))


def interpolation(dtype=torch.float64,device="cpu"):
    """Integer nodal GELU tents; exact-real matrix strictly diagonally dominant."""
    reps=representatives();n=5*len(reps)
    nodes=torch.arange(n,dtype=dtype,device=device)
    x=nodes[:,None]-nodes[None,:]
    gelu=torch.nn.functional.gelu
    B=(gelu(8*(x+1),approximate="none")-2*gelu(8*x,approximate="none")+
       gelu(8*(x-1),approximate="none"))/8
    return reps,B


@torch.no_grad()
def construct(device="cpu",dtype=torch.float64):
    """540-width constructive candidate, not the historical width-12 model.
    Only expert 0 carries memory; expert 1 is an always-selected zero expert.
    Original routing, sign, expert GELU, transition, descent and cap interface
    unchanged. Original affine classifier intentionally zeroed; consume
    decoded actual history with the existing typed prototype readout.
    """
    reps,B=interpolation();n=B.shape[0];width=6*3*n
    model=SourceGeometryOrderedLearner(width=width,experts=4).to(device=device,dtype=dtype).eval()
    for p in model.parameters():p.zero_()
    src=model.history.source;src.history_cap=None
    src.expert_bias.copy_(torch.tensor([2,1,0,0],device=device,dtype=dtype))
    src.orientation.weight[0,1]=1
    for tag in range(4):
        for c,gidx in enumerate(reps):
            model.control_embed.weight[tag*len(GROUP)+gidx,0]=5*c
            model.control_embed.weight[tag*len(GROUP)+gidx,1]=1
    A=src.experts[0][0];W=src.experts[0][2]
    for lane,j in product(range(2),range(3)):
        scalar=3*lane+j
        # seed triad plus the appropriate persistent lane.
        targets=torch.zeros(n,3,dtype=B.dtype)
        e=tuple(int(i==j) for i in range(3))
        for c,gidx in enumerate(reps):
            g=GROUP[gidx];col=act(g,e)
            if lane==1:col=tuple((-1)**g[1]*z for z in col)
            for k,value in enumerate(range(-2,3)):
                targets[5*c+k]=torch.tensor(col,dtype=B.dtype)*value
        coeff=torch.linalg.solve(B,targets)
        for k in range(n):
            for shift,weight in ((1,1),(0,-2),(-1,1)):
                slot=(scalar*n+k)*3+(1-shift)
                A.weight[slot,0]=8
                A.weight[slot,width+j]=8
                A.weight[slot,width+3*(lane+1)+j]=8
                A.bias[slot]=8*(2-k+shift)
                W.weight[3*(lane+1):3*(lane+2),slot]=(coeff[k]*weight/8).to(device=device,dtype=dtype)
    # Original transition input is (canonical old h, canonical x, y).
    for j in range(3,9):
        src.transitions[0].weight[j,2*width+j]=1
    return model


def decode_history(h):
    """Decode ONLY actual recurrent histories, never supplied action register."""
    if h.ndim!=4 or h.shape[1]<2 or h.shape[2]!=2 or h.shape[-1]<9:
        raise ValueError("paired history with three lanes required")
    p=h[:,0];m=p[...,:3]+p[...,3:6];d=p[...,:3]+p[...,6:9]
    return (m[:,0]+m[:,1])/2,(d[:,0]-d[:,1])/2


def predict(model,seed,controls,tags):
    # Actual original rollout; no semantic register input.
    _,trace=model.rollout_group(seed,controls,tags)
    m,d=decode_history(trace[-1].history)
    return TypedPrototypeReadout().to(device=m.device)(m,d),trace


def main():
    import argparse,json
    p=argparse.ArgumentParser()
    p.add_argument("--device",choices=("cpu","cuda"),required=True)
    p.add_argument("--dtype",choices=("float32","float64"),default="float64")
    p.add_argument("--checkpoint",required=True)
    a=p.parse_args()
    if a.device=="cuda" and (not torch.cuda.is_available() or not torch.version.hip or not getattr(torch.cuda.get_device_properties(0),"gcnArchName","").startswith("gfx803")):
        raise RuntimeError("native gfx803 required for GPU path")
    model=construct(a.device,getattr(torch,a.dtype))
    # Exclusive create; no historical checkpoint overwrite.
    with open(a.checkpoint,"xb") as f:torch.save(model.state_dict(),f)
    from hashlib import sha256
    with open(a.checkpoint,"rb") as f:digest=sha256(f.read()).hexdigest()
    print(json.dumps(dict(kind="constructive_numeric_candidate",sha256=digest,width=model.width,
       dtype=a.dtype,device=a.device,training_steps=0,exact_float_certified=False,
       historical_checkpoint_migration=False,original_affine_readout_admitted=False)))


if __name__=="__main__":main()
