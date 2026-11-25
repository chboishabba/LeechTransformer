"""Compact original-equation finite signed action; no optimizer.
Original Leech-LILA: Anatolii Kornienko; DASHI extension.
Exact-real existence by scalar IVT; stored coefficients are numerical.
"""
import math
from itertools import product
import torch

SPATIAL=((0,0,0),(2,0,0),(2,1,0))
WIDTH=30


def scalar_gate():
    def g(x):return x*(1+math.erf(x/math.sqrt(2)))/2
    def H(v,t):return g(v+t)-g(-v+t)-v
    def d(t):return H(2,t)-2*H(1,t)
    lo,hi=2.,3.
    if not d(lo)<0<d(hi):raise RuntimeError('scalar bracket failed')
    for _ in range(60):
        m=(lo+hi)/2
        if d(m)<0:lo=m
        else:hi=m
    t=(lo+hi)/2;return t,1/H(1,t)


def column(g,j,lane):
    k,b,a=g
    return ((2-j if b else j)-k)%3,(-1)**(a+b*lane)


def terms():
    # One rank-one coordinate incidence per distinct destination per input.
    for lane,j in product(range(2),range(3)):
        cols=[column(g,j,lane) for g in SPATIAL]
        for dst in sorted({d for d,_ in cols}):
            yield lane,j,dst,tuple(s if d==dst else 0 for d,s in cols)


@torch.no_grad()
def fill(A,W,T):
    """Original nn.Linear weights (biases independently zeroed)."""
    t,alpha=scalar_gate()
    for i,(lane,j,dst,_) in enumerate(terms()):
        start=3*(lane+1);out=start+dst
        for b in range(2):
            s=(-1)**b;row=2*i+b
            A[row,i]=1;A[row,WIDTH+j]=s;A[row,WIDTH+start+j]=s
            W[out,row]=s*alpha
        T[out,j]-=alpha;T[out,start+j]-=alpha
    for j in range(3,9):T[j,2*WIDTH+j]=1
    return t


def embedding(g,t,dtype,device):
    k,b,a=g;i=SPATIAL.index((k,b,0));x=torch.zeros(WIDTH,dtype=dtype,device=device)
    for j,(_,_,_,coeff) in enumerate(terms()):x[j]=(-1)**a*coeff[i]*t
    x[14]=1
    return x


@torch.no_grad()
def construct(device='cpu',dtype=torch.float64):
    from scripts.x369_signed_control_factorization import common,action_matrix
    from scripts.x369_constructive_finite_source import representatives
    from models.geometric_reflection_fibre import GROUP
    # Guard the independent coordinate incidence against actual source owner.
    for g in SPATIAL:
        for lane,j in product(range(2),range(3)):
            dst,s=column(g,j,lane);P=action_matrix(g,lane)
            assert tuple(P[i][j] for i in range(3))==tuple(s*int(i==dst) for i in range(3))
    model=common(WIDTH,device,dtype);src=model.history.source
    t=fill(src.experts[0][0].weight,src.experts[0][2].weight,src.transitions[0].weight)
    src.orientation.weight[0,14]=1
    for tag,u in product(range(4),representatives()):
        model.control_embed.weight[tag*len(GROUP)+u].copy_(embedding(GROUP[u],t,dtype,device))
    return model


@torch.no_grad()
def self_test():
    # Extracted dense layers, not full recurrence or native acceptance.
    assert len(list(terms()))==14
    t,alpha=scalar_gate();print('offset',t,'scale',alpha)
    v=torch.tensor(list(product(range(-2,3),repeat=3)),dtype=torch.float64)
    for dtype in (torch.float64,torch.float32):
        A=torch.zeros(WIDTH,2*WIDTH,dtype=dtype);W=torch.zeros(WIDTH,WIDTH,dtype=dtype);T=torch.zeros(WIDTH,3*WIDTH,dtype=dtype)
        fill(A,W,T);h=torch.zeros(125+125**2,WIDTH,dtype=dtype)
        h[:125,:3]=v;h[125:,3:6]=v.repeat_interleave(125,0);h[125:,6:9]=v.repeat(125,1)
        worst=0.;n=0
        for k,b,_ in SPATIAL:
            for a in (0,1):
                g=(k,b,a);x=embedding(g,t,dtype,'cpu').expand(len(h),-1)
                y=torch.nn.functional.gelu(torch.cat((x,h),-1)@A.T)@W.T
                nxt=torch.cat((h,x,y),-1)@T.T;want=torch.zeros_like(h)
                for lane,j in product(range(2),range(3)):
                    dst,s=column(g,j,lane);start=3*(lane+1)
                    want[:,start+dst]=s*(h[:,j]+h[:,start+j])
                worst=max(worst,float((nxt-want).abs().max()));n+=len(h)
        assert worst < (1e-12 if dtype==torch.float64 else 1e-5)
        print(str(dtype),n,worst)


def main():
    import argparse,json
    from hashlib import sha256
    p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true');p.add_argument('--device',choices=('cpu','cuda'));p.add_argument('--dtype',choices=('float32','float64'),default='float64');p.add_argument('--checkpoint');p.add_argument('--audit-rows',type=int,default=12);p.add_argument('--audit-depth',type=int,default=4);a=p.parse_args()
    if a.self_test:self_test();return
    if not a.device or not a.checkpoint:p.error('explicit device and checkpoint required')
    if a.device=='cuda' and (not torch.cuda.is_available() or not torch.version.hip or not getattr(torch.cuda.get_device_properties(0),'gcnArchName','').startswith('gfx803')):raise RuntimeError('native gfx803 required')
    from scripts.x369_constructive_finite_source import audit_candidate
    model=construct(a.device,getattr(torch,a.dtype));r=audit_candidate(model,a.audit_rows,a.audit_depth)
    if any(r['head_errors']) or r['route_errors']:raise RuntimeError('bounded replay failed')
    with open(a.checkpoint,'xb') as f:torch.save(model.state_dict(),f)
    with open(a.checkpoint,'rb') as f:digest=sha256(f.read()).hexdigest()
    print(json.dumps(dict(mode='dense30',width=WIDTH,hidden_used=28,sha256=digest,audit=r,training_steps=0,exact_float_certified=False,original_affine_readout_admitted=False)))


if __name__=='__main__':main()
