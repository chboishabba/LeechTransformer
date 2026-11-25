"""Parity-indexed reflection eigenbasis for the original dense GELU source.
Original Leech-LILA: Anatolii Kornienko; DASHI extension.
Exact-real finite carrier; numeric materialization is not float closure.
"""
from itertools import product, combinations, islice
import torch
from scripts.x369_dense30 import scalar_gate, SPATIAL

WIDTH=24


def terms():
    # Input v=s_j+(m_j+d_j)/2 + sign*(m_f-d_f)/2, f=-j mod 3.
    # In parity coordinates the dual vector is (e_j+sign*o_f)/2.
    for j,sign,rotate in product(range(3),(-1,1),range(2)):
        f=(-j)%3
        v=[0.]*9;v[j]=1.;v[3+j]+=0.5;v[6+j]+=0.5
        v[3+f]+=0.5*sign;v[6+f]-=0.5*sign
        w=[0.]*9
        p=(j+rotate)%3;q=(f+rotate)%3
        w[3+p]+=0.5;w[6+p]+=0.5
        w[3+q]+=0.5*sign;w[6+q]-=0.5*sign
        yield tuple(v),tuple(w),(0,1,0) if rotate else (1,0,sign)


@torch.no_grad()
def fill(A,W,T):
    t,alpha=scalar_gate()
    for i,(v,w,_) in enumerate(terms()):
        v=A.new_tensor(v);w=W.new_tensor(w)
        for b in range(2):
            sign=(-1)**b;row=2*i+b
            A[row,i]=1;A[row,WIDTH:WIDTH+9]=sign*v
            W[:9,row]=sign*alpha*w
        T[:9,:9]-=alpha*torch.outer(w,v)
    for j in range(3,9):T[j,2*WIDTH+j]=1
    return t


def embedding(g,t,dtype,device):
    k,b,a=g;i=SPATIAL.index((k,b,0));x=torch.zeros(WIDTH,dtype=dtype,device=device)
    for j,(_,_,c) in enumerate(terms()):x[j]=(-1)**a*c[i]*t
    x[12]=1
    return x


@torch.no_grad()
def construct(device='cpu',dtype=torch.float64):
    from scripts.x369_signed_control_factorization import common,action_matrix
    from scripts.x369_constructive_finite_source import representatives
    from models.geometric_reflection_fibre import GROUP
    model=common(WIDTH,device,dtype);src=model.history.source
    # Verify the actual source's unsigned action matrices, not guessed order.
    for i,g in enumerate(SPATIAL):
        want=torch.zeros(9,9,dtype=torch.float64)
        for lane in range(2):
            P=torch.tensor(action_matrix(g,lane),dtype=want.dtype);j=3*(lane+1)
            want[j:j+3,:3]=P;want[j:j+3,j:j+3]=P
        got=sum(c[i]*torch.outer(torch.tensor(w,dtype=want.dtype),torch.tensor(v,dtype=want.dtype)) for v,w,c in terms())
        if not torch.equal(got,want):raise RuntimeError('original action convention mismatch')
    t=fill(src.experts[0][0].weight,src.experts[0][2].weight,src.transitions[0].weight)
    src.orientation.weight[0,12]=1
    for tag,u in product(range(4),representatives()):
        model.control_embed.weight[tag*len(GROUP)+u].copy_(embedding(GROUP[u],t,dtype,device))
    return model


def basis_audit():
    """Exhaustive exact finite BASIS-family bound, not arbitrary tensor rank.
    Rows are single coordinates or e_i +/- o_j; orientation up to row sign.
    Mod 101 rank is exact for 6x6 input bases by Hadamard (det <=8),
    and for 6x3 dual action matrices after verified integer scaling (minors <101).
    """
    import numpy as np
    p=101;I=np.eye(6,dtype=np.int64);r=np.roll(np.eye(3,dtype=np.int64),1,axis=0)
    f=np.eye(3,dtype=np.int64)[[0,2,1]];z=np.zeros((3,3),dtype=np.int64)
    A=np.block([[r,z],[z,r]]);B=np.block([[z,f],[f,z]])
    rows=np.array(list(I)+[I[i]+s*I[j+3] for i,j,s in product(range(3),range(3),(-1,1))])
    # Vectorized exact modular RREF. For inverse use appended identity.
    inverses=np.array([0]+[pow(i,-1,p) for i in range(1,p)],dtype=np.int64)
    def reduce(X,ncols):
        X=X.copy()%p;n,rr,_=X.shape;ix=np.arange(n);rank=np.zeros(n,dtype=np.int64)
        for c in range(ncols):
            mask=(np.arange(rr)[None,:]>=rank[:,None])&(X[:,:,c]!=0)
            ok=mask.any(1);a=ix[ok];b=rank[ok];q=mask.argmax(1)[ok]
            X[a,b],X[a,q]=X[a,q].copy(),X[a,b].copy()
            pivot=X[a,b]*inverses[X[a,b,c]][:,None]%p
            X[a,b]=pivot;factors=X[a,:,c].copy();factors[np.arange(len(a)),b]=0
            X[a]=(X[a]-factors[:,:,None]*pivot[:,None,:])%p
            rank+=ok
        return rank,X
    it=combinations(range(len(rows)),6);total=good=0;best=19;hist={}
    while True:
        batch=list(islice(it,1024))
        if not batch:break
        L=rows[np.array(batch)];total+=len(L)
        rank,X=reduce(np.concatenate((L,np.broadcast_to(I,L.shape)),axis=2),6)
        keep=rank==6;good+=int(keep.sum());L=L[keep];X=X[keep]
        if not len(L):continue
        V=2*X[:,:,6:]%p;V=np.where(V>p//2,V-p,V)
        assert np.max(abs(V))<=2 and np.array_equal(L@V,np.broadcast_to(2*I,V.shape))
        Y=np.stack((V,A@V,B@V),axis=-1).transpose(0,2,1,3).reshape(-1,6,3)
        ranks,_=reduce(Y,3);sums=ranks.reshape(-1,6).sum(1)
        best=min(best,int(sums.min()))
        for k,n in zip(*np.unique(sums,return_counts=True)):hist[int(k)]=hist.get(int(k),0)+int(n)
    assert total==134596 and good==50086 and best==12
    return dict(candidate_bases=total,invertible=good,minimum_terms=best,rank_histogram=hist)


@torch.no_grad()
def self_test():
    import sympy as S
    v=torch.tensor(list(product(range(-2,3),repeat=3)),dtype=torch.float64)
    # Exact rational linear maps on full 9-coordinate carrier, even though
    # the scalar gate is exact only on the invariant parity-exclusive subset.
    for i,(k,b,a) in enumerate(SPATIAL):
        got=sum((S.Matrix(w)*S.Matrix(v).T*c[i] for v,w,c in terms()),S.zeros(9))
        want=S.zeros(9)
        for lane,j in product(range(2),range(3)):
            dst=((2-j if b else j)-k)%3;out=3*(lane+1)+dst
            want[out,j]=(-1)**(b*lane);want[out,3*(lane+1)+j]=(-1)**(b*lane)
        assert got.applyfunc(S.Rational)==want
    h=torch.zeros(125+2*125,WIDTH,dtype=v.dtype)
    h[:125,:3]=v;h[125:250,3:6]=v;h[125:250,6:9]=v
    h[250:,3:6]=v;h[250:,6:9]=-v
    # Two lane signs, zero duplicated deliberately. Exact gate alphabet.
    assert all(float((h[:,:9]@torch.tensor(l,dtype=v.dtype)).abs().max())<=2 for l,_,_ in terms())
    for dtype in (torch.float64,torch.float32):
        A=torch.zeros(WIDTH,2*WIDTH,dtype=dtype);W=torch.zeros(WIDTH,WIDTH,dtype=dtype);T=torch.zeros(WIDTH,3*WIDTH,dtype=dtype)
        t=fill(A,W,T);hh=h.to(dtype);worst=0.;n=0
        for k,b,_ in SPATIAL:
            for a in (0,1):
                x=embedding((k,b,a),t,dtype,'cpu').expand(len(hh),-1)
                y=torch.nn.functional.gelu(torch.cat((x,hh),-1)@A.T)@W.T
                nxt=torch.cat((hh,x,y),-1)@T.T;want=torch.zeros_like(hh)
                for lane,j in product(range(2),range(3)):
                    dst=((2-j if b else j)-k)%3;start=3*(lane+1)
                    want[:,start+dst]=(-1)**(a+b*lane)*(hh[:,j]+hh[:,start+j])
                worst=max(worst,float((nxt-want).abs().max()));n+=len(hh)
        assert worst < (1e-12 if dtype==torch.float64 else 1e-5)
        print(str(dtype),n,worst)
    print(basis_audit())


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
    print(json.dumps(dict(mode='dense24',width=WIDTH,hidden_used=24,sha256=digest,audit=r,training_steps=0,exact_float_certified=False,original_affine_readout_admitted=False)))


if __name__=='__main__':main()
