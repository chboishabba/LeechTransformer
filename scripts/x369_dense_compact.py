"""Nine-atom parity factorization; dense18 real, dense12 saturated FP.
Original Leech-LILA: Anatolii Kornienko; DASHI extension.
No fitting. Dense12 exactness is conditional on floating GELU/arithmetic.
"""
from itertools import product
import torch

SPATIAL=((0,0,0),(2,0,0),(2,1,0))
T=64


def atoms():
    """Parity order (e0,e1,e2,o0,o1,o2), l,w,c."""
    for i in range(3):
        j=(-i)%3
        l=[0]*6;l[i]=1;l[3+j]=-1
        yield tuple(l),tuple(l),(1,0,0)
    for i,sig in product(range(3),(-1,1)):
        l=[0]*6;l[i]=1;l[3+(-i-1)%3]=sig
        w=[0.]*6;w[(i+1)%3]=sig/2;w[3+(-i)%3]=0.5
        yield tuple(l),tuple(w),(1,sig,1)


def terms():
    for l,w,c in atoms():
        yield (tuple(l[:3])+tuple((l[i]+l[3+i])/2 for i in range(3))+
               tuple((l[i]-l[3+i])/2 for i in range(3)),
               (0,0,0)+tuple(w[i]+w[3+i] for i in range(3))+
               tuple(w[i]-w[3+i] for i in range(3)),c)


def check_tensor():
    import sympy as S
    I=S.eye(6);R=S.Matrix([[0,0,1],[1,0,0],[0,1,0]]);F=S.Matrix([[1,0,0],[0,0,1],[0,1,0]]);Z=S.zeros(3)
    slices=(I,S.diag(R,R),Z.row_join(F).col_join(F.row_join(Z)))
    for i,P in enumerate(slices):
        got=sum((S.Matrix(w)*S.Matrix(l).T*c[i] for l,w,c in atoms()),S.zeros(6)).applyfunc(S.Rational)
        assert got==P
    ranks=[S.Matrix.hstack(*(P[:,j] for P in slices)).rank() for j in range(6)]
    assert ranks==[3]*6 and len(list(atoms()))==9
    # Each allowed input/control atom touches at most two input vertices.
    return dict(slices=3,column_ranks=ranks,minimum_atoms=sum(ranks)//2)


@torch.no_grad()
def fill(A,W,V,bias,mode):
    width=W.shape[0];assert width== (18 if mode=='dense18' else 12)
    if mode=='dense18':
        from scripts.x369_dense30 import scalar_gate
        t,alpha=scalar_gate()
    else:t,alpha=T,None
    slot=0
    for i,(l,w,c) in enumerate(terms()):
        l=A.new_tensor(l);w=W.new_tensor(w)
        if mode=='dense18':
            for s in (1,-1):
                A[slot,i]=1;A[slot,width:width+9]=s*l;W[:9,slot]=s*alpha*w;slot+=1
            V[:9,:9]-=alpha*torch.outer(w,l)
        elif c==(1,0,0):
            # Two binary gates for +1 and -1; both disabled for zero.
            for k,s in ((0,1),(1,-1)):
                A[slot,k]=1;A[slot,width:width+9]=l;W[:9,slot]=s*w;slot+=1
                V[:9,width+k]-=s*w/2
        else:
            # One binary gate plus a linear baseline gives either +/-v.
            k=2 if c[1]==-1 else 3
            A[slot,k]=1;A[slot,width:width+9]=l;W[:9,slot]=2*w;slot+=1
            V[:9,:9]-=torch.outer(w,l);V[:9,width+k]-=w;bias[:9]-=T*w
    assert slot==width
    for j in range(3,9):V[j,2*width+j]=1
    return t


def embedding(g,t,mode,dtype,device):
    k,b,a=g;i=SPATIAL.index((k,b,0));sgn=(-1)**a
    width=18 if mode=='dense18' else 12
    x=torch.zeros(width,dtype=dtype,device=device)
    if mode=='dense18':
        for j,(_,_,c) in enumerate(terms()):x[j]=sgn*c[i]*t
        x[9]=1
    else:
        a0=sgn if i==0 else 0
        x[0]=T if a0==1 else -T;x[1]=T if a0==-1 else -T
        x[2]=T*sgn*(1,-1,1)[i];x[3]=T*sgn;x[4]=1
    return x


def saturation_contract(device,dtype):
    if dtype not in (torch.float32,torch.float64):raise ValueError('only FP32/FP64')
    x=torch.tensor([s*T+v for s in (-1,1) for v in range(-2,3)],device=device,dtype=dtype)
    got=torch.nn.GELU(approximate='none')(x)
    if not torch.equal(got,x.clamp_min(0)):raise RuntimeError('backend GELU saturation failed')
    return 10


def expected(h,g):
    k,b,a=g;want=torch.zeros_like(h)
    for lane,j in product(range(2),range(3)):
        dst=((2-j if b else j)-k)%3;start=3*(lane+1)
        want[...,start+dst]=(-1)**(a+b*lane)*(h[...,j]+h[...,start+j])
    return want


def branch_carrier(width,device,dtype):
    v=torch.tensor(list(product(range(-2,3),repeat=3)),device=device,dtype=dtype)
    h=torch.zeros(375,width,device=device,dtype=dtype)
    h[:125,:3]=v;h[125:250,3:6]=v;h[125:250,6:9]=v
    h[250:,3:6]=v;h[250:,6:9]=-v
    return h


@torch.no_grad()
def certify_branch(expert,transition,mode,t):
    p=next(expert.parameters());dtype,device=p.dtype,p.device;width=transition.out_features
    if mode=='dense12':saturation_contract(device,dtype)
    h=branch_carrier(width,device,dtype);worst=0.;count=0
    for k,b,_ in SPATIAL:
        for a in (0,1):
            g=(k,b,a);x=embedding(g,t,mode,dtype,device).expand(len(h),-1)
            y=expert(torch.cat((x,h),-1));got=transition(torch.cat((h,x,y),-1));want=expected(h,g)
            if not torch.isfinite(got).all():raise RuntimeError('nonfinite branch output')
            err=float((got-want).abs().max());worst=max(worst,err);count+=len(h)
            if mode=='dense12' and not torch.equal(got,want):raise RuntimeError('exact FP branch contract failed')
    if mode=='dense18' and worst> (1e-12 if dtype==torch.float64 else 1e-5):raise RuntimeError('dense18 numeric tolerance failed')
    return dict(branch_cases=count,max_abs_error=worst,value_exact=(worst==0))


@torch.no_grad()
def construct(mode,device='cpu',dtype=torch.float64):
    from scripts.x369_signed_control_factorization import common,action_matrix
    from scripts.x369_constructive_finite_source import representatives
    from models.geometric_reflection_fibre import GROUP
    width={'dense18':18,'dense12':12}[mode];model=common(width,device,dtype);src=model.history.source
    # Guard exact 9-coordinate linear action convention against original owner.
    for i,g in enumerate(SPATIAL):
        want=torch.zeros(9,9,dtype=torch.float64)
        for lane in range(2):
            P=torch.tensor(action_matrix(g,lane),dtype=want.dtype);j=3*(lane+1)
            want[j:j+3,:3]=P;want[j:j+3,j:j+3]=P
        got=sum(c[i]*torch.outer(torch.tensor(w,dtype=want.dtype),torch.tensor(l,dtype=want.dtype)) for l,w,c in terms())
        if not torch.equal(got,want):raise RuntimeError('original action convention mismatch')
    t=fill(src.experts[0][0].weight,src.experts[0][2].weight,src.transitions[0].weight,src.transitions[0].bias,mode)
    src.orientation.weight[0,9 if mode=='dense18' else 4]=1
    for tag,u in product(range(4),representatives()):
        model.control_embed.weight[tag*len(GROUP)+u].copy_(embedding(GROUP[u],t,mode,dtype,device))
    return model,t


@torch.no_grad()
def certify_recurrent(model,mode,batch=128):
    """All reachable seed and persistent pairs, all 12 controls; tags equal.
    Exact value closure for dense12, tolerance only for dense18. No fit.
    """
    from models.geometric_reflection_fibre import GROUP
    from scripts.x369_source_geometry import signed_seed
    from scripts.x369_constructive_finite_source import decode_history
    from scripts.x369_typed_prototype_readout import TypedPrototypeReadout
    readout=TypedPrototypeReadout().to(next(model.parameters()).device)
    p=next(model.parameters());dtype,device=p.dtype,p.device
    triads=list(product((-1,0,1),repeat=3));pairs=list(product(triads,repeat=2))
    seeds=torch.tensor([m+d+(0,0,0) for m,d in pairs],device=device,dtype=dtype)
    init=signed_seed(seeds,model.experts,model.width)
    persistent=[]
    for eta in (1,-1):
        h=init.clone();v=init[:,0,:,:3]
        # Reindex branch seed for odd parity, to keep both semantic labels.
        if eta==-1:v=v.flip(1)
        h.zero_();h[:,0,:,3:6]=v;h[:,0,:,6:9]=eta*v;persistent.append(h)
    h=torch.cat((init,*persistent),0)
    assert torch.equal(model.control_embed.weight.view(4,len(GROUP),-1),model.control_embed.weight.view(4,len(GROUP),-1)[:1].expand(4,-1,-1))
    count=0;worst=0.;heads=0
    for u,g in enumerate(GROUP):
        for a in range(0,len(h),batch):
            src=h[a:a+batch];ctrl=torch.full((len(src),),u,device=device,dtype=torch.long)
            o=model.history.step_controlled(ctrl,src,model.control_embed,train_sign=False)
            target=expected(src,g)
            err=float((o.history-target).abs().max());worst=max(worst,err);count+=len(src)
            if not torch.isfinite(o.history).all() or not (o.ids==ctrl.new_tensor([0,1])).all():raise RuntimeError('route/history contract')
            if mode=='dense12' and not torch.equal(o.history,target):raise RuntimeError('exact FP recurrent value closure failed')
            m,d=decode_history(o.history);tm,td=decode_history(target)
            scores=readout(m,d);truth=readout(tm,td)
            if any(not torch.equal(a.argmax(-1),b.argmax(-1)) for a,b in zip(scores,truth)):raise RuntimeError('typed consumer mismatch')
            heads+=3*len(src)
    if mode=='dense18' and worst>(1e-12 if dtype==torch.float64 else 1e-5):raise RuntimeError('dense18 recurrence tolerance failed')
    return dict(recurrent_edges=count,max_abs_history_error=worst,value_exact=(worst==0),tag_equality=True,typed_head_checks=heads,byte_identity_claim=False)


def affine_gram_obstruction():
    """Exact controlled chart midpoint witness, independent of fitted head.
    Applies to real affine local features, not arbitrary rounded raw outputs.
    """
    G=tuple(product(range(3),range(2),range(2)))
    def mul(g,h):
        k,b,a=g;l,d,c=h;return ((k+(-1)**b*l)%3,b^d,a^c)
    def inv(g):
        k,b,a=g;return (-(-1)**b*k%3,b,a)
    def conj(g,u):return mul(mul(g,u),inv(g))
    def act(g,v):
        k,b,a=g;v=v[::-1] if b else v;return tuple((-1)**a*x for x in v[k:]+v[:k])
    def pair(g,h):return act(g,h[g[1]]),act(g,h[1-g[1]])
    u=(2,0,0);m=(-1,-1,-1);res=[(-1,-1,-1),(1,1,1),(0,0,0)]
    out=[]
    for d in res:
        h=tuple(a+b for a,b in zip(m,d)),tuple(a-b for a,b in zip(m,d))
        cc,hh=max((conj(g,u),pair(g,h)) for g in G)
        mark=next(g for g in G if (conj(g,u),pair(g,h))==(cc,hh))
        stab=tuple(g for g in G if (conj(g,cc),pair(g,hh))==(cc,hh))
        out.append((cc,mark,stab,hh))
    assert out[0][:3]==out[1][:3]==out[2][:3]
    assert all(a+b==2*c for x,y,z in zip(*(o[3] for o in out)) for a,b,c in zip(x,y,z))
    return dict(control=u,mean=m,residuals=res,mark=out[0][1],stabilizer=out[0][2],gram_classes=('all-ones','all-ones','zero'),real_affine_head_impossible=True)


def predict(model,seed,controls,tags):
    from scripts.x369_constructive_finite_source import decode_history
    from scripts.x369_typed_prototype_readout import TypedPrototypeReadout
    _,trace=model.rollout_group(seed,controls,tags)
    m,d=decode_history(trace[-1].history)
    return TypedPrototypeReadout().to(m.device)(m,d),trace


@torch.no_grad()
def self_test():
    print(check_tensor());print(affine_gram_obstruction())
    for mode,width in (('dense18',18),('dense12',12)):
        for dtype in (torch.float64,torch.float32):
            expert=torch.nn.Sequential(torch.nn.Linear(2*width,width),torch.nn.GELU(),torch.nn.Linear(width,width)).to(dtype)
            transition=torch.nn.Linear(3*width,width).to(dtype)
            for p in list(expert.parameters())+list(transition.parameters()):p.zero_()
            t=fill(expert[0].weight,expert[2].weight,transition.weight,transition.bias,mode)
            print(mode,str(dtype),certify_branch(expert,transition,mode,t))


def main():
    import argparse,json,hashlib
    p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true');p.add_argument('--mode',choices=('dense18','dense12'));p.add_argument('--device',choices=('cpu','cuda'));p.add_argument('--dtype',choices=('float32','float64'),default='float64');p.add_argument('--checkpoint');a=p.parse_args()
    if a.self_test:self_test();return
    if not a.mode or not a.device or not a.checkpoint:p.error('explicit mode, device, checkpoint required')
    if a.device=='cuda' and (not torch.cuda.is_available() or not torch.version.hip or not getattr(torch.cuda.get_device_properties(0),'gcnArchName','').startswith('gfx803')):raise RuntimeError('native gfx803 required')
    model,t=construct(a.mode,a.device,getattr(torch,a.dtype));src=model.history.source
    branch=certify_branch(src.experts[0],src.transitions[0],a.mode,t)
    rec=certify_recurrent(model,a.mode)
    with open(a.checkpoint,'xb') as f:torch.save(model.state_dict(),f)
    with open(a.checkpoint,'rb') as f:sha=hashlib.sha256(f.read()).hexdigest()
    print(json.dumps(dict(mode=a.mode,sha256=sha,branch=branch,recurrent=rec,training_steps=0,exact_real=(a.mode=='dense18'),original_affine_readout_admitted=False,source_byte_identity=False)))


if __name__=='__main__':main()
