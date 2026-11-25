"""Nine-atom parity factorization; dense18/dense15 real, dense12 saturated FP.
Original Leech-LILA: Anatolii Kornienko; DASHI extension.
No fitting. Dense12 exactness is conditional on floating GELU/arithmetic.
"""
from itertools import product
import torch

SPATIAL=((0,0,0),(2,0,0),(2,1,0))
T=64
MODES={"dense18":(18,18),"dense15":(12,15),"dense12":(12,12)}
CHECKPOINT_KIND="x369-constructive-signed-v1"


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
def fill15(A,W,V,bias):
    """Exact-real shared-even cancellation: 12 state, 15 hidden coordinates."""
    from scripts.x369_dense30 import scalar_gate
    width=W.shape[0]
    if width!=12 or W.shape[1]!=15 or A.shape!=(15,24) or V.shape!=(12,36):
        raise ValueError("dense15 requires state width 12 and hidden width 15")
    t,alpha=scalar_gate();slot=0
    rows=list(terms())
    for l,w,_ in rows[:3]:
        l=A.new_tensor(l);w=W.new_tensor(w)
        for s in (1,-1):
            A[slot,0]=1;A[slot,width:width+9]=s*l
            W[:9,slot]=s*alpha*w;slot+=1
        V[:9,:9]-=alpha*torch.outer(w,l)
    for i in range(3):
        pair=rows[3+2*i:5+2*i]
        for l,w,coeff in pair:
            l=A.new_tensor(l);w=W.new_tensor(w)
            k=1 if coeff[1]==-1 else 2
            A[slot,k]=1;A[slot,width:width+9]=l
            W[:9,slot]=2*alpha*w;slot+=1
            V[:9,:9]-=alpha*torch.outer(w,l)
            V[:9,width+k]-=alpha*w
        lp=A.new_tensor(pair[1][0])
        w=A.new_tensor(pair[0][1])+A.new_tensor(pair[1][1])
        A[slot,3]=1;A[slot,width:width+9]=lp
        W[:9,slot]=-2*alpha*w;slot+=1
        V[:9,:9]+=(alpha+1)*torch.outer(w,lp)
        V[:9,width+3]+=alpha*w
    assert slot==15
    for j in range(3,9):V[j,2*width+j]=1
    return t


@torch.no_grad()
def fill(A,W,V,bias,mode):
    if mode=='dense15':return fill15(A,W,V,bias)
    width=W.shape[0];assert width==MODES[mode][0]
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
    width=MODES[mode][0]
    x=torch.zeros(width,dtype=dtype,device=device)
    if mode=='dense18':
        for j,(_,_,c) in enumerate(terms()):x[j]=sgn*c[i]*t
        x[9]=1
    elif mode=='dense15':
        x[0]=sgn*t if i==0 else 0
        x[1]=sgn*(1,-1,1)[i]*t;x[2]=sgn*t;x[3]=t;x[4]=1
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
    if mode!='dense12' and worst> (1e-12 if dtype==torch.float64 else 1e-5):raise RuntimeError('exact-real candidate numeric tolerance failed')
    return dict(branch_cases=count,max_abs_error=worst,value_exact=(worst==0))


@torch.no_grad()
def construct(mode,device='cpu',dtype=torch.float64):
    from scripts.x369_signed_control_factorization import action_matrix
    from scripts.x369_source_geometry import SourceGeometryOrderedLearner
    from scripts.x369_constructive_finite_source import representatives
    from models.geometric_reflection_fibre import GROUP
    width,hidden=MODES[mode]
    model=SourceGeometryOrderedLearner(width,4,expert_width=hidden,typed_consumer=True).to(device=device,dtype=dtype).eval()
    for p in model.parameters():p.zero_()
    src=model.history.source;src.history_cap=None
    src.expert_bias.copy_(src.expert_bias.new_tensor([2,1,0,0]))
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
    from scripts.audit_x369_ordered_signed_task import TRIADS
    from scripts.train_x369_ordered_action import GRAMS
    if model.training or not model.typed_consumer:raise ValueError('explicit constructive typed consumer required')
    p=next(model.parameters());dtype,device=p.dtype,p.device
    triads=list(product((-1,0,1),repeat=3));pairs=list(product(triads,repeat=2))
    alphabets=tuple(torch.tensor(x,device=device,dtype=dtype) for x in (TRIADS,GRAMS,TRIADS))
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
            tm,td=decode_history(target)
            x,y,z=td.unbind(-1);q=torch.stack((x*x,y*y,z*z,y*z,z*x,x*y),-1)
            scores=model.consume(o)
            for logits,values,alphabet in zip(scores,(tm,q,td),alphabets):
                matches=(values[:,None,:]==alphabet[None,:,:]).all(-1)
                if not (matches.sum(-1)==1).all():raise RuntimeError('independent target alphabet mismatch')
                labels=matches.to(torch.long).argmax(-1)
                if not torch.equal(logits.argmax(-1),labels):raise RuntimeError('typed consumer mismatch')
                winner=logits.gather(1,labels[:,None])
                if not (winner>logits.masked_fill(matches,-torch.inf)).all():raise RuntimeError('non-strict typed consumer margin')
            heads+=3*len(src)
    if mode=='dense18' and worst>(1e-12 if dtype==torch.float64 else 1e-5):raise RuntimeError('exact-real candidate recurrence tolerance failed')
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
    if not model.typed_consumer:raise ValueError('constructive typed consumer not enabled')
    return model.rollout_group(seed,controls,tags)


@torch.no_grad()
def self_test():
    print(check_tensor());print(affine_gram_obstruction());print(exact_scalar_and_horizon_certificates());print(constructive_quotient_certificate())
    for mode,(width,hidden) in MODES.items():
        for dtype in (torch.float64,torch.float32):
            expert=torch.nn.Sequential(torch.nn.Linear(2*width,hidden),torch.nn.GELU(),torch.nn.Linear(hidden,width)).to(dtype)
            transition=torch.nn.Linear(3*width,width).to(dtype)
            for p in list(expert.parameters())+list(transition.parameters()):p.zero_()
            t=fill(expert[0].weight,expert[2].weight,transition.weight,transition.bias,mode)
            print(mode,str(dtype),certify_branch(expert,transition,mode,t))


def exact_scalar_and_horizon_certificates():
    """Rational certificates for the common-root and finite-horizon claims.
    Taylor bounds enclose sqrt(2*pi) times the even part of GELU.
    They do not assert that arbitrary stored floating weights are exact reals.
    """
    from fractions import Fraction as Q
    def F(x):
        x=Q(x);u=x*x
        if abs(x)>5:raise ValueError("Taylor certificate domain")
        term=u;total=term
        for k in range(1,101):
            term=-term*u*Q(2*k-1,2*k*(2*k+1));total+=term
            if k==99:lo=total
        hi=total
        assert lo<=hi
        return lo,hi
    def add(a,b):return a[0]+b[0],a[1]+b[1]
    def neg(a):return -a[1],-a[0]
    def sub(a,b):return add(a,neg(b))
    def scale(a,k):return (k*a[0],k*a[1]) if k>=0 else (k*a[1],k*a[0])
    def mul(a,b):
        s=[x*y for x in a for y in b];return min(s),max(s)
    def D(t):return sub(sub(F(t+2),F(t-2)),scale(sub(F(t+1),F(t-1)),2))
    def profile(t,v):return sub(scale(add(F(t+v),F(t-v)),Q(1,2)),F(t))
    lo,hi=Q(263167,100000),Q(263169,100000)
    assert D(lo)[1]<0<D(hi)[0]
    mid=(lo+hi)/2;rad=(hi-lo)/2
    pt=[add(profile(mid,v),(-20*rad,20*rad)) for v in (1,2)]
    p0=[profile(Q(0),v) for v in (1,2)]
    det=sub(mul(p0[0],pt[1]),mul(p0[1],pt[0]))
    assert det[0]>0
    # Exact output row sum of dense12. ReLU replacement is the signed
    # action for the entire scalar interval [-6,6], not just the integers.
    columns=[]
    for _,w,c in terms():
        if c==(1,0,0):columns.extend((tuple(map(Q,w)),tuple(-Q(v) for v in w)))
        else:columns.append(tuple(2*Q(v) for v in w))
    row_sum=max(sum(abs(c[i]) for c in columns) for i in range(9))
    assert row_sum==8 and len(columns)==12
    assert Q(58**2,2)>2400*Q(7,10)
    assert sum(Q(7,10)**k / (1 if k<2 else 2 if k==2 else 6) for k in range(4))>2
    # phi(58)<2^-2400. Conservative row sum <=32 yields delta<2^-2395.
    # Persistent signed permutation is nonexpansive; nearest-integer
    # rounding fixes the ideal stabilizer within the 1/16 tube.
    assert Q(2**2390,2**2395)<Q(1,16)
    assert 6*Q(1,16)+3*Q(1,16)**2<Q(1,2)
    return dict(root_interval=(str(lo),str(hi)),
        scaled_even_profile_determinant_interval=tuple(map(float,det)),
        dense12_output_row_sum=int(row_sum),real_step_error_bound_exponent=-2395,
        finite_semantic_horizon_exponent=2390,exact_unbounded_real_state_claim=False)


def encode_constructed_value_history(m,d,mark):
    """Compact integer VALUE codec, not raw IEEE storage. Expert0 only.
    Each branch is nine occupied coordinates; other coordinates/experts zero.
    0=seed, +/-1=persistent parity. The zero state has one canonical mark.
    """
    m,d=tuple(m),tuple(d)
    if mark not in (0,-1,1):raise ValueError('bad mark')
    if len(m)!=3 or len(d)!=3 or any(v not in (-1,0,1) for v in m+d):
        raise ValueError('bad semantic triads')
    if not any(m+d) and mark!=0:raise ValueError('zero has one canonical mark')
    eta=mark or 1
    out=[]
    for b in (1,-1):
        v=tuple(x+b*eta*y for x,y in zip(m,d))
        out.append(v+(0,)*6 if mark==0 else (0,)*3+v+tuple(eta*x for x in v))
    return tuple(out)


def decode_constructed_value_history(h):
    if len(h)!=2 or any(len(p)!=9 for p in h):raise ValueError('bad branch shape')
    h=tuple(tuple(p) for p in h);a,b=h
    if any(type(x)!=int for p in h for x in p):raise ValueError('integer value carrier required')
    m=tuple((a[j]+a[3+j]+b[j]+b[3+j])//2 for j in range(3))
    d=tuple((a[j]+a[6+j]-b[j]-b[6+j])//2 for j in range(3))
    if any(x for p in h for x in p[:3]) or not any(x for p in h for x in p):mark=0
    else:
        marks=[eta for eta in (-1,1) if all(p[6+j]==eta*p[3+j] for p in h for j in range(3))]
        if len(marks)!=1:raise ValueError('not a persistent parity carrier')
        mark=marks[0]
    if encode_constructed_value_history(m,d,mark)!=h:raise ValueError('not admitted history')
    return m,d,mark


def constructed_value_action(g,v):
    k,b,a=g
    return tuple((-1)**a*v[(2-i-k)%3 if b else (i+k)%3] for i in range(3))


def step_constructed_value_history(h,g):
    _,b,_=g
    return tuple((0,)*3+constructed_value_action(g,tuple(p[j]+p[3+j] for j in range(3)))+
        tuple((-1)**b*x for x in constructed_value_action(g,tuple(p[j]+p[6+j] for j in range(3)))) for p in h)


def step_constructed_code(code,g):
    m,d,mark=code;_,b,_=g
    mm=constructed_value_action(g,m)
    dd=tuple((-1)**b*x for x in constructed_value_action(g,d))
    eta=(-1)**b*(mark or 1)
    return mm,dd,eta if any(mm+dd) else 0


def constructive_quotient_certificate():
    """Exact future-typed-consumer sufficiency/minimality on declared carrier.
    Identity control separates distinct (M,D); arbitrary words follow from
    successor closure. Not the raw-output or arbitrary fitted-state quotient.
    """
    triads=tuple(product((-1,0,1),repeat=3))
    group=tuple(product(range(3),range(2),range(2)))
    states={encode_constructed_value_history(m,d,k) for m,d,k in
        product(triads,triads,(0,-1,1)) if any(m+d) or k==0}
    assert len(states)==2185
    classes={};edges=0
    for h in states:
        c=decode_constructed_value_history(h)
        classes.setdefault(c[:2],set()).add(c[2])
        assert encode_constructed_value_history(*c)==h
        for g in group:
            nxt=step_constructed_value_history(h,g);cc=step_constructed_code(c,g)
            assert nxt in states and decode_constructed_value_history(nxt)==cc
            assert encode_constructed_value_history(*cc)==nxt
            edges+=1
    assert len(classes)==729 and edges==26220
    assert sum(len(v)==3 for v in classes.values())==728
    assert classes[((0,0,0),(0,0,0))]=={0}
    assert len({step_constructed_code((m,d,0),(0,0,0))[:2] for m,d in classes})==729
    return dict(physical_value_states=len(states),physical_edges=edges,
        semantic_classes=len(classes),semantic_edges=len(classes)*len(group),
        nonzero_three_marks=728,zero_marks=1)


def constructive_source_hashes():
    """Content provenance, independent of unrelated documentation commits."""
    import hashlib
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    paths=("scripts/x369_dense_compact.py","scripts/x369_dense30.py",
        "scripts/x369_source_geometry.py","scripts/x369_constructive_finite_source.py",
        "scripts/x369_signed_control_factorization.py","scripts/x369_typed_prototype_readout.py",
        "scripts/audit_x369_ordered_signed_task.py","scripts/train_x369_ordered_action.py",
        "models/history_orbit_moe.py","models/geometric_paired_history.py",
        "models/geometric_orbit_descent.py","models/geometric_reflection_fibre.py",
        "models/geometric_history_adapter.py","models/trialectic_wave_prior.py",
        "models/geometric_bitlinear.py","models/punctured_nine_wave_prior.py",
        "models/e8_root_prior.py","models/c3_matched_router_control.py")
    return {p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}


def checkpoint_payload(model,mode,dtype,branch,recurrent):
    if (model.width,model.expert_width)!=MODES[mode] or not model.typed_consumer:
        raise ValueError("constructive model metadata mismatch")
    return dict(kind=CHECKPOINT_KIND,mode=mode,dtype=dtype,width=model.width,
        expert_width=model.expert_width,consumer="typed_prototype",
        branch=branch,recurrent=recurrent,source_hashes=constructive_source_hashes(),
        runtime=dict(torch=str(torch.__version__),hip=getattr(torch.version,'hip',None),
                     device=str(next(model.parameters()).device)),state_dict=model.state_dict())


@torch.no_grad()
def load_constructive_checkpoint(path,expected_sha256,device):
    """Explicit constructive schema; never silently use the zero affine head.
    Replay all deterministic branch/recurrent gates on the receiving backend.
    Historical raw state dictionaries are intentionally not migrated.
    """
    import hashlib,io
    with open(path,"rb") as f:raw=f.read()
    if hashlib.sha256(raw).hexdigest()!=expected_sha256:
        raise ValueError("checkpoint hash mismatch")
    packet=torch.load(io.BytesIO(raw),weights_only=True,map_location="cpu")
    if not isinstance(packet,dict) or packet.get("kind")!=CHECKPOINT_KIND or packet.get("consumer")!="typed_prototype":
        raise ValueError("explicit constructive checkpoint schema required")
    if packet.get('source_hashes')!=constructive_source_hashes():
        raise ValueError('constructive source fingerprint mismatch')
    mode=packet.get("mode");dtype=packet.get("dtype")
    if mode not in MODES or dtype not in ("float32","float64") or (packet.get("width"),packet.get("expert_width"))!=MODES[mode]:
        raise ValueError("checkpoint architecture mismatch")
    device=torch.device(device)
    if device.type not in ("cpu","cuda"):raise ValueError("explicit CPU or native gfx803 device required")
    if device.type=="cuda" and (not torch.cuda.is_available() or not torch.version.hip or not getattr(torch.cuda.get_device_properties(device),"gcnArchName","").startswith("gfx803")):
        raise RuntimeError("native gfx803 required")
    model,t=construct(mode,device,getattr(torch,dtype))
    expected=model.state_dict();state=packet.get("state_dict")
    if not isinstance(state,dict) or set(state)!=set(expected):raise ValueError("state schema mismatch")
    for k,v in state.items():
        if not isinstance(v,torch.Tensor) or v.dtype!=expected[k].dtype or v.shape!=expected[k].shape:
            raise ValueError("state tensor schema mismatch: "+k)
        # This is a deterministic constructive receipt, not a fitted loader.
        if not torch.equal(v,expected[k].cpu()):raise ValueError("state differs from constructed parameters: "+k)
    model.load_state_dict(state,strict=True)
    src=model.history.source
    return model,dict(branch=certify_branch(src.experts[0],src.transitions[0],mode,t),recurrent=certify_recurrent(model,mode))


def main():
    import argparse,json,hashlib
    p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true');p.add_argument('--mode',choices=tuple(MODES));p.add_argument('--device',choices=('cpu','cuda'));p.add_argument('--dtype',choices=('float32','float64'),default='float64');p.add_argument('--checkpoint');a=p.parse_args()
    if a.self_test:self_test();return
    if not a.mode or not a.device or not a.checkpoint:p.error('explicit mode, device, checkpoint required')
    if a.device=='cuda' and (not torch.cuda.is_available() or not torch.version.hip or not getattr(torch.cuda.get_device_properties(0),'gcnArchName','').startswith('gfx803')):raise RuntimeError('native gfx803 required')
    model,t=construct(a.mode,a.device,getattr(torch,a.dtype));src=model.history.source
    branch=certify_branch(src.experts[0],src.transitions[0],a.mode,t)
    rec=certify_recurrent(model,a.mode)
    with open(a.checkpoint,'xb') as f:torch.save(checkpoint_payload(model,a.mode,a.dtype,branch,rec),f)
    with open(a.checkpoint,'rb') as f:sha=hashlib.sha256(f.read()).hexdigest()
    print(json.dumps(dict(mode=a.mode,width=model.width,expert_width=model.expert_width,consumer='typed_prototype',sha256=sha,branch=branch,recurrent=rec,training_steps=0,exact_real=(a.mode!='dense12'),original_affine_readout_admitted=False,source_byte_identity=False)))


if __name__=='__main__':main()
