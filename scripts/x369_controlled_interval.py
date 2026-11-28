"""M5 conditional rational enclosure of the controlled source.
Original Leech-LILA: Anatolii Kornienko; DASHI extension.
External per-stage error bounds are hypotheses, not calibrated HIP proofs.
"""
from fractions import Fraction as Q
from math import isqrt
from functools import lru_cache
from models.geometric_reflection_fibre import GROUP,act,mul,inv
from scripts.x369_guard_geometry import conjugate,transform,flatten,shape
from scripts.x369_r4_interval_certificate import Interval,point,total,dot,fused_activation,selection_certificate,selected_softmax

BITS=48
SCALE=1<<BITS
STAGES=('router','orientation','softmax','preactivation','gelu','expert','transition','cap','mixture','projection','readout','label_projection')

TYPED_STAGES=('semantic_decode','typed_gram','typed_logits')

class Unresolved(ValueError):pass

def fixed_orbit_projection(v,K,error=Q(0)):
    """Shared signed orbit mean, with a single external error per orbit.
    Exact zero for contradictory sign constraints, assuming finite inputs.
    Interval equality alone does not establish correlated native guards.
    """
    if not K or GROUP[0] not in K or any(mul(g,h) not in K for g in K for h in K):
        raise ValueError('not a subgroup')
    E,W=len(v),len(v[0][0])
    if W%3:raise ValueError('complete triads required')
    out=[]
    for e in v:
        dest=[[None]*W for _ in range(2)]
        for t in range(0,W,3):
            x=tuple(e[b][t+j] for b in range(2) for j in range(3))
            unseen=set(range(6))
            while unseen:
                i=min(unseen);b,j=divmod(i,3);images={};zero=False
                for k,f,a in K:
                    d=3*(b^f)+((2-j if f else j)-k)%3;s=(-1)**a
                    if d in images and images[d]!=s:zero=True
                    images[d]=s
                unseen.difference_update(images)
                m=point(0) if zero else widen(total(point(s)*x[d] for d,s in sorted(images.items()))*point(Q(1,len(images))),error)
                for d,s in images.items():dest[d//3][t+d%3]=point(s)*m
        out.append(tuple(map(tuple,dest)))
    return tuple(out)

def typed_decode_box(h,error=Q(0)):
    if len(h)<2 or len(h[0][0])<9:raise ValueError('typed decoder width')
    a,b=h[0]
    m=tuple(widen(point(Q(1,2))*(a[j]+a[j+3]+b[j]+b[j+3]),error) for j in range(3))
    d=tuple(widen(point(Q(1,2))*(a[j]+a[j+6]-b[j]-b[j+6]),error) for j in range(3))
    return m,d

def typed_logits_box(m,d,gram_error=Q(0),logit_error=Q(0)):
    from scripts.audit_x369_ordered_signed_task import TRIADS,gram
    G=tuple(sorted(set(map(gram,TRIADS))));metric=(1,1,1,2,2,2)
    q=tuple(widen(x,gram_error) for x in (square(d[0]),square(d[1]),square(d[2]),d[1]*d[2],d[2]*d[0],d[0]*d[1]))
    def scores(x,labels,weights):
        return tuple(widen(total(point(2*w*y)*v-point(w*y*y) for v,y,w in zip(x,p,weights)),logit_error) for p in labels)
    return scores(m,TRIADS,(1,)*3),scores(q,G,metric),scores(d,TRIADS,(1,)*3)

def unique_margin(logits):
    """Positive lower margin certifies unique argmax throughout box."""
    i=max(range(len(logits)),key=lambda j:logits[j].lo)
    margin=min(logits[i].lo-v.hi for j,v in enumerate(logits) if j!=i)
    return (i,margin) if margin>0 else (None,margin)


def box(x):return x if isinstance(x,Interval) else point(x)
def outward(v):
    a,b=v.lo*SCALE,v.hi*SCALE
    return Interval(Q(a.numerator//a.denominator,SCALE),Q(-((-b.numerator)//b.denominator),SCALE))
def widen(v,e):return outward(v+Interval(-e,e))
def absolute(v):return Interval(0 if v.lo<=0<=v.hi else min(abs(v.lo),abs(v.hi)),max(abs(v.lo),abs(v.hi)))
def square(v):
    a=absolute(v);return Interval(a.lo*a.lo,a.hi*a.hi)
def sqrt_box(v):
    if v.lo<0:raise ValueError('negative radicand')
    def floor(x):return isqrt((x.numerator*SCALE*SCALE)//x.denominator)
    a,b=floor(v.lo),floor(v.hi)
    return Interval(Q(a,SCALE),Q(b,SCALE) if Q(b*b,SCALE*SCALE)==v.hi else Q(b+1,SCALE))

# Machin pi identity; alternating arctan partial sums have adjacent bounds.
def atan_bounds(x,n=40):
    a=sum(((-1)**k*x**(2*k+1)/(2*k+1) for k in range(n)),Q(0));b=a+(-1)**n*x**(2*n+1)/(2*n+1)
    return Interval(min(a,b),max(a,b))
PI=point(16)*atan_bounds(Q(1,5))-point(4)*atan_bounds(Q(1,239))
ROOT=sqrt_box(point(2)*PI)
NORMAL=Interval(1/ROOT.hi,1/ROOT.lo)

@lru_cache(maxsize=2048)
def phi_point(x):
    x=Q(x)
    if x<0:return point(1)-phi_point(-x)
    if x>4:return Interval(phi_point(Q(4)).lo,1)
    # Integral exp(-t^2/2) dt; Taylor remainder has next-term sign
    # for t>=0, even before alternating term magnitudes decrease.
    term=x;a=Q(0)
    for k in range(64):
        a+=term;term=-term*x*x*(2*k+1)/(2*(k+1)*(2*k+3))
    p=point(Q(1,2))+NORMAL*Interval(min(a,a+term),max(a,a+term))
    return outward(Interval(max(0,p.lo),min(1,p.hi)))

def gelu_box(v):
    # Phi is monotone; GELU is NOT globally monotone.
    return outward(v*Interval(phi_point(v.lo).lo,phi_point(v.hi).hi))

def cap_box(v,c):
    if c is None:return tuple(v),'uncapped'
    c=Q(c)
    if c<=0:raise ValueError('positive cap required')
    n=sqrt_box(total(square(x) for x in v));floor=Q(1,10**30)
    alpha=Interval(min(1,c/max(n.hi,floor)),min(1,c/max(n.lo,floor)))
    regime='inside' if max(n.hi,floor)<=c else 'outside' if max(n.lo,floor)>c else 'crossing'
    return tuple(outward(alpha*x) for x in v),regime

def signed_box(i,values):return values[abs(i)-1] if i>0 else -values[abs(i)-1]

def lex_compare(a,b,values):
    """Equal signed coordinates are correlated, not independent intervals."""
    for i,j in zip(a,b):
        if i==j:continue
        d=point(2)*signed_box(i,values) if i==-j else signed_box(i,values)-signed_box(j,values)
        if d.lo>0:return 1
        if d.hi<0:return -1
        if d.lo!=0 or d.hi!=0:return None
    return 0

def choose_max(candidates,values):
    def midpoint(ids):return tuple((signed_box(i,values).lo+signed_box(i,values).hi)/2 for i in ids)
    mark,ids=max(candidates,key=lambda r:midpoint(r[1]));ties=[]
    for g,other in candidates:
        c=lex_compare(ids,other,values)
        if c not in (0,1):raise Unresolved('lexicographic prefix or stabilizer')
        if c==0:ties.append(g)
    return mark,ids,tuple(ties)

def frame(u,h):
    E,W=len(h),len(h[0][0]);values=flatten(tuple(b for e in h for b in e));ids=shape(tuple(range(1,len(values)+1)),E,W)
    c=max(conjugate(g,u) for g in GROUP)
    m,ix,J=choose_max([(g,flatten(transform(g,ids))) for g in GROUP if conjugate(g,u)==c],values)
    K=tuple(g for g in GROUP if g in {mul(j,inv(m)) for j in J})
    return c,m,K,shape(tuple(signed_box(i,values) for i in ix),E,W)

def orientation(score,z,b):
    if score.lo>0:return 1
    if score.hi<0:return -1
    if score!=point(0):raise Unresolved('orientation crosses zero')
    # First nonzero triad theorem: later history cannot change fallback.
    for j in range(0,len(z),3):
        v=z[j:j+3]
        if all(x==point(0) for x in v):continue
        ix=act((0,b,0),(1,2,3))
        _,a,_=choose_max([(k,act((k,0,0),ix)) for k in range(3)],v)
        _,n,_=choose_max([(k,act((k,0,1),ix)) for k in range(3)],v)
        c=lex_compare(a,n,v)
        if c not in (-1,1):raise Unresolved('fallback triad not certified nonzero')
        return c
    return 0

def affine_round_bound(row,x,bias,unit,absolute_error=0):
    """Conditional bound for a once-evaluated product/sum tree (FMA allowed).
    Every elementary rounding satisfies |fl(t)-t| <= unit*|t|+absolute_error;
    finite exact inputs, no overflow, no lower precision or extra operations.
    This does not assert that an uninspected BLAS kernel has that contract.
    """
    if len(row)!=len(x):raise ValueError('affine width')
    unit,eta=Q(unit),Q(absolute_error)
    if not 0<=unit<1 or eta<0:raise ValueError('rounding hypothesis')
    M=absolute(box(bias)).hi+sum(absolute(box(a)).hi*absolute(box(b)).hi for a,b in zip(row,x))
    n=2*len(x)+1;growth=(1+unit)**n
    return (growth-1)*M+n*eta*growth


def controlled_interval(h,u,tag,params,errors=None,linear_rounding=None):
    """Exact ideal reference if errors=None; otherwise conditional joint
    enclosure of ideal and native values, given absolute per-stage budgets.
    Each budget bounds the entire named operation relative to its actual
    stored inputs (router includes mean, unary common term and source order).
    Frame/negation/comparison assumed exact on finite floats. No overflow,
    unsupported dtype or kernel admission is inferred from these bounds.
    """
    if linear_rounding is not None and errors is None:raise ValueError('explicit backend hypotheses required')
    mode='ideal' if errors is None else 'conditional_external_bounds'
    stages=STAGES+TYPED_STAGES if params.get('typed_consumer') else STAGES
    errors={k:Q(0) for k in stages} if errors is None else {k:Q(v) for k,v in errors.items()}
    if set(errors)!=set(stages) or any(v<0 for v in errors.values()):raise ValueError('complete nonnegative stage budgets required')
    def stage(k,v):return widen(v,errors[k])
    def affine(A,b,x,k):
        if len(A)!=len(b):raise ValueError('affine dimension')
        return tuple(widen(dot(tuple(map(box,row)),x)+box(t),errors[k]+(affine_round_bound(row,x,t,*linear_rounding) if linear_rounding is not None else 0)) for row,t in zip(A,b))
    h=tuple(tuple(tuple(map(box,b)) for b in e) for e in h);E=len(h);W=len(h[0][0])
    if E<2 or W<3 or W%3 or any(len(e)!=2 or any(len(b)!=W for b in e) for e in h) or u not in GROUP:raise ValueError('invalid controlled carrier')
    c,m,K,local=frame(u,h);x=tuple(map(box,params['embed'][tag,GROUP.index(c)]))
    z=fused_activation(tuple(signed_box(i,x) for i in act((1,0,0),tuple(range(1,W+1)))),tuple(map(box,params['directed'])))
    scores=[]
    for e,(a,b) in enumerate(local):
        mu=tuple(point(Q(1,2))*(p+q) for p,q in zip(a,b))
        scores.append(stage('router',box(params['bias'][e])+dot(tuple(map(box,params['base'])),tuple(map(absolute,mu)))+dot(z,mu)))
    order=tuple(sorted(range(E),key=lambda i:scores[i].lo+scores[i].hi,reverse=True)[:2]);pair=lambda i,j:scores[i]-scores[j]
    if not selection_certificate(order,E,pair)[1]:raise Unresolved('ordered topk')
    weights=tuple(stage('softmax',v) for v in selected_softmax(order,pair));signs=[];regimes=[];raw=list(local);terms=[[],[]]
    for slot,e in enumerate(order):
        A,a,B,b,T,t=params['experts'][e];branches=[];ss=[]
        for j,old in enumerate(local[e]):
            score=stage('orientation',point(2)*dot(tuple(map(box,params['orientation'])),x+old));sign=orientation(score,x+old,j);ss.append(sign)
            X=tuple(point(sign)*v for v in x);H=tuple(point(sign)*v for v in old)
            p=affine(A,a,X+H,'preactivation');g=tuple(stage('gelu',gelu_box(v)) for v in p);y=affine(B,b,g,'expert')
            v=tuple(point(sign)*v for v in affine(T,t,H+X+y,'transition'));v,regime=cap_box(v,params['cap']);regimes.append(regime)
            branches.append(tuple(stage('cap',v0) for v0 in v));terms[j].append(tuple(point(sign)*weights[slot]*v0 for v0 in y))
        raw[e]=tuple(branches);signs.append(tuple(ss))
    output=(tuple(stage('mixture',total(col)) for col in zip(*terms[0])),tuple(stage('mixture',total(col)) for col in zip(*terms[1])))
    def project(v):return fixed_orbit_projection(v,K,errors['projection'])
    def decode(v):
        values=flatten(tuple(b for e in v for b in e));ix=shape(tuple(range(1,len(values)+1)),len(v),W)
        return shape(tuple(signed_box(i,values) for i in flatten(transform(inv(m),ix))),len(v),W)
    lh=project(raw);ly=project((output,));wh=decode(lh);wy=decode(ly)[0]
    wh=tuple(wh[e] if e in order else h[e] for e in range(E));logits=[]
    if params.get('typed_consumer'):
        m0,d0=typed_decode_box(wh,errors['semantic_decode'])
        logits=typed_logits_box(m0,d0,errors['typed_gram'],errors['typed_logits'])
    elif 'readout' in params:
        A,a,tables=params['readout'];p=affine(A,a,flatten(ly[0])+flatten(tuple(b for e in lh for b in e)),'readout');off=0
        for table in tables:
            n=len(table[0]);v=p[off:off+n];off+=n
            proj=tuple(stage('label_projection',point(Q(1,len(K)))*total(v[table[GROUP.index(g)][i]] for g in K)) for i in range(n))
            logits.append(tuple(proj[i] for i in table[GROUP.index(inv(m))]))
        if off!=len(p):raise ValueError('readout shape')
    return dict(mode=mode,backend_bounds_authenticated=False,history=wh,output=wy,logits=tuple(logits),ids=order,weights=weights,signs=tuple(signs),mark=m,stabilizer=K,local_control=c,cap_regimes=tuple(regimes))


def source_parameters(model):
    """Read original coefficients only; native replay has a separate gate."""
    from scripts.x369_source_geometry import SourceGeometryOrderedLearner
    from models.history_orbit_moe import HistoryOrbitMoE
    from scripts.x369_exact_origin_ray import vec,mat
    if type(model) is not SourceGeometryOrderedLearner or model.training:raise ValueError('new eval source required')
    s=model.history.source;W=model.width
    if type(s) is not HistoryOrbitMoE or s.route_geometry!='c2_c3_residual' or s.transition_rank is not None or s.top_k!=2 or s.router_selection!='topk' or getattr(s,'route_probe_action',None) is not None:raise ValueError('unsupported original source')
    if any(e[1].approximate!='none' for e in s.experts):raise ValueError('exact GELU contract required')
    return dict(typed_consumer=model.typed_consumer,embed={(t,i):vec(model.control_embed.weight[t*len(GROUP)+i]) for t in range(4) for i in range(len(GROUP))},base=vec(s.router.weight)[W:],directed=vec(s.c3_residual_weight)[W:],bias=vec(s.expert_bias),orientation=vec(s.orientation.weight),experts=tuple((mat(e[0].weight),vec(e[0].bias),mat(e[2].weight),vec(e[2].bias),mat(t.weight),vec(t.bias)) for e,t in zip(s.experts,s.transitions)),cap=None if s.history_cap is None else Q(float(s.history_cap)),readout=(mat(model.readout.weight),vec(model.readout.bias),tuple(getattr(model,f'label_perm_{i}').detach().cpu().tolist() for i in range(3))))


def contains(v,x):return v.lo<=x<=v.hi


def audit_native_edge(model,h,u,errors,linear_rounding=None):
    """One original native edge, with externally supplied bounds. A matching
    replay is diagnostic; it does not authenticate uniform error budgets.
    h is the original byte key. Refuses uncalibrated implicit zero budgets.
    """
    import torch
    from scripts.x369_source_future_acceptance import native_future_contract
    from scripts.x369_exact_origin_ray import vec
    if errors is None:raise ValueError('explicit external stage bounds required')
    _,step,_=native_future_contract(model,'typed' if model.typed_consumer else 'raw');p=next(model.parameters());dt=p.dtype
    if dt not in (torch.float32,torch.float64):raise ValueError('unsupported dtype')
    E,W=model.experts,model.width;v=torch.frombuffer(bytearray(h),dtype=dt).reshape(E,2,W)
    history=tuple(tuple(vec(b) for b in e) for e in v);g,tag=u
    try:r=controlled_interval(history,GROUP[g],tag,source_parameters(model),errors,linear_rounding)
    except Unresolved as e:return dict(status='unresolved',reason=str(e),backend_bounds_authenticated=False)
    obs,n=step(h,u);obs=dict(obs)
    def observed(name,dtype):return tuple(torch.frombuffer(bytearray(obs[name][2]),dtype=dtype).tolist())
    def check(v,x):return len(v)==len(x) and all(contains(a,Q(float(b))) for a,b in zip(v,x))
    checks=dict(history=check(flatten(tuple(b for e in r['history'] for b in e)),torch.frombuffer(bytearray(n),dtype=dt).tolist()))
    if not model.typed_consumer:
        checks.update(output=check(flatten(r['output']),observed('output',dt)),weights=check(r['weights'],observed('weights',dt)),ids=r['ids']==observed('ids',torch.long),signs=flatten(r['signs'])==observed('local_signs',dt))
    for i,v in enumerate(r['logits']):checks[f'logits_{i}']=check(v,observed(f'logits_{i}',dt))
    return dict(status='observed_inside' if all(checks.values()) else 'observed_outside',checks=checks,cap_regimes=r['cap_regimes'],backend_bounds_authenticated=False,training_steps=0)


def certificate():
    from itertools import product
    import sympy as s
    from scripts.x369_guard_geometry import frame_receipt
    # Exact symbolic sqrt enclosure and numerical high-precision GELU sanity
    # checks (proof of enclosure is the analytic remainder above, not evalf).
    for x in (Q(0),Q(2),Q(1,3),Q(9,4)):
        r=sqrt_box(point(x));assert r.lo**2<=x<=r.hi**2
    for k in range(-20,21):
        x=Q(k,4);r=gelu_box(point(x));t=s.Rational(x);y=t*(1+s.erf(t/s.sqrt(2)))/2
        assert s.Rational(r.lo)<=y.evalf(70)<=s.Rational(r.hi)
    a=Interval(Q(-1,4),Q(1,4));b=Interval(1,2)
    assert lex_compare((1,2),(1,-2),(a,b))==1
    assert lex_compare((1,2),(-1,2),(a,b)) is None
    assert cap_box((Interval(Q(1,2),Q(3,2)),point(0)),1)[1]=='crossing'
    for h in product((-1,0,1),repeat=6):
        H=shape(h,1,3)
        for u in GROUP:
            c,m,K,v=frame(u,shape(tuple(map(point,h)),1,3));ref=frame_receipt(u,H)
            assert (c,m,K)==(ref['control'],ref['mark'],ref['stabilizer']) and tuple(x.lo for x in flatten(tuple(b for e in v for b in e)))==ref['history']
    # Exact source-shaped nonlinear fixture, not fitted or GPU acceptance.
    W=3;E=2;zero=(0,)*W;I=tuple(tuple(int(i==j) for j in range(W)) for i in range(W));Z=((0,)*6,);T=tuple((0,)*6+row for row in I)
    p=dict(embed={(0,i):(1,0,0) for i in range(12)},base=zero,directed=zero,bias=(3,1),orientation=(1,0,0)+zero,experts=((Z,(1,),((1,),(0,),(0,)),zero,T,zero),)*E,cap=Q(1,4),readout=(((0,)*18,), (2,), (tuple((0,) for _ in GROUP),)))
    H=((zero,zero),)*E;u=(0,0,0);r=controlled_interval(H,u,0,p)
    assert r['ids']==(0,1) and r['cap_regimes']==('outside',)*4
    assert all(v==point(0) for v in flatten(tuple(b for e in r['history'] for b in e))+flatten(r['output']))
    assert r['logits']==((point(2),),)
    # A common unary term may erase an ideal strict gap; do not cancel its
    # rounding error. Unknown topk and zero-score boundary are fail closed.
    err={k:Q(0) for k in STAGES};err['router']=2
    try:controlled_interval(H,u,0,p,err)
    except Unresolved as e:assert str(e)=='ordered topk'
    else:raise AssertionError('uncertain topk accepted')
    err['router']=0;err['orientation']=3
    try:controlled_interval(H,u,0,p,err)
    except Unresolved as e:assert str(e)=='orientation crosses zero'
    else:raise AssertionError('uncertain orientation accepted')
    # Nonzero interval box with unique mark and trivial stabilizer.
    H=(((4,2,1),(3,0,-1)),((0,0,0),(0,0,0)));r=controlled_interval(H,u,0,p)
    assert len(r['stabilizer'])==1 and r['ids']==(0,1)
    q=dict(p);q['orientation']=(0,)*6;assert controlled_interval(H,u,0,q)['signs']==((1,1),(1,1))
    # Exact emulated grid rounding, with separate products and FMA.
    checks=0
    rnd=lambda v:Q(round(16*v),16)
    for a,b,c in product(map(Q,(-1,0,1)),repeat=3):
        row=(a+Q(1,3),b-Q(1,5));x=(c+Q(1,7),a-Q(1,9));bias=b+Q(1,13)
        true=sum(i*j for i,j in zip(row,x))+bias;bound=affine_round_bound(row,x,bias,0,Q(1,32))
        sep=rnd(rnd(rnd(row[0]*x[0])+rnd(row[1]*x[1]))+bias);fma=rnd(row[1]*x[1]+rnd(row[0]*x[0]+bias))
        assert abs(sep-true)<=bound and abs(fma-true)<=bound;checks+=2
    nonzero_symbolic_case()
    print(dict(affine_round_checks=checks,nonzero_symbolic=True,frame_checks=729*12,gelu_sanity_checks=41,sqrt_checks=4,cap_crossing=True,projected_zero=True,readout=True,uncertain_guards_rejected=2,nonzero_frame=True,zero_score_fallback=True,training_steps=0))


def nonzero_symbolic_case():
    """Independent literal ideal-real stage oracle, nonzero router/readout,
    nonidentity mark, inactive expert and interval input corners.
    """
    import sympy as s
    from scripts.x369_guard_geometry import frame_receipt
    E,W=3,3;z=(0,)*3;u=(0,0,0);h=(((1,2,-5),(3,-1,0)),((1,-1,2),(0,1,0)),((0,0,1),(0,0,0)))
    f=frame_receipt(u,h);assert f['mark']!=GROUP[0] and len(f['stabilizer'])==1
    A=((1,0,0,0,1,0),);a=(Q(1,4),);B=((1,),(0,),(Q(-1,2),));b=(0,1,0);T=tuple(tuple(Q(int(i==j),2) if j<3 else int(j==6+i) for j in range(9)) for i in range(3));t=(0,0,0)
    x=(2,-1,1);base=(Q(1,8),0,0);directed=(Q(1,16),Q(-1,8),Q(1,32));bias=(5,1,-5);w=(1,0,0,0,0,0)
    # 3-class coordinate permutation toy head, not actual task vocabularies.
    tables=(tuple(tuple(v-1 for v in act((g[0],g[1],0),(1,2,3))) for g in GROUP),)
    L=tuple(tuple(Q((i+j)%5-2,7) for j in range(24)) for i in range(3));lb=(1,0,-1)
    p=dict(embed={(0,i):x for i in range(12)},base=base,directed=directed,bias=bias,orientation=w,experts=((A,a,B,b,T,t),)*E,cap=Q(1),readout=(L,lb,tables))
    # A signed permutation of the original coordinates, then direct dot
    # channels (not fused_activation). Compare exact symbolic expression
    # evaluated at high precision as a regression, not the analytic proof.
    def oracle(h):
        f=frame_receipt(u,h);loc=shape(f['history'],E,W);X=s.Matrix(x);scores=[]
        for i,(v,k) in enumerate(loc):
            M=tuple((s.Rational(j)+s.Rational(l))/2 for j,l in zip(v,k));a0,b0,c0=act((1,0,0),x);A0,B0,C0=M
            d=(a0*A0+b0*B0+c0*C0,a0*B0+b0*C0+c0*A0,a0*C0+b0*A0+c0*B0)
            scores.append(bias[i]+sum(j*abs(k) for j,k in zip(base,M))+sum(j*k for j,k in zip(directed,d)))
        ids=tuple(sorted(range(E),key=lambda i:scores[i],reverse=True)[:2]);den=sum(s.exp(scores[i]) for i in ids);weights=tuple(s.exp(scores[i])/den for i in ids)
        raw=list(loc);y=[s.zeros(3,1),s.zeros(3,1)]
        for e,wgt in zip(ids,weights):
            new=[]
            for j,v in enumerate(loc[e]):
                H=s.Matrix(v);Y=s.Matrix(B)*(s.Matrix(A)*X.col_join(H)+s.Matrix(a)).applyfunc(lambda x:x*(1+s.erf(x/s.sqrt(2)))/2)+s.Matrix(b)
                q=s.Matrix(T)*H.col_join(X).col_join(Y)+s.Matrix(t);norm=s.sqrt(q.dot(q));alpha=min(s.S.One,1/norm) if norm else s.S.One
                new.append(tuple(alpha*q));y[j]+=wgt*Y
            raw[e]=tuple(new)
        assert len(f['stabilizer'])==1
        out=transform(inv(f['mark']),(tuple(tuple(v) for v in y),));wh=shape(flatten(transform(inv(f['mark']),raw)),E,W);wh=tuple(wh[e] if e in ids else h[e] for e in range(E))
        logits=s.Matrix(L)*s.Matrix(flatten(tuple(tuple(v) for v in y))+flatten(tuple(v for e in raw for v in e)))+s.Matrix(lb)
        logits=tuple(logits[i] for i in tables[0][GROUP.index(inv(f['mark']))])
        return ids,wh,out,weights,logits
    r=controlled_interval(h,u,0,p);flat=flatten(tuple(b for e in h for b in e));H=shape(tuple(Interval(Q(v)-Q(1,64),Q(v)+Q(1,64)) for v in flat),E,W);R=controlled_interval(H,u,0,p)
    for offset in (0,Q(1,64),Q(-1,64)):
        h0=shape(tuple(Q(v)+offset for v in flat),E,W);ids,wh,out,wg,lg=oracle(h0);q=r if offset==0 else R
        assert q['ids']==ids and len(q['stabilizer'])==1
        for intervals,values in ((flatten(tuple(b for e in q['history'] for b in e)),flatten(tuple(b for e in wh for b in e))),(flatten(q['output']),flatten(out)),(q['weights'],wg),(q['logits'][0],lg)):
            assert len(intervals)==len(values) and all(s.Rational(I.lo)<=s.sympify(v).evalf(70)<=s.Rational(I.hi) for I,v in zip(intervals,values))
    assert r['history'][2]==tuple(tuple(map(point,b)) for b in h[2])


if __name__=='__main__':certificate()
