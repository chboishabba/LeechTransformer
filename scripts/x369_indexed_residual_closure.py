"""Reversible semantic/residual transport and finite source closure.
Original Leech-LILA: Anatolii Kornienko; DASHI extension.
Inspired by Hopf369PhaseRetainingObserverCubeExact, not a physics identification.
Pure exact-value algebra; native bit-state exploration is a separate contract.
"""
from collections import deque
from fractions import Fraction as Q
from itertools import product


def add(a,b):
    if len(a)!=len(b): raise ValueError('dimension mismatch')
    return tuple(x+y for x,y in zip(a,b))


def sub(a,b):
    return add(a,tuple(-x for x in b))


def split(h,q,section):
    z=q(h);r=sub(h,section(z))
    if q(section(z))!=z or any(q(r)):
        raise ValueError('decoder/section must be a linear split on this state')
    return z,r


def join(packet,section):
    z,r=packet
    return add(section(z),r)


def transport(h,q0,q1,S0,S1,T,A):
    """Time-indexed exact signed defect, no state identification across time."""
    z,r=split(h,q0,S0);n=T(h);w,s=split(n,q1,S1)
    defect=sub(w,A(z))
    assert join((add(A(z),defect),s),S1)==n
    return dict(before=(z,r),after=(w,s),defect=defect)


def two_step(h,q0,q1,q2,S0,S1,S2,T,U,A,B):
    p=transport(h,q0,q1,S0,S1,T,A)
    t=transport(T(h),q1,q2,S1,S2,U,B)
    defect=sub(q2(U(T(h))),B(A(q0(h))))
    # B must be linear for this cocycle identity.
    assert defect==add(B(p['defect']),t['defect'])
    return defect


def explore(seeds,controls,step,action,max_states=10000):
    """Forward graph on ORIGINAL hashable states, not task labels.
    Each seed: (state, label, original seed ID). Each edge uses actual step.
    action is the independent task target, never supplied to step.
    An exhausted queue proves finite closure only for this deterministic
    state/step/control contract. Limit is explicit, not a quotient proof.
    """
    controls=tuple(controls)
    if not controls or len(set(controls))!=len(controls) or max_states<1:
        raise ValueError('invalid control alphabet or budget')
    table={};parents={};queue=deque();edges=0
    def path(h):
        word=[]
        while parents[h][0] is not None:
            h,u=parents[h];word.append(u)
        return dict(seed=parents[h][1],word=list(reversed(word)))
    def collision(h,z,second):
        return dict(status='collision',states=len(table),edges=edges,
                    first=path(h),second=second,labels=(table[h],z))
    for h,z,i in seeds:
        z=tuple(z)
        if h in table:
            if table[h]!=z:return collision(h,z,dict(seed=i,word=[]))
            continue
        if len(table)>=max_states:return dict(status='budget',states=len(table),edges=edges,table=None)
        table[h]=z;parents[h]=(None,i);queue.append(h)
    if not table:raise ValueError('empty seed carrier')
    while queue:
        h=queue.popleft()
        for u in controls:
            n=step(h,u);z=tuple(action(u,table[h]));edges+=1
            if n in table:
                if table[n]!=z:
                    p=path(h);p['word'].append(u)
                    return collision(n,z,p)
            else:
                if len(table)>=max_states:
                    return dict(status='budget',states=len(table),edges=edges,table=None)
                table[n]=z;parents[n]=(h,u);queue.append(n)
    return dict(status='closed',states=len(table),edges=edges,table=table)


def native_source(model):
    """Actual new source, fixed eval model and single-row gfx803 execution.
    Returns byte-exact state keys, seed and repeated step. No CPU fallback,
    no historical checkpoint migration, no semantic target in the source.
    """
    import torch
    from scripts.x369_source_geometry import SourceGeometryOrderedLearner,signed_seed
    from models.geometric_reflection_fibre import GROUP
    if type(model) is not SourceGeometryOrderedLearner or model.training:
        raise ValueError('new eval architecture required')
    p=next(model.parameters());dev=p.device;dt=p.dtype
    if dev.type!='cuda' or not torch.version.hip or not getattr(torch.cuda.get_device_properties(dev),'gcnArchName','').startswith('gfx803'):
        raise RuntimeError('native gfx803 required')
    shape=(1,model.experts,2,model.width)
    def key(h):
        if tuple(h.shape)!=shape or h.dtype!=dt or not torch.isfinite(h).all():raise ValueError('invalid history')
        return h.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    def seed(m,d):
        with torch.no_grad():
            v=torch.tensor([tuple(m)+tuple(d)+(0,0,0)],device=dev,dtype=dt)
            return key(signed_seed(v,model.experts,model.width))
    def once(h,u):
        g,tag=u
        if not 0<=g<len(GROUP) or not 0<=tag<4:raise ValueError('invalid control')
        with torch.no_grad():
            x=torch.frombuffer(bytearray(h),dtype=dt).reshape(shape).to(dev)
            c=torch.tensor([g],device=dev,dtype=torch.long)
            return key(model.history.step_controlled(c,x,lambda z:model.control_embed(tag*len(GROUP)+z),train_sign=False).history)
    def step(h,u):
        a=once(h,u);b=once(h,u)
        if a!=b:raise RuntimeError('native transition not repeatable on this edge')
        return a
    def action(u,z):
        # Independent signed task action in the original left-roll convention.
        k,b,a=GROUP[u[0]];m,d=z[:3],z[3:]
        def transform(v):
            w=v[::-1] if b else v
            return tuple((-1)**a*w[(i+k)%3] for i in range(3))
        return transform(m)+tuple((-1)**b*x for x in transform(d))
    return seed,step,action


def self_test():
    # Exact-value fixtures, not fitted native histories.
    q=lambda h:h[:2];S=lambda z:z+(Q(0),)
    T=lambda h:(h[1]+h[2],h[0],h[2]-1)
    U=lambda h:(-h[0],h[1]+2*h[2],h[2]+1)
    A=lambda z:(z[1],z[0]);B=lambda z:(-z[0],z[1])
    n=0
    for h in product(map(Q,(-1,0,1)),repeat=3):
        assert join(split(h,q,S),S)==h
        two_step(h,q,q,q,S,S,S,T,U,A,B);n+=1
    # All 729 pairs, all 12 signed S3 actions on a literal fixture carrier.
    from itertools import product as P
    group=tuple(P(range(3),range(2),range(2)))
    def action(g,z):
        k,b,a=g
        def f(v):
            w=v[::-1] if b else v
            return tuple((-1)**a*w[(i+k)%3] for i in range(3))
        return f(z[:3])+tuple((-1)**b*x for x in f(z[3:]))
    seeds=[(z,z,i) for i,z in enumerate(P((-1,0,1),repeat=6))]
    good=explore(seeds,group,lambda h,u:action(u,h),action)
    assert good['status']=='closed' and good['states']==729 and good['edges']==8748
    bad=explore([(0,(1,),0)],(0,),lambda h,u:h,lambda u,z:(-z[0],))
    assert bad['status']=='collision' and bad['first']['word']==[] and bad['second']['word']==[0]
    budget=explore([(0,(0,),0)],(0,),lambda h,u:h+1,lambda u,z:z,max_states=3)
    assert budget['status']=='budget' and budget['states']==3
    print(dict(residual_cases=n,fixture_states=good['states'],fixture_edges=good['edges'],collision='PASS',budget='PASS'))


def semantic_adjoint(z,like):
    """Adjoint of the existing constructed history decoder (not signed_seed)."""
    import torch
    if like.ndim!=4 or like.shape[2]!=2 or like.shape[-1]<9 or z.shape!=(len(like),6):
        raise ValueError('paired source shape and six semantic coordinates required')
    if z.dtype!=like.dtype or z.device!=like.device:raise ValueError('matching dtype/device required')
    m,d=z[:,:3],z[:,3:];s=torch.zeros_like(like)
    s[:,0,0,:3]=(m+d)/2;s[:,0,1,:3]=(m-d)/2
    s[:,0,:,3:6]=m[:,None,:]/2
    s[:,0,0,6:9]=d/2;s[:,0,1,6:9]=-d/2
    return s


def semantic_repair(raw,target,ids,*,require_value_exact=False):
    """Opt-in next-history correction, preserves raw receipt and complement.

    q is the existing constructed decoder, S=q^T. This is NOT unchanged
    source transition equivalence, and must not count as learned action.
    Cap/admissibility are the caller's explicit responsibility.
    """
    import torch
    from scripts.x369_constructive_finite_source import decode_history
    if raw.ndim!=4 or raw.shape[2]!=2 or raw.shape[-1]<9 or target.shape!=(len(raw),6):
        raise ValueError('invalid repair carrier')
    if raw.dtype!=target.dtype or raw.device!=target.device or not raw.is_floating_point():
        raise ValueError('matching real history/target required')
    if ids.ndim!=2 or ids.shape[0]!=len(raw) or ids.dtype!=torch.long or ids.device!=raw.device or not ((ids>=0)&(ids<raw.shape[1])).all():
        raise ValueError('valid source expert ids required')
    if not torch.isfinite(raw).all() or not torch.isfinite(target).all():raise ValueError('finite repair carrier')
    z=torch.cat(decode_history(raw),-1);delta=target-z
    need=(delta!=0).any(-1);selected=(ids==0).any(-1)
    if (need&~selected).any():raise RuntimeError('semantic defect cannot be repaired on active experts')
    correction=semantic_adjoint(delta,raw)
    # Preserve inactive experts exactly. Even at zero defect, the selected
    # branch uses the projected derivative, not the uncorrected derivative.
    h=raw.clone();h[:,0]=torch.where(selected[:,None,None],raw[:,0]+correction[:,0],raw[:,0])
    after=torch.cat(decode_history(h),-1);err=after-target
    if not torch.isfinite(h).all() or not torch.isfinite(err).all():raise RuntimeError('nonfinite correction')
    if require_value_exact and (err!=0).any():raise RuntimeError('backend exact value repair failed')
    return dict(history=h,raw_history=raw,correction=correction,raw_semantic_defect=-delta,
                after_defect=err,byte_reopen_via_retained_raw=True,nonzero_correction=bool(need.any()))


def corrected_source_step(model,history,control,tags,*,require_value_exact=False):
    """Known action plus unknown residual, explicit modified recurrence.

    Raw original output/route receipt remains available; next repaired history
    differs from source when defect is nonzero. No typed learned-accuracy claim.
    """
    import torch
    from scripts.x369_constructive_finite_source import decode_history
    from models.geometric_reflection_fibre import GROUP
    from models.geometric_orbit_descent import apply_group_indices
    if model.history.source.history_cap is not None:raise ValueError('repair requires uncapped source; cap feasibility separate')
    if tags.shape!=control.shape or tags.dtype!=torch.long or not ((tags>=0)&(tags<4)).all():raise ValueError('literal tag carrier')
    m,d=decode_history(history)
    target=torch.cat((apply_group_indices(m,control),apply_group_indices(d,control)*torch.where((control//2)%2==1,-1,1)[:,None]),-1)
    out=model.history.step_controlled(control,history,lambda c:model.control_embed(tags*len(GROUP)+c),train_sign=model.training)
    receipt=semantic_repair(out.history,target,out.ids,require_value_exact=require_value_exact)
    receipt['raw_step']=out;receipt['target']=target
    return receipt


def semantic_repair_certificate():
    """Exact matrix identity and local minimality for E=4, width=12."""
    import sympy as S
    from itertools import product
    E,W=4,12;N=E*2*W;Q=S.zeros(6,N)
    for j in range(3):
        for b in range(2):
            Q[j,b*W+j]=Q[j,b*W+3+j]=S.Rational(1,2)
            Q[3+j,b*W+j]=Q[3+j,b*W+6+j]=S.Rational((-1)**b,2)
    P=Q.T*Q
    assert Q*Q.T==S.eye(6) and P.T==P and P*P==P
    assert P.rank()==6 and (S.eye(N)-P).rank()==90
    for k,b,a in product(range(3),range(2),range(2)):
        G=S.zeros(N);A=S.zeros(6)
        for e,j,l in product(range(E),range(2),range(W)):
            dst=3*(l//3)+((2-l%3 if b else l%3)-k)%3
            G[e*2*W+(j^b)*W+dst,e*2*W+j*W+l]=(-1)**a
        for lane,j in product(range(2),range(3)):
            dst=((2-j if b else j)-k)%3
            A[3*lane+dst,3*lane+j]=(-1)**(a+b*lane)
        assert Q*G==A*Q and G*P==P*G and G*Q.T==Q.T*A
    # All exact corrections are Q^T delta + ker Q; the two parts
    # are orthogonal, so the former is uniquely minimum norm.
    assert Q*(S.eye(N)-P)==S.zeros(6,N) and Q*(S.eye(N)-P)*Q.T==S.zeros(6)
    return dict(semantic_rank=6,residual_rank=90,group_intertwiners=12,minimum_norm_unique=True,
                inactive_expert0_nonzero_defect_impossible=True)


def semantic_repair_numeric_certificate():
    """Dyadic deterministic source-shaped fixture; no optimizer or fit."""
    import torch
    from scripts.x369_constructive_finite_source import decode_history
    from itertools import product
    result={}
    for dt in (torch.float64,torch.float32):
        h=(torch.arange(729*96).reshape(729,4,2,12)%17-8).to(dt)/4
        h.requires_grad_()
        z=torch.tensor(list(product((-1,0,1),repeat=6)),dtype=dt)
        ids=torch.tensor([[0,1]]*len(z))
        q=lambda x:torch.cat(decode_history(x),-1)
        p=lambda x:x-semantic_adjoint(q(x),x)
        r=semantic_repair(h,z,ids,require_value_exact=True)
        assert torch.equal(q(r['history']),z) and torch.equal(p(h),p(r['history']))
        assert torch.equal(r['history'][:,1:],h[:,1:])
        assert torch.equal(r['history']-r['correction'],h)
        g=torch.autograd.grad(q(r['history']).sum(),h,retain_graph=True)[0]
        assert g.abs().max()==0
        grad=torch.autograd.grad((p(r['history'])*p(h.detach())).sum(),h)[0]
        assert torch.equal(grad,p(h.detach()))
        a=semantic_repair(h,q(h.detach()),ids,require_value_exact=True)
        assert torch.autograd.grad(q(a['history']).sum(),h)[0].abs().max()==0
        try:
            semantic_repair(h,z,torch.tensor([[1,2]]*len(z)),require_value_exact=True)
        except RuntimeError as e:
            if 'active experts' not in str(e):raise
        else:raise AssertionError('inactive correction not rejected')
        result[str(dt)]=dict(rows=729,exact_value=True,complement=True,gradient=True,inactive_guard=True)
    return result


if __name__=='__main__':
    self_test()
    print(semantic_repair_certificate())
    print(semantic_repair_numeric_certificate())
