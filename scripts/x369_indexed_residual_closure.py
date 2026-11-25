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


if __name__=='__main__':self_test()
