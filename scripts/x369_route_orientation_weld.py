"""Exact ideal-real guard reductions, not a native TopK tie contract.
Original Leech-LILA: Anatolii Kornienko; DASHI extension.
"""
from fractions import Fraction as F
from itertools import product, permutations
from models.geometric_reflection_fibre import GROUP,act,mul,inv
from scripts.x369_guard_geometry import frame_receipt,shape,transform,flatten,conjugate
from scripts.x369_capped_semantic_geometry import cap_profile,exact_repair
from scripts.x369_r4_interval_certificate import point,fused_activation


def sign(x):return int(x>0)-int(x<0)


def lex_sign(a,b):return int(a>b)-int(a<b)


def full_chart(z,b):
    z=act((0,b,0),tuple(z))
    return lex_sign(max(act((k,0,0),z) for k in range(3)),max(act((k,0,1),z) for k in range(3)))


def first_chart(z,b):
    if not z or len(z)%3 or b not in (0,1):raise ValueError('complete triads required')
    for i in range(0,len(z),3):
        t=tuple(z[i:i+3])
        if any(t):return full_chart(t,b)
    return 0


def orientation(x,h,w,b):
    z=tuple(x)+tuple(h)
    if len(x)!=len(h) or len(w)!=len(z) or not x or len(x)%3:raise ValueError('invalid branch')
    score=2*sum(a*v for a,v in zip(w,z))
    return sign(score) if score else first_chart(z,b)


def route_scores(x,history,base,directed,bias):
    """R4 logits modulo a shared activation-only scalar, ideal-real exact.
    Reuse existing fused directed-score owner; no native rounding claim.
    """
    if not x or len(x)%3 or len(base)!=len(x) or len(directed)!=len(x) or len(bias)!=len(history):raise ValueError('invalid router')
    z=[v.lo for v in fused_activation(list(map(point,act((1,0,0),x))),list(map(point,directed)))];out=[]
    for e,(a,b) in enumerate(history):
        if len(a)!=len(x) or len(b)!=len(x):raise ValueError('branch width')
        m=[(F(u)+F(v))/2 for u,v in zip(a,b)]
        out.append(bias[e]+sum(p*abs(v)+q*v for p,q,v in zip(base,z,m)))
    return tuple(out)


def route_receipt(logits,order):
    """Validate weak ideal-real order, certify unique order only if strict.
    Actual native ties require original source receipt, never invent a tie-break.
    """
    order=tuple(order);n=len(logits)
    if not order or len(set(order))!=len(order) or any(i not in range(n) for i in order):raise ValueError('invalid selected order')
    outside=sorted(set(range(n))-set(order))
    pairs=tuple(zip(order,order[1:]))+tuple((order[-1],j) for j in outside)
    gaps=tuple(logits[i]-logits[j] for i,j in pairs)
    return dict(weak=all(g>=0 for g in gaps),strict=all(g>0 for g in gaps),pairs=pairs,gaps=gaps,
                relative=tuple(logits[i]-logits[order[0]] for i in order[1:]))


def same_guard(a,b):
    p=frame_receipt((0,0,0),a);q=frame_receipt((0,0,0),b)
    assert p['maximizers']==q['maximizers'] and p['mark']==q['mark'] and p['stabilizer']==q['stabilizer']
    assert cap_profile(a)==cap_profile(b)
    assert exact_repair(a,(0,)*6,range(4),1)['defect']==exact_repair(b,(0,)*6,range(4),1)['defect']
    assert all(sum(v*v for v in row)<=1 for h in (a,b) for e in h for row in e)
    return shape(p['history'],4,12),shape(q['history'],4,12)


def witness_certificate():
    W=12;zero=(0,)*W;x=(1,)*W
    def e(*entries):return tuple(dict(entries).get(i,0) for i in range(W))
    def state(p,anchor=zero):return ((anchor,anchor),(p,p),(zero,zero),(zero,zero))
    # Identical profile/J/order/orientation but different selected logit gap.
    a,b=same_guard(state(e((0,1))),state(e((0,F(3,5)),(9,F(4,5)))))
    p,q=[route_receipt(route_scores(x,h,x,zero,(3,1,0,-1)),(0,1)) for h in (a,b)]
    assert p['strict'] and q['strict'] and p['relative']==(-1,) and q['relative']==(F(-3,5),)
    w=x+zero
    assert all(orientation(x,h[i][j],w,j)==1 for h in (a,b) for i,j in product((0,1),range(2)))
    # Same profile/J/route/relative gap, opposite nonzero learned score.
    a,b=same_guard(state(e((0,F(3,5)),(9,F(4,5)))),state(e((0,F(3,5)),(9,F(-4,5)))))
    assert route_scores(x,a,x,zero,(3,1,0,-1))==route_scores(x,b,x,zero,(3,1,0,-1))
    w=zero+e((9,1));assert [orientation(x,h[1][0],w,0) for h in (a,b)]==[1,-1]
    # Zero embedding and zero learned scores: anchor frame on expert0.
    a,b=same_guard(state(e((0,1)),e((0,1))),state(e((0,-1)),e((0,1))))
    assert route_scores(zero,a,x,zero,(3,1,0,-1))==route_scores(zero,b,x,zero,(3,1,0,-1))
    assert [orientation(zero,h[1][0],zero+zero,0) for h in (a,b)]==[1,-1]
    return dict(weight_separator=True,learned_orientation_separator=True,zero_embedding_fallback_separator=True)


def certificate():
    triads=tuple(product((-1,0,1),repeat=3));n=0
    for x,h in product(triads,repeat=2):
        for b in range(2):
            assert first_chart(x+h,b)==full_chart(x+h,b)
            if any(x):assert first_chart(x+h,b)==full_chart(x,b)
            n+=1
    directed_checks=0
    for x,h in product(triads,repeat=2):
        a,b,c=act((1,0,0),x);A,B,C=h;w=(2,-3,5);base=(1,-2,3)
        want=sum(k*abs(v) for k,v in zip(base,h))+2*(a*A+b*B+c*C)-3*(a*B+b*C+c*A)+5*(a*C+b*A+c*B)
        assert route_scores(x,((h,h),),base,w,(0,))==(want,)
        directed_checks+=1
    # Three adjacent/boundary comparisons suffice for ordered top2 E4.
    routes=0;ties=0
    for logits in product(range(-1,2),repeat=4):
        for order in permutations(range(4),2):
            r=route_receipt(logits,order);i,j=order;outside=set(range(4))-set(order)
            assert r['weak']==(logits[i]>=logits[j] and all(logits[k]>=logits[l] for k in order for l in outside))
            assert r['strict']==(logits[i]>logits[j] and all(logits[k]>logits[l] for k in order for l in outside))
            assert len(r['pairs'])==3 and route_receipt(tuple(v+17 for v in logits),order)==r
            routes+=1;ties+=int(r['weak'] and not r['strict'])
    # Existing controlled frame: related inputs have equal local pair and K.
    # Mark covariance modulo local K, not raw mark equality.
    orbit=0
    for t,u,g in product(triads,GROUP,GROUP):
        h=((t,(0,0,0)),);p=frame_receipt(u,h)
        hg=shape(flatten(transform(g,h)),1,3);q=frame_receipt(conjugate(g,u),hg)
        assert (p['history'],p['control'],p['stabilizer'])==(q['history'],q['control'],q['stabilizer'])
        assert mul(mul(q['mark'],g),inv(p['mark'])) in p['stabilizer']
        orbit+=1
    return dict(fallback_checks=n,directed_score_checks=directed_checks,route_order_checks=routes,weak_tie_cases=ties,frame_covariance_checks=orbit,**witness_certificate())


if __name__=='__main__':print(certificate())
