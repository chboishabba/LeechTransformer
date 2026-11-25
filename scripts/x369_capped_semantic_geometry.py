"""Ideal-real cap geometry for the existing X369 six-row decoder.
Original Leech-LILA: Anatolii Kornienko; DASHI extension.
No fitted, native floating-point or byte-preservation claim.
"""
from fractions import Fraction as F
from itertools import product


def exact_repair(history, target, selected, cap):
    """Rational value audit, shape E x 2 x W, W>=9.

    Cap applies only to selected expert branches, as in PairedHistoryMoE.
    Return the unique complement-preserving candidate and exact feasibility;
    never clip or silently modify an inactive branch. Retain original history.
    Inputs must be exact rationals (floats rejected).
    """
    def rational(x):
        if type(x) not in (int,F):
            raise ValueError('exact rational input required')
        return F(x)
    h=tuple(tuple(tuple(map(rational,b)) for b in e) for e in history)
    z=tuple(map(rational,target)); c=rational(cap)
    if not h or len(z)!=6 or c<=0 or len(h[0])!=2:
        raise ValueError('invalid carrier or cap')
    w=len(h[0][0]); ids=tuple(selected)
    if w<9 or any(len(e)!=2 or any(len(b)!=w for b in e) for e in h):
        raise ValueError('invalid paired carrier')
    if len(set(ids))!=len(ids) or any(type(i)!=int or not 0<=i<len(h) for i in ids):
        raise ValueError('invalid selected experts')
    a,b=h[0]
    m=tuple((a[j]+b[j]+a[j+3]+b[j+3])/2 for j in range(3))
    d=tuple((a[j]-b[j]+a[j+6]-b[j+6])/2 for j in range(3))
    defect=tuple(x-y for x,y in zip(m+d,z))
    if 0 not in ids and any(defect):
        return dict(feasible=False, reason='inactive semantic expert', raw=h, defect=defect)
    dm,dd=defect[:3],defect[3:]
    repaired=tuple(tuple(v-t for v,t in zip(h[0][i],
        tuple((x+(-1)**i*y)/2 for x,y in zip(dm,dd))+
        tuple(x/2 for x in dm)+tuple((-1)**i*y/2 for y in dd)+(F(0),)*(w-9)))
        for i in range(2))
    out=(repaired,)+h[1:] if 0 in ids else h
    norms=tuple((i,j,sum(v*v for v in out[i][j])) for i in ids for j in range(2))
    feasible=all(n<=c*c for _,_,n in norms)
    return dict(feasible=feasible, reason='ok' if feasible else 'cap', raw=h,
                candidate=out, defect=defect, selected_norm_squares=norms)


def certificate():
    import sympy as s
    I=s.eye(6); Q=[]
    for b in range(2):
        A=s.zeros(6,12)
        for j in range(3):
            A[j,j]=A[j,j+3]=s.Rational(1,2)
            A[j+3,j]=A[j+3,j+6]=s.Rational((-1)**b,2)
        Q.append(A)
    E=s.zeros(6)
    for j in range(3): E[j,j+3]=E[j+3,j]=1
    P=(I+E)/2; N=(I-E)/2
    roots=[(s.sqrt(3)*P+N)/2,(P+s.sqrt(3)*N)/2]
    inverses=[2*P/s.sqrt(3)+2*N,2*P+2*N/s.sqrt(3)]
    k=(s.sqrt(3)+1)/2
    B=[A.T*V/k for A,V in zip(Q,inverses)]
    zero=lambda A:A.applyfunc(s.simplify)==s.zeros(*A.shape)
    assert Q[0]*Q[0].T+Q[1]*Q[1].T==I
    for A,L,V,D in zip(Q,roots,inverses,B):
        assert zero(A*A.T-L*L) and zero(L*V-I)
        assert zero(D.T*D-I/k**2)
    assert zero(Q[0]*B[0]+Q[1]*B[1]-I)
    # Sharpness: unit direction of m=d, all coordinates +/-1.
    z=s.ones(6,1); v=z/s.sqrt(6)
    assert zero(P*v-v) and zero(N*v)
    assert s.simplify(sum((A.T*v).norm() for A in Q)-k)==0
    assert s.simplify((v.T*z)[0]/k-s.sqrt(6)*(s.sqrt(3)-1))==0
    fixed_max=F(0); cap_one=0; count=0
    h=tuple(((0,)*12,(0,)*12) for _ in range(4))
    for z in product((-1,0,1),repeat=6):
        r=exact_repair(h,z,(0,1),1); count+=1; cap_one+=r['feasible']
        a,b=z[:3],z[3:]; n=sum(x*x for x in z); dot=sum(x*y for x,y in zip(a,b))
        assert r['selected_norm_squares'][:2]==((0,0,F(n+dot,2)),(0,1,F(n-dot,2)))
        fixed_max=max(fixed_max,F(n+abs(dot),2))
        x,y=r['candidate'][0]
        assert tuple((x[j]+y[j]+x[j+3]+y[j+3])/2 for j in range(3)) + tuple((x[j]-y[j]+x[j+6]-y[j+6])/2 for j in range(3))==z
        assert r['candidate'][1:]==h[1:]
    assert fixed_max==F(9,2)
    assert not exact_repair(h,(1,0,0,0,0,0),(1,2),3)['feasible']
    h2=(((0,)*9+(2,0,0),(0,)*12),)+h[1:]
    assert not exact_repair(h2,(0,)*6,(0,1),1)['feasible']
    assert exact_repair(h2,(0,)*6,(1,2),1)['feasible']  # inactive is not recapped
    return dict(ternary_targets=count,zero_residual_cap_one=cap_one,
        zero_residual_uniform_cap_squared=str(fixed_max),
        flexible_uniform_cap_squared=str(s.simplify(6/k**2)),
        exact_balanced_section=True, inactive_guard=True, residual_cap_guard=True,
        training_steps=0, native_gpu=False)


if __name__=='__main__':
    print(certificate())
