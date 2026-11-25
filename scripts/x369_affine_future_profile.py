"""Exact affine pullback specialization of X369 future cap consumers.
Original Leech-LILA: Anatolii Kornienko; DASHI extension.
Not a fitted nonlinear/guarded or native floating-point certificate.
"""
import sympy as s


def rational_matrix(A):
    A=s.Matrix(A)
    if any(x.is_Rational is not True for x in A):
        raise ValueError('exact rational matrices required')
    return A


def affine_lift(A,b):
    A,b=rational_matrix(A),rational_matrix(b)
    if not A.rows or A.rows!=A.cols or b.shape!=(A.rows,1):
        raise ValueError('invalid affine map')
    return A.row_join(b).col_join(s.zeros(1,A.cols).row_join(s.ones(1,1)))


def linear_form(row, bias=0):
    row=rational_matrix([list(row)]); n=row.cols
    if s.sympify(bias).is_Rational is not True:raise ValueError('exact bias required')
    H=s.zeros(n+1);H[:n,n]=row.T/2;H[n,:n]=row/2;H[n,n]=bias
    return H


def energy_form(A,b):
    A,b=rational_matrix(A),rational_matrix(b)
    if b.shape!=(A.rows,1):raise ValueError('invalid energy map')
    U=A.row_join(b);return U.T*U


def compile_future(maps, forms):
    """Exact finite span of all affine pullbacks, with literal word witnesses.

    maps: ordered (label, homogeneous matrix) pairs; forms: symmetric matrices.
    Full declared affine carrier, all words legal, no cap/guard switch inferred.
    Equality of basis evaluations iff equality of every future observation.
    Minimal linear span, not a lower bound on arbitrary nonlinear codec size.
    """
    maps=tuple((u,rational_matrix(T)) for u,T in maps)
    forms=tuple(map(rational_matrix,forms))
    if not forms or not maps or len({u for u,_ in maps})!=len(maps):raise ValueError('nonempty indexed maps/forms required')
    d=forms[0].rows
    if d<2 or any(H.shape!=(d,d) or H!=H.T for H in forms):raise ValueError('symmetric forms required')
    if any(T.shape!=(d,d) or list(T[-1,:])!=[0]*(d-1)+[1] for _,T in maps):raise ValueError('affine lift required')
    ij=tuple((i,j) for i in range(d) for j in range(i,d)); pivots={};basis=[];witness=[]
    def admit(H,origin,word):
        v=[H[i,j] for i,j in ij]
        for p,r in sorted(pivots.items()):
            if v[p]:
                a=v[p];v=[x-a*y for x,y in zip(v,r)]
        p=next((i for i,x in enumerate(v) if x),None)
        if p is None:return
        a=v[p];pivots[p]=[x/a for x in v];basis.append(H);witness.append((origin,word))
    for i,H in enumerate(forms):admit(H,i,())
    i=0
    while i<len(basis):
        origin,word=witness[i]
        for u,T in maps:admit(T.T*basis[i]*T,origin,(u,)+word)
        i+=1
    # Recover exact coordinate matrices in the original independent basis.
    columns=s.Matrix([[H[i,j] for H in basis] for i,j in ij]) if basis else s.zeros(len(ij),0)
    def coordinates(H):
        v=s.Matrix([H[i,j] for i,j in ij]); c=columns.gauss_jordan_solve(v)[0]
        assert columns*c==v
        return c.T
    decoder=s.Matrix.vstack(*(coordinates(H) for H in forms))
    transitions={u:s.Matrix.vstack(*(coordinates(T.T*H*T) for H in basis)) if basis else s.zeros(0) for u,T in maps}
    return dict(basis=tuple(basis),witness=tuple(witness),decoder=decoder,transitions=transitions,
                rank=len(basis),ambient_dimension=len(ij))


def encode(c,x):
    X=s.Matrix(list(x)+[1]);return s.Matrix(len(c['basis']),1,[(X.T*H*X)[0] for H in c['basis']])


def x369_profile_forms(experts=4,width=12):
    """Derive 6 semantic + 14 cap forms from the existing exact decoder.

    Ambient order is flattened [expert,branch,coordinate]. No guessed
    learned decoder, fitted transition or native rounding assertion.
    """
    from scripts.x369_capped_semantic_geometry import exact_repair
    if type(experts)!=int or type(width)!=int or experts<1 or width<9:
        raise ValueError('invalid source dimensions')
    n=2*experts*width
    def shaped(x):
        return tuple(tuple(tuple(x[(2*e+b)*width:(2*e+b+1)*width]) for b in range(2)) for e in range(experts))
    columns=[]
    for j in range(n):
        h=shaped([int(i==j) for i in range(n)])
        columns.append(exact_repair(h,(0,)*6,range(experts),2)['defect'])
    Q=s.Matrix(columns).T
    assert Q*Q.T==s.eye(6) and Q[:,2*width:]==s.zeros(6,n-2*width)
    R=s.eye(n)-Q.T*Q
    V=Q[:,:width]*R[:width,:]
    rows=Q.col_join(V)
    return tuple(linear_form(rows[i,:]) for i in range(12))+tuple(
        energy_form(R[i:i+width,:],s.zeros(width,1)) for i in range(0,n,width))


def source_forms_certificate():
    from fractions import Fraction as F
    from scripts.x369_capped_semantic_geometry import exact_repair
    forms=x369_profile_forms();assert len(forms)==20
    for seed in range(2):
        h=tuple(tuple(tuple(F(((seed+2)*(i+3)+7*b+11*e)%19-9,4) for i in range(12)) for b in range(2)) for e in range(4))
        raw=exact_repair(h,(0,)*6,range(4),20);r=raw['candidate'];a=r[0][0]
        v=tuple((a[j]+a[j+3])/2 for j in range(3))+tuple((a[j]+a[j+6])/2 for j in range(3))
        expected=raw['defect']+v+tuple(sum(x*x for x in b) for e in r for b in e)
        assert list(encode(dict(basis=forms),[x for e in h for b in e for x in b]))==list(expected)
    return dict(source_forms=len(forms),source_shape_checks=2,ambient_dimension=97*98//2)


def certificate():
    # Source-legal fixed route, cap1, unit ball, zero FFN and homogeneous
    # transition on expert1's equal branches; other experts zero. The four
    # triad scalars commute with signed spatial action; no cap activates.
    scales=[s.Rational(1,2**i) for i in range(1,5)]
    A=s.diag(*(v for v in scales for _ in range(3)));T=affine_lift(A,s.zeros(12,1))
    H=energy_form(s.eye(12),s.zeros(12,1));c=compile_future([('tick',T)],[H])
    assert c['rank']==4 and [len(w) for _,w in c['witness']]==[0,1,2,3]
    V=s.Matrix([[v**(2*t) for v in scales] for t in range(4)])
    assert V.det()!=0
    # 3 samples cannot separate all triad energy profiles; construct two
    # nonnegative unit-ball profiles with same first three and different fourth.
    d=V[:3,:].nullspace()[0];d=d/(2*sum(abs(v) for v in d))
    p=s.Matrix([max(v,0) for v in d]);q=s.Matrix([max(-v,0) for v in d])
    assert sum(p)<=1 and sum(q)<=1 and V[:3,:]*p==V[:3,:]*q and V[3,:]*p!=V[3,:]*q
    def state(e):return s.Matrix([v for a in e for v in (s.sqrt(a),0,0)])
    x,y=state(p),state(q);cx,cy=encode(c,x),encode(c,y)
    assert (cx-cy).applyfunc(s.simplify)==V*(p-q)
    assert (encode(c,A*x)-c['transitions']['tick']*cx).applyfunc(s.simplify)==s.zeros(4,1)
    assert (c['decoder']*cx)[0]==sum(p)
    # Add a signed linear observer, distinct from energy; pullback recovers
    # four signed coordinate probes, not every direction within each triad.
    L=linear_form([1,0,0]*4);signed=compile_future([('tick',T)],[H,L])
    assert signed['rank']==8 and encode(signed,x)!=encode(signed,-x)
    # Affine translation needs degree 0/1 terms; multiple controls need words.
    U=affine_lift([[1,1],[0,1]],[0,1]);W=affine_lift([[0,1],[1,0]],[0,0])
    z=compile_future([('a',U),('b',W)],[energy_form([[1,0]],[0])])
    assert z['rank']==6
    h=s.Matrix([2,-3]);code=encode(z,h)
    for u,M in [('a',U),('b',W)]:
        h1=(M*s.Matrix([*h,1]))[:2];assert encode(z,h1)==z['transitions'][u]*code
    zero=compile_future([('tick',T)],[s.zeros(13)]);assert zero['rank']==0
    assert encode(zero,[0]*12).shape==(0,1) and zero['decoder']*encode(zero,[0]*12)==s.zeros(1,1)
    return dict(energy_rank=4,first_separating_depth=3,signed_rank=8,
                switched_affine_rank=6,zero_rank=0,source_fixture='ideal-real equations',training_steps=0)


if __name__=='__main__':
    print(certificate())
    print(source_forms_certificate())
