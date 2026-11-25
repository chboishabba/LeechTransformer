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
    X=s.Matrix(list(x)+[1]);return s.Matrix([(X.T*H*X)[0] for H in c['basis']])


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
    return dict(energy_rank=4,first_separating_depth=3,signed_rank=8,
                switched_affine_rank=6,zero_rank=0,source_fixture='ideal-real equations',training_steps=0)


if __name__=='__main__':
    print(certificate())
