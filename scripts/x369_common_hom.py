"""Exact rational signed S3 x C2 common-Hom audit.
Original Leech-LILA: Anatolii Kornienko; DASHI extension.
Row-action convention of audit_x369_joint_action/hidden_generators.
Stored float values are exact dyadics, not certified underlying real maps.
"""
import sympy as s


def exact(A):
    if hasattr(A,'detach'):A=A.detach().cpu().tolist()
    def q(x):
        if isinstance(x,(float,s.Float)):
            if not s.sympify(x).is_finite:raise ValueError('finite rational matrix required')
            return s.Rational(x)
        x=s.sympify(x)
        if x.is_Rational is not True:raise ValueError('finite rational matrix required')
        return x
    return s.Matrix(A).applyfunc(q)


def relations(A):
    R,J,N=(exact(A[k]) for k in ('R','J','N'));n=R.rows
    if not n or any(M.shape!=(n,n) for M in (R,J,N)):raise ValueError('square generators required')
    I=s.eye(n)
    return {'R3':R**3-I,'J2':J**2-I,'N2':N**2-I,'JRJ_R2':J*R*J-R**2,'NR_RN':N*R-R*N,'NJ_JN':N*J-J*N}


def canonical(A):
    """Canonical rational column basis for transposed row representations."""
    if any(any(M) for M in relations(A).values()):raise ValueError('not an exact representation')
    R,J,N=(exact(A[k]).T for k in ('R','J','N'));I=s.eye(R.rows)
    P=(I+R+R**2)/3;Q=I-P;cols=[];blocks=[];multiplicities={}
    for a in (1,-1):
        Z=(I+a*N)/2
        for b in (1,-1):
            V=(Z*P*(I+b*J)/2).columnspace();key=(a,'dc',b)
            multiplicities[key]=len(V)
            for v in V:blocks.append((key,len(cols),1));cols.append(v)
        V=(Z*Q*(I+J)/2).columnspace();key=(a,'phase',0)
        multiplicities[key]=len(V)
        for v in V:blocks.append((key,len(cols),2));cols.extend((v,R*v))
    B=s.Matrix.hstack(*cols)
    assert B.shape==I.shape and B.det()!=0
    return B,blocks,multiplicities


def relation_quotient(A):
    """Largest row quotient satisfying the six defining relations.
    Close relator row images under all right generator actions. Any map
    into an exact target annihilates this invariant subspace.
    """
    A={k:exact(A[k]) for k in ('R','J','N')};n=A['R'].rows
    W=s.Matrix.vstack(*relations(A).values()).rowspace()
    while W:
        B=s.Matrix.vstack(*W)
        V=s.Matrix.vstack(B,*(B*M for M in A.values())).rowspace()
        if len(V)==len(W):break
        W=V
    B=s.Matrix.vstack(*W) if W else s.zeros(0,n)
    vectors=B.nullspace();P=s.Matrix.hstack(*vectors) if vectors else s.zeros(n,0)
    assert B*P==s.zeros(B.rows,P.cols)
    if not vectors:return B,P,{}
    # A P=P Abar has a unique solution since P has full column rank.
    bar={k:P.gauss_jordan_solve(M*P)[0] for k,M in A.items()}
    assert all(A[k]*P==P*bar[k] for k in A) and not any(any(M) for M in relations(bar).values())
    return B,P,bar


def audit(A,T):
    """Exact maximal common-Hom rank for two admitted group actions.
    Nonrepresentations factor through their largest exact relation quotient.
    Only the quotient reverse map is equivariant in that case.
    """
    A={k:exact(A[k]) for k in ('R','J','N')};T={k:exact(T[k]) for k in ('R','J','N')}
    target_bad=relations(T)
    if any(any(M) for M in target_bad.values()):raise ValueError('target must be exact')
    bad={k:M for k,M in relations(A).items() if any(M)}
    if bad:
        W,P,bar=relation_quotient(A)
        r=dict(status='relation_obstruction',invertible=False,relation_witnesses={k:next((i,j,str(M[i,j])) for i in range(M.rows) for j in range(M.cols) if M[i,j]) for k,M in bad.items()},relation_space=W,quotient_projection=P,quotient_dimension=P.cols)
        if not P.cols:
            r.update(max_rank=0,hom_dimension=0,forward=s.zeros(P.rows,T['R'].rows));return r
        q=audit(bar,T);C=P*q['forward']
        assert C.rank()==q['max_rank'] and all(A[k]*C==C*T[k] for k in A)
        r.update(max_rank=q['max_rank'],hom_dimension=q['hom_dimension'],forward=C,
                 quotient_forward=q['forward'],quotient_reverse=q['reverse'],quotient_multiplicities=q['source_multiplicities'],target_multiplicities=q['target_multiplicities'])
        return r
    B,bs,m=canonical(A);D,ds,n=canonical(T)
    H=s.zeros(D.rows,B.rows);K=s.zeros(B.rows,D.rows);rank=0;hom=0
    for key in m:
        a=[(i,d) for k,i,d in bs if k==key];b=[(i,d) for k,i,d in ds if k==key]
        hom+=len(a)*len(b)
        for (i,d),(j,e) in zip(a,b):
            assert d==e;H[j:j+d,i:i+d]=s.eye(d);K[i:i+d,j:j+d]=s.eye(d);rank+=d
    C=(D*H*B.inv()).T;L=(B*K*D.inv()).T
    assert all(A[k]*C==C*T[k] and T[k]*L==L*A[k] for k in A)
    assert C.rank()==rank and C*L*C==C and L*C*L==L
    invertible=rank==C.rows==C.cols
    if invertible:assert C*L==s.eye(C.rows) and L*C==s.eye(C.cols)
    return dict(status='equivalent' if invertible else 'multiplicity_obstruction',invertible=invertible,
                source_multiplicities=m,target_multiplicities=n,hom_dimension=hom,max_rank=rank,forward=C,reverse=L)


def target_action(joint=True):
    """Independent original task endpoint on basis vectors, not source action."""
    from scripts.audit_x369_ordered_signed_task import endpoint
    z=(0,0,0);rows={}
    for k,w,g in [('R','a',(1,0,0,0)),('J','j',(0,0,1,0)),('N','n',(0,0,0,1))]:
        vs=[]
        for i in range(6 if joint else 3):
            v=tuple(int(i==j) for j in range(6 if joint else 3));m,d=(v[:3],v[3:]) if joint else (z,v)
            p,q,_=endpoint(w,g,m,d);vs.append(p+q if joint else q)
        rows[k]=s.Matrix(vs)
    return rows


def fitted_receipt(fit,joint=True):
    """Read existing independently fitted A maps; never replace by known action."""
    from hashlib import sha256
    import json
    A={k:exact(fit['A'][k]) for k in ('R','J','N')};T=target_action(joint)
    def matrix(M):return [[str(x) for x in row] for row in M.tolist()]
    r=audit(A,T);r['source_generators']={k:matrix(v) for k,v in A.items()};r['target_generators']={k:matrix(v) for k,v in T.items()}
    for k in ('forward','reverse','relation_space','quotient_projection','quotient_forward','quotient_reverse'):
        if k in r:r[k]=matrix(r[k])
    for k in ('source_multiplicities','target_multiplicities','quotient_multiplicities'):
        if k in r:r[k]=[(list(t),v) for t,v in r[k].items()]
    r['stored_matrix_sha256']=sha256(json.dumps(r['source_generators'],sort_keys=True).encode()).hexdigest()
    r.update(exact_stored_values_only=True,learned_same_action_established=False,training_steps=0)
    return r


def certificate():
    T=target_action();U=target_action(False)
    assert audit(T,T)['hom_dimension']==6 and audit(U,U)['hom_dimension']==2
    P=s.eye(6);P[0,4]=s.Rational(3,5);P[2,1]=2;P[5,3]=-1
    A={k:P*T[k]*P.inv() for k in T};r=audit(A,T)
    assert r['invertible'] and r['max_rank']==6 and r['hom_dimension']==6
    # Same dimensions, all relations exact, wrong signed reflection sector.
    wrong=dict(T);wrong['J']=s.diag(T['J'][:3,:3],T['J'][:3,:3]);r=audit(wrong,T)
    assert not r['invertible'] and r['max_rank']==5
    # Exact group action but a central-sign mismatch leaves no common Hom.
    wrong=dict(T);wrong['N']=s.eye(6);r=audit(wrong,T)
    assert r['max_rank']==0 and r['hom_dimension']==0
    # No invertible weld can repair a nonzero defining relation.
    bad=dict(T);bad['R']=T['R'].copy();bad['R'][0,0]+=s.Rational(1,2**20)
    r=audit(bad,T);assert r['status']=='relation_obstruction' and 'R3' in r['relation_witnesses']
    # Residual-only target embedded in joint source: maximal rank3, not bijective.
    r=audit(T,U);assert r['max_rank']==3 and not r['invertible'] and r['hom_dimension']==3
    # Nonrepresentation with an exact six-dimensional common quotient.
    M={k:s.diag(T[k],2 if k=='R' else 1) for k in T};r=audit(M,T)
    assert r['status']=='relation_obstruction' and r['quotient_dimension']==6 and r['max_rank']==6 and r['hom_dimension']==6
    M={k:s.Matrix([[2 if k=='R' else 1]]) for k in T};r=audit(M,T)
    assert r['quotient_dimension']==0 and r['max_rank']==0
    # Cross-check the geometric rank/Hom dimension against literal linear equations.
    for src in (T,A,bad,{k:s.diag(T[k],2 if k=='R' else 1) for k in T}):
        r=audit(src,T);n=src['R'].rows;d=6
        L=s.Matrix.vstack(*(s.kronecker_product(src[k],s.eye(d))-s.kronecker_product(s.eye(n),T[k].T) for k in T))
        assert n*d-L.rank()==r['hom_dimension']
    import json
    json.dumps(fitted_receipt({'A':A}));json.dumps(fitted_receipt({'A':{k:[[float(x) for x in row] for row in v.tolist()] for k,v in A.items()}}))
    assert exact([[0.1]])[0,0]==s.Rational(0.1)
    print('PASS: signed sectors, BIDI conjugacy, reflection/central/relator obstructions, rank-3 quotient, largest relation quotient, four literal Hom dimension checks, dyadic JSON')


if __name__=='__main__':certificate()
