"""Exact controlled frame receipt and full-domain linear guard obstruction.
Original Leech-LILA: Anatolii Kornienko; DASHI extension.
Ideal real/rational value semantics, not byte identity or fitted reachability.
"""
from itertools import product
from functools import lru_cache
import sympy as s
from models.geometric_reflection_fibre import GROUP, mul, inv, act


def conjugate(g,u):
    return mul(mul(g,u),inv(g))


def transform(g,h):
    return tuple(act(g,e[b^g[1]]) for e in h for b in range(2))


def flatten(rows):
    return tuple(x for row in rows for x in row)


def shape(h,E,W):
    return tuple(tuple(tuple(h[(2*e+b)*W:(2*e+b+1)*W]) for b in range(2)) for e in range(E))


def frame_receipt(u,h):
    """Same first-max order as canonical_controlled; one tie set owns K.
    Prefix index is a witness, not an independently authenticated proof.
    """
    if u not in GROUP or not h or any(len(e)!=2 for e in h):raise ValueError('invalid carrier')
    W=len(h[0][0])
    if not W or W%3 or any(len(b)!=W for e in h for b in e):raise ValueError('invalid width')
    c=max(conjugate(g,u) for g in GROUP)
    candidates=[(g,flatten(transform(g,h))) for g in GROUP if conjugate(g,u)==c]
    v=max(v for _,v in candidates);J=tuple(g for g,w in candidates if w==v);m=J[0]
    K=tuple(g for g in GROUP if g in {mul(j,inv(m)) for j in J})
    prefix=tuple((g,next((i for i,(x,y) in enumerate(zip(v,w)) if x!=y),None)) for g,w in candidates if g!=m)
    return dict(control=c,mark=m,stabilizer=K,maximizers=J,history=v,prefix=prefix)


@lru_cache(None)
def action_matrix(g,E,W):
    n=2*E*W;G=s.MutableSparseMatrix(n,n,{})
    for e,b,j in product(range(E),range(2),range(W)):
        k,r,a=g;dst=3*(j//3)+((2-j%3 if r else j%3)-k)%3
        G[(2*e+(b^r))*W+dst,(2*e+b)*W+j]=(-1)**a
    return s.ImmutableSparseMatrix(G)


def frame_matrices(receipt,E,W,selected):
    """Supplied admitted mark/stabilizer/selected set; no guard inference."""
    K=tuple(receipt['stabilizer']);m=receipt['mark'];ids=set(selected)
    if m not in GROUP or not K or len(set(K))!=len(K) or any(g not in GROUP for g in K) or (0,0,0) not in K or any(mul(a,b) not in K for a in K for b in K) or not ids<=set(range(E)):
        raise ValueError('invalid frame/selection')
    n=2*E*W;G=action_matrix(m,E,W);P=sum((action_matrix(g,E,W) for g in K),s.SparseMatrix(n,n,{}))/len(K)
    D=s.SparseMatrix(n,n,{(i,i):1 for i in range(n) if i//(2*W) in ids})
    return G,P,G.T*D*P,s.SparseMatrix.eye(n)-D


def cap_guard_rank():
    """Recover Q,V from the existing exact decoder; rank12, hidden dim84.
    Any affine linear extras with rank <84 on ker[Q;V] leave an antipode
    collision for the exact world-to-local mark at identity control.
    """
    from scripts.x369_capped_semantic_geometry import exact_repair
    E,W=4,12;n=96
    Q=s.Matrix([exact_repair(shape([int(i==j) for i in range(n)],E,W),(0,)*6,range(E),1)['defect'] for j in range(n)]).T
    R=s.eye(n)-Q.T*Q;V=Q[:,:W]*R[:W,:];L=Q.col_join(V)
    assert Q*Q.T==s.eye(6) and L.rank()==12
    basis=L.nullspace();assert len(basis)==84
    for w in basis:
        w=w/(1+sum(abs(x) for x in w));assert L*w==s.zeros(12,1)
        h=shape(w,E,W);neg=shape(-w,E,W)
        assert all(sum(x*x for x in b)<=1 for e in h for b in e)
        p=frame_receipt((0,0,0),h);q=frame_receipt((0,0,0),neg)
        assert p['history']==q['history'] and p['mark']!=q['mark']
        assert [sum(x*x for x in b) for e in h for b in e]==[sum(x*x for x in b) for e in neg for b in e]
    return dict(initial_linear_rank=12,hidden_linear_dimension=84,antipodal_basis_checks=84)


def certificate():
    checks=0;counts=set()
    for h0 in product((-1,0,1),repeat=6):
        h=shape(h0,1,3)
        for u in GROUP:
            r=frame_receipt(u,h);m=r['mark'];v=r['history'];c=r['control'];local=shape(v,1,3)
            pairs=[(GROUP.index(conjugate(g,u)),flatten(transform(g,h))) for g in GROUP]
            assert pairs[GROUP.index(m)]==max(pairs) and GROUP.index(m)==pairs.index(max(pairs))
            K=tuple(g for g in GROUP if conjugate(g,c)==c and flatten(transform(g,local))==v)
            assert K==r['stabilizer']
            assert all(i is None or v[i]>flatten(transform(g,h))[i] for g,i in r['prefix'])
            counts.add(len(r['prefix'])+1);checks+=1
    assert counts=={4,6,12}
    E,W=4,12;n=96;h=s.Matrix([s.Rational(i%7-3,8) for i in range(n)])
    for a,b in product(GROUP,repeat=2):assert action_matrix(a,E,W)*action_matrix(b,E,W)==action_matrix(mul(a,b),E,W)
    # Exercise every stabilizer type appearing in a small exhaustive carrier.
    projection_checks=0
    for u in GROUP:
        for raw in (h,s.zeros(n,1),s.ones(n,1)):
            r=frame_receipt(u,shape(raw,E,W));G,P,A,B=frame_matrices(r,E,W,(0,2))
            assert G*raw==s.Matrix(r['history']) and P*G*raw==G*raw and P*P==P and P.T==P
            assert A*G*raw+B*raw==raw
            # Arbitrary local candidate: inactive original state restored.
            z=h;v=A*z+B*raw
            for i in (1,3):assert v[i*2*W:(i+1)*2*W,0]==raw[i*2*W:(i+1)*2*W,0]
            projection_checks+=1
    return dict(frame_checks=checks,candidate_counts=sorted(counts),group_products=144,projection_checks=projection_checks,**cap_guard_rank())


if __name__=='__main__':
    print(certificate())
