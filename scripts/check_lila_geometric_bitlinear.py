from itertools import product, permutations
from collections import Counter
import sympy as s

I=s.eye(3); R=s.Matrix([[0,1,0],[0,0,1],[1,0,0]])
J=s.Matrix([[0,0,1],[0,1,0],[1,0,0]])
G=[R**k for k in range(3)]+[R**k*J for k in range(3)]
assert R**3==I and J**2==I and J*R*J==R**2
assert sum(G[:3],s.zeros(3))==sum(G[3:],s.zeros(3))==s.ones(3)
F=s.Matrix.hstack(*(g.reshape(9,1) for g in G))
assert F.rank()==5 and F.nullspace()==[s.Matrix([-1,-1,-1,1,1,1])]
# Two independent signed-permutation orbits: global sign and C3, not reflection.
T=list(product((-1,0,1),repeat=3)); unseen=set(T); sizes=[]
while unseen:
    t=s.Matrix(min(unseen)); orbit={tuple(sign*g*t) for g in G[:3] for sign in (-1,1)}
    unseen-=orbit;sizes.append(len(orbit))
assert sorted(sizes)==[1,2,6,6,6,6]
# Symmetric square in monomial (not normalized orthogonal) coordinates.
x=s.Matrix(s.symbols('x:3')); pairs=[(0,0),(1,1),(2,2),(0,1),(1,2),(2,0)]
def lift(v):return s.Matrix([v[i]*v[j] for i,j in pairs])
def sym2(A):
    y=A*x
    return s.Matrix([[s.expand(y[i]*y[j]).coeff(x[k],2) if k==l else s.expand(y[i]*y[j]).coeff(x[k],1).coeff(x[l],1) for k,l in pairs] for i,j in pairs])
S=[sym2(g) for g in G]
for i,g in enumerate(G):
    assert S[i]*lift(x)==lift(g*x)
    assert sym2(-g)==S[i]
    assert S[i].T*S[i]==s.eye(6)
    for j,h in enumerate(G):
        assert S[i]*S[j]==S[G.index(g*h)]
# Independent ternary coefficients create a nontrivial relation; lift AFTER summing.
C=[]; rank_counts=Counter(); lift_counts=Counter()
for t in T:
    A=sum((c*g for c,g in zip(t,G[:3])),s.zeros(3)); C.append(A)
    r=A.rank(); rank_counts[r]+=1
    L=sym2(A); assert s.expand(L*lift(x)-lift(A*x))==s.zeros(6,1)
    assert L.rank()==r*(r+1)//2; lift_counts[r+L.rank()]+=1
assert rank_counts=={0:1,1:2,2:6,3:18}
assert lift_counts=={0:1,2:2,5:6,9:18}
assert sym2(I+R)!=sym2(I)+sym2(R)
assert max(abs(v) for v in sym2(I+R+R**2))==2
# All 729 six-coefficient S3 sums and their exact gauge classes.
classes={}; ranks=Counter()
for t in product((-1,0,1),repeat=6):
    A=tuple(F*s.Matrix(t)); classes.setdefault(A,[]).append(t)
assert len(classes)==665
for A,ts in classes.items():
    for t in ts:
        diff=s.Matrix(t)-s.Matrix(ts[0]); assert F*diff==s.zeros(9,1)
    ranks[s.Matrix(3,3,A).rank()]+=1
# Full tensor has symmetric six and alternating three; cross-product law.
h=s.Matrix(s.symbols('h:3'))
for g in G:
    assert (g*x).cross(g*h)==g.det()*g*(x.cross(h))
# Cross terms recover via polarization, and oriented antisymmetry cannot be recovered from symmetric part alone.
assert s.expand(lift(x+h)-lift(x)-lift(h)-s.Matrix([x[i]*h[j]+h[i]*x[j] for i,j in pairs]))==s.zeros(6,1)
assert x.cross(h)==-h.cross(x)
# Exact fixed spaces for prior 9--12 tagged recurrence.
assert [9+(A-I).rank() for A in [I,-R-R**2,R,s.zeros(3)]]==[9,10,11,12]
print('PASS',s.__version__,'orbits',sorted(sizes),'circulant ranks',dict(rank_counts),'lift ranks',dict(lift_counts),'S3 classes',len(classes),'S3 ranks',dict(sorted(ranks.items())),'group products',36)
