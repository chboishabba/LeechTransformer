import sympy as s
from itertools import product
I=s.eye(3); R=s.Matrix([[0,0,1],[1,0,0],[0,1,0]]); J=s.Matrix([[1,0,0],[0,0,1],[0,1,0]])
G=[s.diag(R**k*J**b,(-1)**b*R**k*J**b)*(-1)**n for k,b,n in product(range(3),range(2),range(2))]
assert len({tuple(g) for g in G})==12
assert all(a*b in G for a,b in product(G,repeat=2))
# Hom commutant: projector on 36-dimensional matrices
H=sum((s.kronecker_product(g,g) for g in G),s.zeros(36))/12
assert H*H==H and H.T==H and H.rank()==6
# first-operand action on logits; de-rotate each row, 3-cycle difference
D=s.Matrix([[-1,1,0],[0,-1,1],[1,0,-1]])
P=s.ones(3)/3
assert D.T*D==3*(I-P)
# training-only edges cannot constrain the missing node, even with both shifts
nodes=list(product(range(3),repeat=2));train=nodes[:-1]
edges=[(x,tuple((v+(j==i))%3 for j,v in enumerate(x))) for x in train for i in range(2)]
edges=[(x,y) for x,y in edges if y in train];B=s.zeros(len(edges),9)
for k,(x,y) in enumerate(edges):B[k,nodes.index(x)]=-1;B[k,nodes.index(y)]=1
assert len(edges)==14 and B.rank()==7 and B[:,8]==s.zeros(14,1)
# full two-generator torus connected
E=s.zeros(18,9)
for k,(x,i) in enumerate(product(nodes,range(2))):
 y=tuple((v+(j==i))%3 for j,v in enumerate(x));E[k,nodes.index(x)]=-1;E[k,nodes.index(y)]=1
assert E.rank()==8
# exact carry using ties-to-even with rational scaling; independent 12 group actions signed perms
from fractions import Fraction
for g in G:
 for x in [s.Matrix(v) for v in product([-s.Rational(3,2),-s.Rational(1,2),0,s.Rational(1,2),s.Rational(3,2)],repeat=2) for v in [v+(0,0,0,0)]]:
  c=x.applyfunc(round); assert (g*x).applyfunc(round)==g*c
# non-signed-permutation mixing yields collision
A=s.Matrix([[1,1],[0,1]]);a=s.Matrix([s.Rational(2,5),s.Rational(2,5)]);b=s.zeros(2,1)
assert a.applyfunc(round)==b.applyfunc(round) and (A*a).applyfunc(round)!=(A*b).applyfunc(round)
print('PASS group 12, Reynolds rank 6, C3 defect identity, train edges 14 rank 7 vs full rank 8, signed permutation rounding and carry collision')
