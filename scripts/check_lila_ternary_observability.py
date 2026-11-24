from itertools import product, permutations
import sympy as s
from fractions import Fraction

# Two extreme TopK queries, all legal tie orders. Scores are exact integers.
# Recover selected scores using weighted mean and exact logit differences.
w=s.symbols('w', positive=True)
u,v=s.symbols('u v', real=True)
m=w*u+(1-w)*v
d=u-v
assert s.expand(m+(1-w)*d-u)==0
assert s.expand(m-w*d-v)==0
R=s.Matrix([[0,1,0],[0,0,1],[1,0,0]])
assert R*R.T==s.eye(3) and R**3==s.eye(3)
# Source-legal directed self channel after rotate3: coefficient basis probes.
h=s.Matrix(s.symbols('h:3'))
for k in range(3):
    q=R.T*s.eye(3)[:,k]
    assert (R*q).dot(h)==h[k]
    assert (R*(-q)).dot(h)==-h[k]
# Symbolic selected inversion with actual softmax exp ratios for integer scores.
# exp(d) positive; use exp(d) exactly then simplify.
for a,b in product(range(-1,2),repeat=2):
    wa=s.exp(a)/(s.exp(a)+s.exp(b)); wb=1-wa
    mean=wa*a+wb*b
    assert s.simplify(mean+wb*(a-b)-a)==0
    assert s.simplify(mean-wa*(a-b)-b)==0
    assert s.simplify(s.log(s.exp(a-b))-(a-b))==0

def orders(z):
    return {p[:2] for p in permutations(range(4)) if all(z[p[i]]>=z[p[i+1]] for i in range(3))}
count=0; missing=0
for z in product(range(-1,2),repeat=4):
    for top,bottom in product(orders(z),orders(tuple(-a for a in z))):
        known={i:z[i] for i in (*top,*bottom)}
        upper=z[top[-1]]; lower=z[bottom[-1]]
        for i in range(4):
            if i not in known:
                assert lower<=z[i]<=upper
                assert upper==lower
                known[i]=upper;missing+=1
        assert tuple(known[i] for i in range(4))==z
        count+=1
# Exact controlled switched observability: delayed visibility, 3 modes incl noncommuting.
C=s.Matrix([[1,0,0]])
A=s.Matrix([[0,1,0],[0,0,1],[0,0,0]])
B=s.Matrix([[1,0,0],[0,0,1],[0,1,0]])
rows=C
ranks=[rows.rank()]
while True:
    new=s.Matrix.vstack(rows,rows*A,rows*B)
    basis=s.Matrix.vstack(*new.rowspace())
    if basis.rank()==rows.rank():break
    rows=basis;ranks.append(rows.rank())
assert ranks==[1,2,3]
# Fixed-grid weight rounding changes exact observable dimension in both directions.
I=s.eye(3);J=s.diag(1,1,s.Rational(7,8));Q=s.eye(3)
assert (J-I).rank()==1 and (Q-I).rank()==0
# For fixed-unit rounding, a row dependency can be destroyed.
D=s.Matrix([[s.Rational(3,5),s.Rational(2,5)],[s.Rational(6,5),s.Rational(4,5)]])
T=s.Matrix([[1,0],[1,1]])
assert D.rank()==1 and T.rank()==2
# BitNet-style mean-absolute scale can destroy an exact fixed vector (ideal scales).
W=s.Matrix([[1,0],[0,1]]); scale=sum(abs(t) for t in W)/4
Wq=scale*W
assert (W-I[:2,:2]).rank()==0 and (Wq-I[:2,:2]).rank()==2
# signed support mask dot product and 2-bit encoding using BitNet's +2 mapping.
pack_count=0
for t in product((-1,0,1),repeat=4):
    packed=sum((x+2)<<(2*i) for i,x in enumerate(t))
    unpack=tuple(((packed>>(2*i))&3)-2 for i in range(4))
    assert unpack==t
    for x in product((-128,0,127),repeat=4):
        assert sum(a*b for a,b in zip(t,x))==sum(b for a,b in zip(t,x) if a==1)-sum(b for a,b in zip(t,x) if a==-1)
        pack_count+=1
print('PASS',s.__version__,'score vectors 81, legal ordered top/bottom pairs',count,'missing tie recoveries',missing,'pack/dot',pack_count,'switched ranks',ranks)
