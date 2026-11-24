import sympy as s
from itertools import product
I=s.eye(3); Z=s.zeros(3); J=s.Matrix([[0,0,1],[0,1,0],[1,0,0]])
a,b=s.symbols('a b', positive=True)
C=s.zeros(9,12)
for j in range(3):
    C[3*j:3*j+3,0:3]=a*I
    C[3*j:3*j+3,3*(j+1):3*(j+2)]=b*I
K=s.Matrix.vstack(b*I,-a*I,-a*I,-a*I)
assert C*K==s.zeros(9,3)
# Restrict weights to a nonzero rational pair for numerical matrix regressions;
# the symbolic K identity and row relation below are general.
A=s.Matrix(3,3,s.symbols('A:9')); B=s.Matrix(3,3,s.symbols('B:9'))
T=s.diag(A,B,I,I)
R=C*T*K
assert s.expand(R[:3,:]-a*b*(A-B))==Z
assert s.expand(R[3:6,:]-a*b*(A-I))==Z
assert s.expand(R[6:9,:]-a*b*(A-I))==Z
Cr=C.subs({a:s.Rational(2,3),b:s.Rational(1,3)})
Kr=K.subs({a:s.Rational(2,3),b:s.Rational(1,3)})
# arbitrary affine offsets cancel in differences
matrices=[I,J,-J,Z,s.Matrix([[1,1,0],[0,1,0],[0,0,1]])]
count=0
for mats in product(matrices,repeat=4):
    D=s.Matrix.vstack(*(m-I for m in mats))
    Ts=[s.diag(*(mats[e] if e in (0,j) else I for e in range(4))) for j in (1,2,3)]
    # Restrict two-step rows to the instantaneous kernel; rank equality implies exact two-step quotient.
    RK=s.Matrix.vstack(*(Cr*Tj*Kr for Tj in Ts))
    assert RK.rank()==D.rank()
    # Check each common fixed vector is invariant under every selected transition.
    for v in D.nullspace():
        for Tj in Ts:
            assert Tj*Kr*v==Kr*v
    count+=1
for M,dim in [(I,9),(J,10),(-J,11),(Z,12)]:
    assert 9+(M-I).rank()==dim
P=(I+J)/2; Q=(I-J)/2
assert P*P==P and Q*Q==Q and P*Q==Z and P+Q==I
assert P.rank()==2 and Q.rank()==1
# Reflection-changing linear update: even consumer sees zero at t=0 and nonzero next.
v=s.Matrix([1,0,-1]); L=s.Matrix([[1,0,0],[0,0,0],[0,0,0]])
assert P*v==s.zeros(3,1) and P*L*v!=s.zeros(3,1)
print(f'X369_AFFINE_RESIDUAL_PASS {count} transition quadruples; symbolic block identities; ranks 9/10/11/12; signed reflection controls')
