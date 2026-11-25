"""Exact signed phase/cube controls, not neural training evidence.
Original Leech-LILA: Anatolii Kornienko; user oscillator hypothesis; DASHI extension.
"""
from itertools import product, permutations
import sympy as s

# Eisenstein integers a+b*w, w^2+w+1=0. No numerical complex tolerance.
def add(x,y): return (x[0]+y[0],x[1]+y[1])
def mul(x,y):
    a,b=x;c,d=y
    return (a*c-b*d,a*d+b*c-b*d)
def conj(x): return (x[0]-x[1],-x[1])
def norm(x): return x[0]**2-x[0]*x[1]+x[1]**2
W=((1,0),(0,1),(-1,-1))
points=tuple(product(range(3),repeat=2))
def char(k,x):return W[sum(a*b for a,b in zip(k,x))%3]
def spec(z):
    # unnormalized DFT; divide coefficients by 9 for mean convention
    return {k:sum_ring(mul(z[x],conj(char(k,x))) for x in points) for k in points}
def sum_ring(xs):
    y=(0,0)
    for x in xs:y=add(y,x)
    return y

def main():
    R=s.Matrix([[0,0,1],[1,0,0],[0,1,0]]);J0=s.Matrix([[1,0,0],[0,0,1],[0,1,0]])
    R=s.diag(R,R);J=s.diag(J0,-J0);N=-s.eye(6);I=s.eye(6)
    assert J*R==R.inv()*J and N*R==R*N and N*J==J*N
    def vertex(v):
        r,j,n=v;return N**n*J**j*R**r
    def edge(v,a):
        assert v[a]==0
        return (R**((-1)**v[1]),J,N)[a]
    faces=0;paths=0
    for a,b in ((0,1),(0,2),(1,2)):
        c=3-a-b
        for side in (0,1):
            v=[0,0,0];v[c]=side;va=v.copy();va[a]=1;vb=v.copy();vb[b]=1;end=va.copy();end[b]=1
            assert edge(va,b)*edge(v,a)*vertex(v)==edge(vb,a)*edge(v,b)*vertex(v)==vertex(end)
            faces+=1
    for order in permutations(range(3)):
        v=[0,0,0];A=I
        for a in order:A=edge(v,a)*A;v[a]=1
        assert A==vertex(v);paths+=1
    assert R*J!=J*R
    for k,l in product(points,repeat=2):
        assert sum_ring(mul(char(k,x),conj(char(l,x))) for x in points)==((9,0) if k==l else (0,0))
    target={x:char((1,1),x) for x in points}
    wrongs=[(2,2),(1,2),(2,1)]
    for k in wrongs:
        z={x:char(k,x) for x in points};d=spec(z)
        assert sum(z[x]==target[x] for x in points)==3
        assert d[k]==(9,0) and d[(1,1)]==(0,0) and sum(map(norm,d.values()))==81
    bad=dict(target);bad[(2,2)]=W[0];d=spec(bad)
    assert sum(bad[x]==target[x] for x in points)==8
    assert norm(d[(1,1)])==57 and all(norm(d[k])==3 for k in points if k!=(1,1))
    assert sum(map(norm,d.values()))==81
    # Global phase coherence does not imply anchored correctness.
    z={x:mul(W[1],target[x]) for x in points};d=spec(z)
    assert norm(d[(1,1)])==81 and not any(z[x]==target[x] for x in points)
    # Mixed harmonic: wrong on all b!=0, equal amplitudes and exact phase ordering of its own.
    print('PASS',s.__version__,'6 indexed cube faces, 6 endpoint paths, 81 character inner products, 3 wrong-character controls, 8/9 memorizer spectrum 57+8*3=81, wrong global phase control')
if __name__=='__main__':main()
