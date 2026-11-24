from itertools import product
from collections import Counter
import sympy as sp
import torch
from models.geometric_reflection_fibre import (GROUP, IDENTITY, mul, inv, act, stabilizer, encode, indexed_step, reflection_chart, signed_pair_apply, signed_pair_residual)


def main():
    for g,h,k in product(GROUP,repeat=3):
        assert mul(mul(g,h),k)==mul(g,mul(h,k))
    for g in GROUP:
        assert mul(g,inv(g))==mul(inv(g),g)==IDENTITY
    for n in (1,2):
        vectors=list(product((-1,0,1),repeat=3*n))
        classes={encode(v).canonical for v in vectors}
        hist=Counter(12//len(stabilizer(c)) for c in classes)
        assert sum(k*v for k,v in hist.items())==len(vectors)
        for v in vectors:
            e=encode(v); assert e.decode()==v
            assert len({act(g,v) for g in GROUP})*len(stabilizer(v))==12
            for g in GROUP:
                assert e.transport(g)==encode(act(g,v))
                for h in GROUP:
                    assert e.transport(h).transport(g)==e.transport(mul(g,h))
        print('triads',n,'states',len(vectors),'classes',len(classes),'orbit histogram',dict(sorted(hist.items())))
    # Real-arithmetic equivariant nonlinear recurrence: cubic coordinate map
    # and signed permutation-equivariant input combination. Use integers.
    def T(x,h):
        y=tuple(a+b for a,b in zip(x,h))
        return y,tuple(a**3-b for a,b in zip(y,h))
    count=0
    for h in product((-1,0,1),repeat=3):
        state=encode(h)
        for x in ((1,0,-1),(0,0,0),(-1,1,0),(1,1,1)):
            y,h=T(x,h); yy,state=indexed_step(state,x,T)
            assert y==yy and h==state.decode();count+=1
    print('indexed steps',count)
    z=torch.tensor(list(product((-1,0,1),repeat=6)),dtype=torch.float64)
    R=lambda t:t.reshape(-1,2,3).roll(-1,-1).reshape_as(t)
    J=lambda t:t.reshape(-1,2,3).flip(-1).reshape_as(t)
    c=reflection_chart(z)
    assert torch.equal(reflection_chart(R(z)),c)
    assert torch.equal(reflection_chart(J(z)),c.flip(-1))
    assert torch.equal(reflection_chart(-z),-c)
    assert torch.equal((c==0).all(-1),(z==0).all(-1))
    f=lambda t:torch.nn.functional.gelu(t)+.25
    y,_=signed_pair_apply(z,f)
    for transform in (R,J,lambda t:-t):
        assert torch.allclose(signed_pair_apply(transform(z),f)[0],transform(y),atol=1e-14,rtol=1e-14)
    m,d,marks=signed_pair_residual(z,f)
    assert torch.equal(m,y) and torch.equal(marks,c)
    for transform,parity in ((R,1),(J,-1),(lambda t:-t,1)):
        mm,dd,_=signed_pair_residual(transform(z),f)
        assert torch.allclose(mm,transform(m),atol=1e-14,rtol=1e-14)
        assert torch.allclose(dd,parity*transform(d),atol=1e-14,rtol=1e-14)
    assert torch.allclose(m+d,c[:,0:1]*f(c[:,0:1]*z),atol=1e-14,rtol=1e-14)
    assert torch.allclose(m-d,c[:,1:2]*f(c[:,1:2]*z),atol=1e-14,rtol=1e-14)
    opposite=(c[:,0]==-c[:,1])&(c[:,0]!=0)
    assert torch.allclose(y[opposite],z[opposite]/2,atol=1e-14,rtol=1e-14)
    # exact symbolic identity: opposite marks cancel the even component.
    a,b=sp.symbols('f_plus f_minus'); assert sp.expand((a-b)/2-((a+b)/2-b))==0
    print('PASS',torch.__version__,sp.__version__,'chart rows',len(z),'opposite',int(opposite.sum()),'aligned',int((~opposite & (c[:,0]!=0)).sum()))

if __name__=='__main__':main()
