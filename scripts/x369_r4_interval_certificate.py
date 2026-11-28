"""Exact rational enclosure reference for X369 B2--B6.

Inputs are externally certified intervals; this does not certify Torch kernels.
Original directed geometry: models/trialectic_wave_prior.py.
Retains existing Anatolii Kornienko attribution and F5 exclusion.
"""
from fractions import Fraction as Q
from dataclasses import dataclass
from decimal import Decimal, localcontext
from itertools import combinations

@dataclass(frozen=True)
class Interval:
    lo: Q
    hi: Q
    def __post_init__(self):
        object.__setattr__(self, "lo", Q(self.lo))
        object.__setattr__(self, "hi", Q(self.hi))
        if self.lo > self.hi:
            raise ValueError("reversed interval")
    def __add__(self, b):
        return Interval(self.lo+b.lo, self.hi+b.hi)
    def __neg__(self):
        return Interval(-self.hi, -self.lo)
    def __sub__(self,b):
        return self + (-b)
    def __mul__(self,b):
        t=[a*c for a in (self.lo,self.hi) for c in (b.lo,b.hi)]
        return Interval(min(t),max(t))

def point(x):
    return Interval(Q(x),Q(x))

def total(xs):
    return sum(xs,point(0))

def dot(a,b):
    if len(a)!=len(b):
        raise ValueError("width mismatch")
    return total(x*y for x,y in zip(a,b))

def fused_activation(x,w):
    if len(x)!=len(w) or not x or len(x)%3:
        raise ValueError("matching nonempty triadic width required")
    z=[]
    for i in range(0,len(x),3):
        a,b,c=x[i:i+3]
        s,f,r=w[i:i+3]
        z.extend((s*a+r*b+f*c,f*a+s*b+r*c,r*a+f*b+s*c))
    return z

def absolute_scores(base,unary,z,histories,rounding):
    if not len(base)==len(histories)==len(rounding):
        raise ValueError("expert count mismatch")
    return [b+unary+dot(z,h)+r for b,h,r in zip(base,histories,rounding)]

def pairwise_difference(i,j,base,z,histories,rounding,history_difference=None):
    # Optional directly certified history difference, not an assumed correlation.
    d=history_difference
    if d is None:
        if len(histories[i])!=len(histories[j]):
            raise ValueError("history width mismatch")
        d=[a-b for a,b in zip(histories[i],histories[j])]
    return base[i]-base[j]+dot(z,d)+rounding[i]-rounding[j]

def selection_certificate(order,n,pair):
    if not order or len(set(order))!=len(order) or any(i<0 or i>=n for i in order):
        raise ValueError("invalid selected order")
    outside=set(range(n))-set(order)
    membership=all(pair(i,j).lo>0 for i in order for j in outside)
    slots=all(pair(i,j).lo>0 for i,j in zip(order,order[1:]))
    return membership, membership and slots

def exp_point(x):
    # Decimal.exp is correctly rounded (ROUND_HALF_EVEN); adjacent
    # representable decimals enclose it. First enclose rational conversion.
    with localcontext() as c:
        from decimal import ROUND_FLOOR, ROUND_CEILING
        c.prec=50
        c.rounding=ROUND_FLOOR
        low=Decimal(x.numerator)/Decimal(x.denominator)
        c.rounding=ROUND_CEILING
        high=Decimal(x.numerator)/Decimal(x.denominator)
        a=low.exp().next_minus()
        b=high.exp().next_plus()
        if not a.is_finite() or not b.is_finite():
            raise ValueError("exponential outside supported range")
        return Interval(max(Q(0),Q(a)),Q(b))

def selected_softmax(order,pair):
    if not order or len(set(order))!=len(order):
        raise ValueError("invalid selection")
    out=[]
    for i in order:
        terms=[]
        for j in order:
            if j!=i:
                d=pair(j,i)
                terms.append(Interval(exp_point(d.lo).lo,exp_point(d.hi).hi))
        den=point(1)+total(terms)
        out.append(Interval(1/den.hi,1/den.lo))
    return out

def response(weights,values):
    return dot(weights,values)

def nat_cell(y,scale):
    q=Q(scale)
    if q<=0 or y.lo<0:
        return None
    lo,hi=q*y.lo,q*y.hi
    n=lo.numerator//lo.denominator
    return n if hi<n+1 else None

def self_test():
    x=list(map(point,[2,3,5]))
    w=list(map(point,[7,11,13]))
    h=[list(map(point,[17,19,23])),list(map(point,[1,2,3]))]
    z=fused_activation(x,w)
    a,b,c=[v.lo for v in x]
    A,B,C=[v.lo for v in h[0]]
    assert dot(z,h[0])==point(7*(a*A+b*B+c*C)+11*(a*B+b*C+c*A)+13*(a*C+b*A+c*B))
    base=[point(0),point(0)]
    err=[point(0),point(0)]
    pair=lambda i,j:pairwise_difference(i,j,base,z,h,err)
    scores=absolute_scores(base,Interval(-100,100),z,h,err)
    assert pair(0,1).lo>0
    assert selection_certificate([0],2,pair)==(True,True)
    assert selection_certificate([0],2,lambda i,j:point(0))==(False,False)
    assert selection_certificate([0,1],2,lambda i,j:point(0))==(True,False)
    p=selected_softmax([0,1],lambda i,j:point(0))
    assert all(v.lo<=Q(1,2)<=v.hi for v in p)
    y=response(p,[point(-1),point(1)])
    assert y.lo<=0<=y.hi
    assert nat_cell(Interval(Q(21,100),Q(29,100)),10)==2
    assert nat_cell(Interval(Q(21,100),Q(3,10)),10) is None
    assert nat_cell(Interval(-1,1),10) is None
    # Large common unary term can erase an ideal-real strict gap in float.
    assert float(2**54+1)==float(2**54)
    print("X369 rational enclosure reference controls PASS")

if __name__=="__main__":
    self_test()
