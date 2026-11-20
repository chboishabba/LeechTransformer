"""Exact rational two-channel quantization reference, not a native HIP certificate."""
from fractions import Fraction as Q
from scripts.x369_r4_interval_certificate import (
    Interval, point, total, selection_certificate, selected_softmax, response, nat_cell)

def quantize(x, quantum):
    x, q = Q(x), Q(quantum)
    if q <= 0: raise ValueError("positive quantum required")
    # Exact nearest grid, ties to even; no saturation.
    return q * round(x/q)

def encode(triads, weights, quantum, omit_dc=False):
    """Return ideal compressed score and signed omitted/quantized residual."""
    if len(triads)!=len(weights) or not triads: raise ValueError("triad count")
    candidate=Q(0); error=Q(0)
    for (s,f,r),(ws,wf,wr) in zip(triads,weights):
        s,f,r,ws,wf,wr=map(Q,(s,f,r,ws,wf,wr))
        dc=s+f+r
        sq,fq=quantize(s,quantum),quantize(f,quantum)
        dq=Q(0) if omit_dc else quantize(dc,quantum)
        candidate+=(ws-wr)*sq+(wf-wr)*fq+wr*dq
        error+=(ws-wr)*(s-sq)+(wf-wr)*(f-fq)+wr*(dc-dq)
    return candidate,error

def certify(order, candidate, error, values, scale, *, source_rounding, candidate_rounding, softmax_error, response_error):
    """Common pair enclosure covers BOTH source and compressed consumers.
    All arithmetic/FFN intervals must be supplied by their owning producers.
    No assertion that zero native rounding is valid for actual Torch.
    """
    n=len(candidate)
    if n<1 or any(len(v)!=n for v in (error,values,source_rounding,candidate_rounding)):
        raise ValueError("expert count mismatch")
    if len(softmax_error)!=len(order): raise ValueError("slot count mismatch")
    def pair(i,j):
        if i==j: return point(0)
        c=point(Q(candidate[i])-Q(candidate[j]))
        a=c+error[i]-error[j]+source_rounding[i]-source_rounding[j]
        b=c+candidate_rounding[i]-candidate_rounding[j]
        return Interval(min(a.lo,b.lo),max(a.hi,b.hi))
    member,ordered=selection_certificate(order,n,pair)
    if not ordered: return dict(membership=member,ordered=False,nat=None)
    weights=[w+e for w,e in zip(selected_softmax(order,pair),softmax_error)]
    y=response(weights,[values[i] for i in order])+response_error
    return dict(membership=member,ordered=True,nat=nat_cell(y,scale),response=y)

def self_test():
    t=[(2,3,-5),(4,-1,7)]; w=[(1,2,3),(4,5,6)]
    for omit in (False,True):
        c,e=encode(t,w,Q(3,2),omit)
        assert c+e==sum(sum(a*b for a,b in zip(u,v)) for u,v in zip(t,w))
    z=[point(0)]*3
    kw=dict(source_rounding=z,candidate_rounding=z,softmax_error=[point(0)],response_error=point(0))
    assert certify([2],[0,1,5],z,[point(0),point(0),point(Q(1,4))],10,**kw)["nat"]==2
    assert not certify([2],[0,5,5],z,z,10,**kw)["ordered"]
    assert not certify([2],[0,1,5],[point(0),point(10),point(0)],z,10,**kw)["ordered"]
    assert certify([2],[0,1,5],z,[point(0),point(0),Interval(Q(29,100),Q(3,10))],10,**kw)["nat"] is None
    print("X369 exact quantization reference controls PASS")
if __name__=="__main__":self_test()
