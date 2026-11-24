"""Finite reachable chart refinement; exact symbolic control, not fitted LILA.
Original Leech-LILA: Anatolii Kornienko; DASHI extension.
"""
from itertools import product
from collections import deque
import sympy as s


def refine(states, labels, chart, step):
    """step returns (hashable observation, successor) or None (inadmissible).
    Fail closed if any successor leaves the declared finite domain.
    Return nested partitions, including initial chart and stable last partition.
    """
    states=tuple(states); labels=tuple(labels)
    if len(set(states))!=len(states) or not states:raise ValueError('unique nonempty domain required')
    domain=set(states);table={(z,u):step(z,u) for z in states for u in labels}
    if any(v is not None and v[1] not in domain for v in table.values()):raise ValueError('domain not successor closed')
    def classes(keys):
        ids={};return {z:ids.setdefault(k,len(ids)) for z,k in zip(states,keys)}
    p=classes(chart(z) for z in states);history=[p]
    while True:
        q=classes((p[z],tuple((False,) if table[z,u] is None else (True,table[z,u][0],p[table[z,u][1]]) for u in labels)) for z in states)
        if q==p:return history,table
        history.append(q);p=q


def witness(a,b,labels,chart,table):
    """Shortest distinguishing word including chart at all visited states."""
    queue=deque([(a,b,())]);seen={(a,b)}
    while queue:
        x,y,w=queue.popleft()
        if chart(x)!=chart(y):return w
        for u in labels:
            p,q=table[x,u],table[y,u]
            if (p is None)!=(q is None):return w+(u,)
            if p is None:continue
            if p[0]!=q[0]:return w+(u,)
            if chart(p[1])!=chart(q[1]):return w+(u,)
        for u in labels:
            p,q=table[x,u],table[y,u]
            if p is not None and q is not None:
                pair=(p[1],q[1])
                if pair not in seen:seen.add(pair);queue.append((*pair,w+(u,)))
    return None


def main():
    states=tuple(product((-1,0,1),repeat=6))
    # q is a support chart. A swap exposes the second triad. The centered
    # GELU even response has exactly the support fibres on ternary scalars.
    chart=lambda z:tuple(v*v for v in z[:3])
    def step(z,u):
        d,r=z[:3],z[3:]
        if u=='swap':return chart(z),r+d
        if u=='center':return chart(z),z
        if u=='shift':return d,z # exact injective label of shifted even GELU response, proven below
        if u=='guard':return (r[0]>0),z # separate abstract guard control
        raise ValueError(u)
    def count(h):return [len(set(p.values())) for p in h]
    h,t=refine(states,('center','swap'),chart,step);assert count(h)==[8,64]
    h,t=refine(states,('center','swap','shift'),chart,step);assert count(h)==[8,216,729]
    a=(1,0,0,1,-1,0);b=(1,0,0,-1,1,0)
    assert witness(a,b,('center','swap','shift'),chart,t)==('swap','shift')
    h,t=refine(states,('center','guard'),chart,step);assert count(h)==[8,16]
    assert witness(a,b,('center','guard'),chart,t)==('guard',)
    # Distinguishing admissibility alone, not a fabricated observation.
    def partial(z,u):return (0,z) if z[3]>0 else None
    h,t=refine(states,('admit',),chart,partial);assert count(h)==[8,16]
    assert witness(a,b,('admit',),chart,t)==('admit',)
    try:refine([a],('bad',),chart,lambda z,u:(0,b))
    except ValueError:pass
    else:raise AssertionError('nonclosed domain admitted')
    # Source-legal single-history variant: fixed swap transition on both
    # modes, centered or shifted even GELU observer. Exact label classes
    # are justified by strict monotonicity of E(t) for t>=0.
    def source_labels(z,u):
        d,r=z[:3],z[3:]
        return (chart(z) if u=='center' else d),r+d
    h,t=refine(states,('center',),chart,source_labels);assert count(h)==[8,64]
    h,t=refine(states,('center','shift'),chart,source_labels);assert count(h)==[8,216,729]
    assert witness(a,b,('center','shift'),chart,t)==('center','shift')
    # Local nonlinear observation: squares blind at zero to first derivatives,
    # but shifted squares expose signs; no Jacobian-only certificate at zero.
    d=s.Matrix(s.symbols('d:3'));r=s.Matrix(s.symbols('r:3'));x=s.Matrix.vstack(d,r)
    F=s.Matrix([v*v for v in d]);G=s.Matrix([v*v for v in r]);H=s.Matrix([(v+1)**2 for v in r])
    assert F.col_join(G).jacobian(x).subs(dict.fromkeys(x,0)).rank()==0
    assert F.col_join(G).jacobian(x).subs(dict.fromkeys(x,1)).rank()==6
    assert G.col_join(H).jacobian(x).rank()==3
    # Exact GELU shifted mean E(d+1) strictly ordered on d=-1,0,1.
    E=lambda t:s.Rational(1,2)*t*s.erf(t/s.sqrt(2))
    vals=[E(t+1) for t in (-1,0,1)]
    assert vals[0]==0 and vals[1].is_positive and (vals[2]-vals[1]).evalf()>0
    t=s.symbols('t',real=True);assert s.diff(E(t),t).subs(t,0)==0
    assert s.diff(E(t),t).subs(t,1).is_positive
    print('PASS',s.__version__,'729 states; support 8->64; signed 8->216->729; guard/admissibility 8->16; shortest witnesses; nonlinear Jacobian controls')

if __name__=='__main__':main()
