"""Exact finite consumer compiler; no neural or hardware assumptions.
Original Leech-LILA: Anatolii Kornienko; DASHI extension.
"""
from collections import deque
from itertools import product
from scripts.check_x369_nonlinear_closure import refine,witness
from scripts.x369_future_realization import compile_realization,verify_realization,quotient_step,unpack

def compile_observers(seeds,controls,step,observe,*,budget=100000):
    """Finite reachable product, including admission as an observed result.
    Budget exhaustion is a refusal, not a positive quotient certificate.
    All callbacks must be deterministic and hashable.
    """
    seeds=tuple(dict.fromkeys(seeds));controls=tuple(controls)
    if not seeds or not controls or len(set(controls))!=len(controls) or budget<len(seeds):
        raise ValueError('invalid scope')
    seen=dict.fromkeys(seeds);queue=deque(seeds);table={}
    # A rejected action leaves state unchanged and emits a distinct admission tag.
    def edge(h,u):
        v=step(h,u)
        return (('refused',),h) if v is None else (('admitted',v[0]),v[1])
    while queue:
        h=queue.popleft()
        for u in controls:
            o,n=edge(h,u);table[h,u]=(o,n)
            if n not in seen:
                if len(seen)>=budget:raise ValueError('state budget; no closure')
                seen[n]=None;queue.append(n)
    states=tuple(seen);chart={h:observe(h) for h in states}
    ps,_=refine(states,controls,chart.__getitem__,lambda h,u:table[h,u])
    r=compile_realization(states,seeds,controls,chart,table,ps[-1])
    v=verify_realization(r,seeds,controls,observe,edge)
    k=v['classes'];a=len(controls)
    # Class identifiers alone; table, observation payload, and codebook charged separately.
    bits=(k-1).bit_length()
    return r,dict(**v,partition_counts=[len(set(p.values())) for p in ps],
                  state_id_bits=bits,transition_index_bits=k*a*bits,
                  table_entries=k*a,observation_payload_cost_excluded=True,
                  construction_edges=len(table),physical_cost_certified=False)

def product_observers(observers):
    """Join declared observers BEFORE minimization; do not multiply minimal sizes."""
    observers=tuple(observers)
    if not observers:raise ValueError('empty portfolio')
    seeds=tuple(product(*(o[0] for o in observers)))
    def step(h,u):
        v=tuple(o[1](s,u) for o,s in zip(observers,h))
        if any(x is None for x in v):return None
        return tuple(x[0] for x in v),tuple(x[1] for x in v)
    def chart(h):return tuple(o[2](s) for o,s in zip(observers,h))
    return seeds,step,chart

def execute(receipt,seed_index,word):
    code=receipt['class_of'][receipt['seeds'][seed_index]]
    trace=[unpack(receipt['rows'][code]['chart'])]
    for u in word:
        obs,code=quotient_step(receipt,code,u)
        trace.append((obs,unpack(receipt['rows'][code]['chart'])))
    return code,tuple(trace)

def certificate():
    controls=('a','b','j','n')
    latch=lambda r,u:r if r is not None else u if u in ('a','b') else None
    step=lambda r,u:(r,latch(r,u))
    r,m=compile_observers((None,),controls,step,lambda r:r)
    assert m['classes']==3 and m['state_id_bits']==2
    for suffix in product(controls,repeat=4):
        a,_=execute(r,0,('a','b')+suffix);b,_=execute(r,0,('b','a')+suffix)
        assert a!=b
    # Joint first and parity observer: six reachable classes, not independent histories.
    seeds,advance,chart=product_observers((((None,),step,lambda r:r),((0,),lambda p,u:(p,p^1),lambda p:p)))
    joint,j=compile_observers(seeds,controls,advance,chart)
    assert j['classes']==6
    # Hidden counter with delayed separation, and explicit admission boundary.
    d,q=compile_observers((0,),('tick',),lambda h,u:(h==3,(h+1)%4),lambda h:0)
    assert q['classes']==4
    _,p=compile_observers((0,1),('go',),lambda h,u:None if h==0 else (0,h),lambda h:0)
    assert p['classes']==2
    try:compile_observers((0,),('tick',),lambda h,u:(0,h+1),lambda h:0,budget=8)
    except ValueError:pass
    else:raise AssertionError('unbounded graph admitted')
    return dict(latch=m,joint=j,delayed=q,admission=p,optimizer_steps=0)
if __name__=='__main__':
    import json
    print(json.dumps(certificate(),sort_keys=True))
