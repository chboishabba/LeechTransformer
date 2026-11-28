"""Concrete finite section/quotient weld to the existing refinement owner.
Original Leech-LILA: Anatolii Kornienko; DASHI extension.
A certificate is an exact finite receipt, not a compression cost claim.
"""
from collections import deque
from fractions import Fraction
from scripts.check_x369_nonlinear_closure import refine


def pack(x):
    """Lossless JSON representation of the declared audit value types."""
    if type(x) is bytes:return ['bytes',x.hex()]
    if type(x) is Fraction:return ['rational',x.numerator,x.denominator]
    if type(x) is tuple:return ['tuple',[pack(v) for v in x]]
    if type(x) in (int,bool,str):return [type(x).__name__,x]
    if x is None:return ['none']
    raise TypeError('unsupported certificate value: '+type(x).__name__)


def unpack(v):
    if not isinstance(v,list) or not v:raise ValueError('invalid packed value')
    tag=v[0]
    if tag=='bytes' and len(v)==2:return bytes.fromhex(v[1])
    if tag=='rational' and len(v)==3:return Fraction(v[1],v[2])
    if tag=='tuple' and len(v)==2:return tuple(unpack(x) for x in v[1])
    if tag in ('int','bool','str') and len(v)==2 and type(v[1]).__name__==tag:return v[1]
    if v==['none']:return None
    raise ValueError('invalid packed value')


def compile_realization(states,seeds,controls,codes,table,classes):
    """Use the stable classes returned by the existing finite refiner.
    Verify against the complete original graph before returning the receipt.
    Only quotient rows are needed for later quotient execution; fine rows and
    section are verification/provenance data, not part of the runtime code.
    """
    states=tuple(states);controls=tuple(controls);seeds=tuple(seeds)
    idx={h:i for i,h in enumerate(states)};cl=[classes[h] for h in states]
    k=len(set(cl));section=[cl.index(i) for i in range(k)]
    rows=[]
    for i in section:
        h=states[i]
        rows.append(dict(chart=pack(codes[h]),edges=[(pack(table[h,u][0]),classes[table[h,u][1]]) for u in controls]))
    r=dict(schema=1,controls=[pack(u) for u in controls],states=[pack(h) for h in states],
           seeds=[idx[h] for h in seeds],class_of=cl,section=section,rows=rows,
           fine_successors=[[idx[table[h,u][1]] for u in controls] for h in states])
    verify_realization(r,seeds,controls,codes.__getitem__,lambda h,u:table[h,u])
    return r


def verify_realization(r,seeds,controls,chart,step):
    """Check BIDI section, exact observations, all commuting edges, reachability
    and minimality. A caller must supply independently admitted source callbacks
    for native replay; passing recorded table callbacks checks table consistency.
    """
    def require(ok,msg):
        if not ok:raise ValueError(msg)
    def index(i,n):return type(i) is int and 0<=i<n
    require(r['schema']==1,'schema')
    states=tuple(map(unpack,r['states']));controls=tuple(controls);seeds=tuple(dict.fromkeys(seeds))
    n=len(states);rows=r['rows'];k=len(rows);cl=r['class_of'];sec=r['section'];fine=r['fine_successors'];a=len(controls)
    require(n and k and a and seeds and len(set(states))==n and len(set(controls))==a,'carrier')
    require(tuple(map(unpack,r['controls']))==controls,'control identity')
    require(len(cl)==n and all(index(i,k) for i in cl) and set(cl)==set(range(k)),'class map')
    require(len(sec)==k and all(index(i,n) for i in sec) and all(cl[i]==j for j,i in enumerate(sec)),'section')
    require(len(fine)==n and all(len(e)==a and all(index(i,n) for i in e) for e in fine),'fine edges')
    require(all(index(i,n) for i in r['seeds']) and tuple(states[i] for i in r['seeds'])==seeds,'seed identity')
    qchart=tuple(unpack(row['chart']) for row in rows);qedge=[]
    for row in rows:
        require(len(row['edges'])==a and all(index(j,k) for _,j in row['edges']),'quotient edges')
        qedge.append(tuple((unpack(o),j) for o,j in row['edges']))
    for i,h in enumerate(states):
        require(chart(h)==qchart[cl[i]],'chart descent')
        for j,u in enumerate(controls):
            obs,nxt=step(h,u);target=fine[i][j]
            require(nxt==states[target],'source successor')
            require((obs,cl[target])==qedge[cl[i]][j],'consumer/transition descent')
    visited=set(r['seeds']);queue=deque(visited)
    while queue:
        for j in fine[queue.popleft()]:
            if j not in visited:visited.add(j);queue.append(j)
    require(len(visited)==n,'unreachable states')
    ui={u:i for i,u in enumerate(controls)}
    ps,_=refine(range(k),controls,qchart.__getitem__,lambda j,u:qedge[j][ui[u]])
    require(len(set(ps[-1].values()))==k,'nonminimal quotient')
    return dict(states=n,classes=k,edges=n*a,section_roundtrips=k,minimal=True)


def quotient_step(r,code,control):
    """Execute the verified quotient without reconstructing raw history."""
    if type(code) is not int or not 0<=code<len(r['rows']):raise ValueError('invalid code')
    j=tuple(map(unpack,r['controls'])).index(control)
    obs,n=r['rows'][code]['edges'][j]
    return unpack(obs),n


def replay_separator(receipt,seeds,controls,chart,step):
    """Replay a saved path pair against an admitted source, even without closure.
    Does not prove globally shortest word or authenticate source callbacks.
    """
    seeds=tuple(dict.fromkeys(seeds));controls=tuple(controls)
    def control(u):
        def restore(v):return tuple(map(restore,v)) if type(v) is list else v
        u=restore(u)
        if u not in controls:raise ValueError('invalid source control')
        return u
    def run_path(p):
        i=p['seed']
        if type(i) is not int or not 0<=i<len(seeds):raise ValueError('seed index')
        h=seeds[i]
        for u in p['word']:
            _,h=step(h,control(u))
        return h
    a,b=run_path(receipt['left']),run_path(receipt['right'])
    if a==b or chart(a)!=chart(b):raise ValueError('not a same-chart distinct-state pair')
    initial=(pack(a),pack(b));fields=set();first=None;trace=[]
    for depth,u in enumerate(receipt['word'],1):
        u=control(u)
        A,a=step(a,u);B,b=step(b,u);left,right=chart(a),chart(b);diff=[]
        if A!=B:diff.append(('consumer',pack(A),pack(B)))
        if left!=right:diff.append(('successor_chart',pack(left),pack(right)))
        if diff:
            fields.update(x[0] for x in diff)
            if first is None:first=depth
            trace.append(dict(depth=depth,differences=diff))
    if not fields or sorted(fields)!=sorted(receipt['fields']):raise ValueError('separator replay mismatch')
    return dict(initial_states=initial,fields=sorted(fields),first_distinguishing_depth=first,differences=trace)


def certificate():
    import json,copy
    states=(b'\x00',b'\x80',b'\x01',b'\x81');seeds=states[:2];controls=('tick','hold')
    chart=lambda h:Fraction(h[0]&1,3)
    def step(h,u):
        nxt=bytes([h[0]^int(u=='tick')]);return (('value',chart(h)),nxt)
    ps,t=refine(states,controls,chart,step);codes={h:chart(h) for h in states}
    r=compile_realization(states,seeds,controls,codes,t,ps[-1]);r=json.loads(json.dumps(r))
    v=verify_realization(r,seeds,controls,chart,step);assert v==dict(states=4,classes=2,edges=8,section_roundtrips=2,minimal=True)
    for i,h in enumerate(states):
        for u in controls:
            obs,n=step(h,u);assert quotient_step(r,r['class_of'][i],u)==(obs,r['class_of'][states.index(n)])
    cases=[]
    for kind in ('section','successor','observation','unreachable','nonminimal'):
        b=copy.deepcopy(r)
        if kind=='section':b['section'][0]=b['section'][1]
        elif kind=='successor':b['fine_successors'][0][0]=0
        elif kind=='observation':b['rows'][0]['edges'][0][0]=pack(('forged',))
        elif kind=='unreachable':b['seeds']=[0]
        else:
            b['class_of']=list(range(4));b['section']=list(range(4));b['rows']=[dict(chart=pack(chart(h)),edges=[(pack(step(h,u)[0]),states.index(step(h,u)[1])) for u in controls]) for h in states]
        try:verify_realization(b,seeds if kind!='unreachable' else seeds[:1],controls,chart,step)
        except ValueError:cases.append(kind)
        else:raise AssertionError('invalid receipt accepted: '+kind)
    assert unpack(pack((b'\x00',b'\x80',Fraction(-2,3),True,0,None)))==(b'\x00',b'\x80',Fraction(-2,3),True,0,None)
    # An independent delayed replay checks the path and exact observed mismatch.
    from scripts.x369_source_future_acceptance import future_acceptance
    delayed=lambda h,u:(h==3,(h+2)%4)
    a=future_acceptance(range(4),('tick',),delayed,lambda h:0,budget=4)
    w=a['shortest_pair_separator'];replayed=replay_separator(w,range(4),('tick',),lambda h:0,delayed)
    assert replayed['first_distinguishing_depth']==2
    bad=copy.deepcopy(w);bad['word']=['tick']
    try:replay_separator(bad,range(4),('tick',),lambda h:0,delayed)
    except ValueError:pass
    else:raise AssertionError('forged shorter separator accepted')
    a=future_acceptance(range(4),((0,0),),delayed,lambda h:0,budget=4)
    w=json.loads(json.dumps(a['shortest_pair_separator']))
    assert replay_separator(w,range(4),((0,0),),lambda h:0,delayed)['first_distinguishing_depth']==2
    return dict(**v,rejected=cases,json_roundtrip=True,separator_depth=2,forged_separator_rejected=True,tuple_control_json_replay=True)


if __name__=='__main__':print(certificate())
