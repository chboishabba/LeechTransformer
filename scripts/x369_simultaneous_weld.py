"""One original graph for semantic decoding, residuals and future consumers.
Original Leech-LILA: Anatolii Kornienko; DASHI extension.
Exact finite stored-value contract, not native execution in this test.
"""
from fractions import Fraction as Q
from scripts.x369_source_future_acceptance import future_acceptance
from scripts.x369_exact_semantic_decoder import fit,decode
from scripts.check_x369_nonlinear_closure import refine
from scripts.x369_future_realization import compile_realization,verify_realization,pack,unpack


def literal_matrix(u):
    """Independent task endpoint on basis vectors; no learned action assumed."""
    from scripts.audit_x369_ordered_signed_task import endpoint
    g,t=u
    if type(g) is not int or type(t) is not int or t not in range(4):raise ValueError('invalid literal')
    choices=({0:0,8:1,4:2} if t<2 else {0:0,10:1} if t==2 else {0:0,1:1})
    if g not in choices:raise ValueError('not an original literal control')
    guards=tuple(choices[g] if i==t else 0 for i in range(4));cols=[]
    for j in range(6):
        z=tuple(int(i==j) for i in range(6));m,d,_=endpoint('abjn'[t],guards,z[:3],z[3:]);cols.append(m+d)
    return tuple(zip(*cols))


def apply_matrix(A,z):return tuple(sum((a*v for a,v in zip(row,z)),Q(0)) for row in A)


def simultaneous(labelled_seeds,controls,step,chart,history,action,*,budget,seconds=90,export=False,constructed=None,score=None):
    """Reuse the existing BFS/refiner. Only one source observation per edge.
    Targets flow into a separate receipt, never the actual step or state key.
    A closed compatible graph admits a synthesized decoder/table, not a
    train-only generalization claim. Incomplete graphs admit negatives only.
    """
    seeds=tuple(labelled_seeds);controls=tuple(controls)
    if not seeds or len({i for _,_,i in seeds})!=len(seeds):raise ValueError('unique seed IDs required')
    labels={};paths={};table={};collision=None;defect=None;defects=0;seed_defects=0;readout={}
    def remember(h,z,p):
        nonlocal collision
        z=tuple(z)
        if len(z)!=6:raise ValueError('six signed coordinates required')
        if h in labels:
            if labels[h]!=z and collision is None:
                collision=dict(state=pack(h),left=paths[h],right=p,labels=(pack(labels[h]),pack(z)))
        else:labels[h]=z;paths[h]=p
    for h,z,i in seeds:
        remember(h,z,dict(seed=i,word=[]))
        if constructed is not None:seed_defects+=int(tuple(constructed(h))!=tuple(z))
    def observed(h,u):
        nonlocal defect,defects
        obs,n=step(h,u);table[h,u]=(obs,n);z=tuple(action(u,labels[h]));p=dict(seed=paths[h]['seed'],word=paths[h]['word']+[u]);remember(n,z,p)
        if constructed is not None:
            a=tuple(action(u,constructed(h)));b=tuple(constructed(n));delta=tuple(y-x for x,y in zip(a,b))
            if any(delta):
                defects+=1
                if defect is None:defect=dict(before=paths[h],control=u,after=p,expected=pack(a),actual=pack(b),signed_defect=pack(delta))
        if score is not None:
            for key,status in score(obs,z):
                counts=readout.setdefault(key,{});counts[status]=counts.get(status,0)+1
        return obs,n
    g=future_acceptance((h for h,_,_ in seeds),controls,observed,chart,budget=budget,seconds=seconds)
    closed=g['status']=='closed';r=dict(future=g,observed_edges=len(table),labelled_states=len(labels),semantic_collision=collision,
        semantic_table_certified=False,linear_decoder_certified=False,universal_full_domain=False,training_steps=0,
        constructed=dict(seed_mismatches=seed_defects,edge_defects=defects,first_defect=defect,checked=constructed is not None),
        readout_path_counts=readout,readout_scope='first retained seed path per source state; no correctness promotion on collision or incomplete graph')
    if collision is not None:return r
    states=tuple(labels)
    records=[(i,history(h),labels[h]) for i,h in enumerate(states)]
    f=fit(records,include_family=closed);r['decoder_fit']=f
    if not f['consistent']:
        ids=set(f.get('witnesses',()))|{i for i,_,_,_ in f.get('conflict',())}
        f['witness_records']=[dict(id=i,path=paths[states[i]],state=pack(states[i]),history=pack(records[i][1]),label=pack(records[i][2])) for i in sorted(ids)]
    elif not closed:
        # Do not emit a sampled-only decoder as an all-word realization.
        f.pop('coefficients',None)
    if not closed:return r
    assert len(table)==len(states)*len(controls)==g['edges']
    for h,z,_ in seeds:assert labels[h]==tuple(z)
    for (h,u),(_,n) in table.items():assert labels[n]==tuple(action(u,labels[h]))
    r['semantic_table_certified']=True
    r['constructed_exact_on_closed_graph']=constructed is not None and not seed_defects and not defects
    r['readout_unique_correct_on_closed_graph']=score is not None and bool(readout) and all(v=={'correct_unique':len(table)} for v in readout.values())
    if f['consistent']:
        c=tuple(map(Q,f['coefficients']));decoded={h:decode(hist,c) for h,(_,hist,_) in zip(states,records)}
        assert decoded==labels
        assert all(decoded[n]==tuple(action(u,decoded[h])) for (h,u),(_,n) in table.items())
        r['linear_decoder_certified']=True
        f['scope']='synthesized on entire closed graph; not heldout-selected train-only decoder'
    # Retain the declared candidate chart AND independent semantic labels.
    codes={h:(chart(h),labels[h]) for h in states}
    ps,_=refine(states,controls,codes.__getitem__,lambda h,u:table[h,u]);cl=ps[-1]
    r['joint_partition_counts']=[len(set(p.values())) for p in ps]
    r['joint_classes']=len(set(cl.values()));r['joint_edges']=len(table)
    r['joint_scope']='minimal for declared chart, independent semantic label and edge consumer on this closed graph'
    if export:r['joint_realization']=compile_realization(states,tuple(dict.fromkeys(h for h,_,_ in seeds)),controls,codes,table,cl)
    return r


def verify_joint_realization(r,labelled_seeds,controls,chart,step,action):
    """Reopen actual source and independent seed/action authority, not labels
    supplied by the saved table alone. Caller binds checkpoint/runtime identity.
    """
    seeds=tuple(labelled_seeds);controls=tuple(controls)
    states=tuple(map(unpack,r['states']));cl=r['class_of'];rows=r['rows']
    if len(cl)!=len(states) or any(type(i) is not int or not 0<=i<len(rows) for i in cl):raise ValueError('class map')
    labels={h:unpack(rows[i]['chart'])[1] for h,i in zip(states,cl)}
    for h,z,_ in seeds:
        if labels.get(h)!=tuple(z):raise ValueError('semantic seed mismatch')
    # Existing verifier checks reachability, chart/consumer and every original
    # edge. The wrapper checks semantic transport on those very same edges.
    def observed(h,u):
        obs,n=step(h,u)
        if n not in labels or labels[n]!=tuple(action(u,labels[h])):raise ValueError('semantic edge mismatch')
        return obs,n
    return verify_realization(r,tuple(dict.fromkeys(h for h,_,_ in seeds)),controls,lambda h:(chart(h),labels[h]),observed)


def native_simultaneous(model,controls,*,budget,seconds=90,export=False):
    import torch
    from itertools import product
    from scripts.x369_source_future_acceptance import native_future_contract
    from scripts.x369_indexed_residual_closure import native_source
    from scripts.audit_x369_joint_action import MEANS
    from scripts.audit_x369_ordered_signed_task import TRIADS,gram
    from scripts.train_x369_ordered_action import LABELS
    controls=tuple(controls);matrices={u:literal_matrix(u) for u in controls}
    source_seeds,step,chart=native_future_contract(model,'raw');seed,_,native_action=native_source(model)
    seeds=tuple((seed(m,d),m+d,i) for i,(m,d) in enumerate(product(MEANS,TRIADS)))
    assert tuple(dict.fromkeys(h for h,_,_ in seeds))==source_seeds
    for _,z,_ in seeds:
        for u in controls:assert apply_matrix(matrices[u],z)==native_action(u,z)
    dt=next(model.parameters()).dtype;dims=(model.experts,2,model.width)
    def history(h):return tuple(tuple(tuple(Q(float(x)) for x in b) for b in e) for e in torch.frombuffer(bytearray(h),dtype=dt).reshape(dims).tolist())
    def score(obs,z):
        values=dict(obs);targets=(z[:3],gram(z[3:]),z[3:])
        for i,(L,y) in enumerate(zip(LABELS,targets)):
            name=f'logits_{i}';dtype,shape,raw=values[name]
            if dtype!=str(dt) or tuple(shape)!=(1,len(L)):raise ValueError('invalid head receipt')
            v=torch.frombuffer(bytearray(raw),dtype=dt).tolist();w=[j for j,x in enumerate(v) if x==max(v)];target=L.index(y)
            yield name,('correct_unique' if w==[target] else 'wrong_unique' if len(w)==1 else 'tie_target' if target in w else 'tie_without_target')
    return simultaneous(seeds,controls,step,chart,history,lambda u,z:apply_matrix(matrices[u],z),budget=budget,seconds=seconds,export=export,constructed=lambda h:chart(h)[0],score=score)


def certificate():
    from itertools import product
    from scripts.audit_x369_ordered_signed_task import endpoint
    from scripts.x369_future_realization import quotient_step,unpack
    zero=(0,0,0);m=(1,0,0);z=m+zero;neg=lambda z:tuple(-x for x in z)
    def h(y):
        M,D=y[:3],y[3:];return ((tuple(Q(a+b) for a,b in zip(M,D)),tuple(Q(a-b) for a,b in zip(M,D))),)
    act=lambda u,y:neg(y);seeds=((0,z,0),);ctrl=('n',)
    a=simultaneous(seeds,ctrl,lambda x,u:(0,1-x),lambda x:0,lambda x:h(z if x==0 else neg(z)),act,budget=2,export=True)
    assert a['semantic_table_certified'] and a['linear_decoder_certified'] and a['future']['partition_counts']==[1] and a['joint_classes']==2
    r=a['joint_realization'];_,n=quotient_step(r,r['class_of'][0],'n');assert unpack(r['rows'][n]['chart'])==(0,neg(z))
    import json,copy
    r=json.loads(json.dumps(r));assert verify_joint_realization(r,seeds,ctrl,lambda x:0,lambda x,u:(0,1-x),act)['edges']==2
    bad=copy.deepcopy(r);bad['rows'][n]['chart']=pack((0,z))
    try:verify_joint_realization(bad,seeds,ctrl,lambda x:0,lambda x,u:(0,1-x),act)
    except ValueError:pass
    else:raise AssertionError('forged semantic label accepted')
    # Byte/path collision rules out ANY deterministic semantic decoder.
    b=simultaneous(seeds,ctrl,lambda x,u:(0,x),lambda x:0,lambda x:h(z),act,budget=1)
    assert b['semantic_collision'] and not b['semantic_table_certified']
    # Distinct source histories can admit a semantic table but no linear decoder.
    c=simultaneous(seeds,ctrl,lambda x,u:(0,1-x),lambda x:0,lambda x:h(z if x==0 else tuple(2*v for v in z)),act,budget=2,constructed=lambda x:tuple((x+1)*v for v in z),score=lambda o,y:(('head','wrong_unique'),))
    assert c['constructed']['edge_defects']==2 and not c['constructed_exact_on_closed_graph'] and not c['readout_unique_correct_on_closed_graph']
    assert c['semantic_table_certified'] and not c['linear_decoder_certified'] and not c['decoder_fit']['consistent'] and c['decoder_fit']['witness_records']
    # Budget outside edge can still refute all linear decoders; no positive.
    d=simultaneous(seeds,ctrl,lambda x,u:(0,x+1),lambda x:0,lambda x:h(tuple((x+1)*v for v in z)),act,budget=1,export=True)
    assert d['future']['status']=='state_budget' and not d['decoder_fit']['consistent'] and 'joint_realization' not in d
    e=simultaneous(seeds,ctrl,lambda x,u:(0,x+1),lambda x:0,lambda x:h(z if x%2==0 else neg(z)),act,budget=1,export=True)
    assert e['decoder_fit']['consistent'] and 'coefficients' not in e['decoder_fit'] and not e['linear_decoder_certified']
    # Original endpoint agrees on all signed ternary labels; fractional
    # constructed residuals use linear extension, not integer floor division.
    count=0
    for t in range(4):
        for v in range(3 if t<2 else 2):
            g=4*((-v)%3) if t<2 else 10*v if t==2 else v;A=literal_matrix((g,t));guards=tuple(v if i==t else 0 for i in range(4))
            for y in product((-1,0,1),repeat=6):
                M,D,_=endpoint('abjn'[t],guards,y[:3],y[3:]);assert apply_matrix(A,y)==M+D;count+=1
            assert apply_matrix(A,tuple(Q(x,3) for x in z))==tuple(Q(x,3) for x in apply_matrix(A,z))
    print(dict(positive_closed=True,base_classes=1,joint_classes=2,byte_collision=True,nonlinear_table_only=True,bounded_negative=True,bounded_nonpromotion=True,literal_checks=count,joint_replay=True,forged_label_rejected=True,defect_checks=2,training_steps=0))


if __name__=='__main__':certificate()
