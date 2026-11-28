"""Source-owned X369 future-fibre audit across the full consumer portfolio.

Exact input state keys and returned observations belong to existing
x369_source_future_acceptance.native_future_contract. No label oracle,
training, edited recurrence, or inferred finite neural universe.
Original Leech-LILA: Anatolii Kornienko; DASHI extension.
"""
from collections import deque
from time import monotonic
from hashlib import sha256
import json

RAW_CONSUMERS = ('logits_0', 'logits_1', 'logits_2',
                 'output', 'ids', 'weights', 'local_signs')
TYPED_CONSUMERS = ('logits_0', 'logits_1', 'logits_2')


def fibre_audit(seeds, controls, step, charts, consumers, *, budget=1000,
                seconds=90):
    """One source graph, many independently scoped candidate projections.

    step(state, control) -> (named observation pairs, next BYTE state).
    A quotient is refuted as soon as equal current chart values yield
    unequal original observations or unequal successor chart values.
    Successful FINITE closure is not a theorem on all native float states.
    """
    seeds=tuple(dict.fromkeys(seeds));controls=tuple(controls)
    charts=dict(charts);consumers=tuple(consumers)
    if (not seeds or not controls or len(set(controls))!=len(controls)
        or not charts or not consumers or len(set(consumers))!=len(consumers)
        or budget<len(seeds) or not 0<seconds<=90):
        raise ValueError('invalid source audit scope')
    if any(not callable(f) for f in charts.values()):
        raise ValueError('all charts must be executable')
    seen={};queue=deque();edges={};values={};collisions={}
    fibre_sizes={name:{} for name in charts}
    compared={name:{} for name in charts}
    for i,h in enumerate(seeds):
        seen[h]=(i,());queue.append(h)
    deadline=monotonic()+seconds
    def chart_value(h,name):
        key=(h,name)
        if key not in values:values[key]=charts[name](h)
        return values[key]
    def path(h):
        i,word=seen[h]
        return dict(seed=i,word=list(word))
    def identity(h):
        if isinstance(h,bytes):
            return dict(kind='original_state_bytes',sha256=sha256(h).hexdigest(),length=len(h))
        return dict(kind='synthetic_fixture',sha256=sha256(repr(h).encode()).hexdigest())
    def fingerprint(value):
        # Exact, untruncated, stable within the declared Python source
        # contract. Native observations contain immutable typed byte tuples.
        return sha256(repr(value).encode()).hexdigest()
    def receipt(name,consumer,h,k,u,kind,old,new):
        return dict(chart=name,consumer=consumer,left=path(h),
                    right=path(k),control=u,kind=kind,
                    left_state=identity(h),right_state=identity(k),
                    left_value=repr(old)[:250],right_value=repr(new)[:250],
                    left_value_sha256=fingerprint(old),
                    right_value_sha256=fingerprint(new))
    def compare():
        """Compare ORIGINAL completed edges per (chart value, control).

        Grouping after control selection also detects bounded-prefix negatives
        when the earliest chart representative has no completed edge.
        """
        for chart_name in charts:
            group={}
            for h in seen:
                value=chart_value(h,chart_name)
                try:
                    fibre_sizes[chart_name][value]=fibre_sizes[chart_name].get(value,0)+1
                except TypeError as exc:
                    raise ValueError('chart values must be hashable') from exc
                for u in controls:
                    edge=edges.get((h,u))
                    if edge is None:continue
                    key=(value,u)
                    prior=group.setdefault(key,h)
                    if prior==h:continue
                    compared[chart_name][u]=compared[chart_name].get(u,0)+1
                    obs0,n0=edges[prior,u];obs1,n1=edge
                    for consumer in consumers:
                        if consumer not in obs0 or consumer not in obs1:
                            raise ValueError('consumer absent from original source edge')
                        if obs0[consumer]!=obs1[consumer]:
                            collisions.setdefault((chart_name,consumer),
                                receipt(chart_name,consumer,prior,h,u,
                                        'observation',obs0[consumer],obs1[consumer]))
                    v0=chart_value(n0,chart_name)
                    v1=chart_value(n1,chart_name)
                    if v0!=v1:
                        collisions.setdefault((chart_name,'__transition__'),
                            receipt(chart_name,'__transition__',prior,h,u,
                                    'successor_chart',v0,v1))
    status='closed'
    while queue:
        if monotonic()>=deadline:
            status='time_budget';break
        h=queue.popleft()
        for u in controls:
            if monotonic()>=deadline:
                status='time_budget';break
            obs,n=step(h,u)
            if not isinstance(obs,(tuple,list,dict)):
                raise ValueError('named original observations required')
            named=dict(obs)
            if len(named)!=len(obs) or any(k not in named for k in consumers):
                raise ValueError('duplicate/missing source consumer')
            if (h,u) in edges and edges[h,u]!=(named,n):
                raise RuntimeError('source transition nonrepeatable')
            edges[h,u]=(named,n)
            if n not in seen:
                if len(seen)>=budget:
                    status='state_budget';break
                start,word=seen[h];seen[n]=(start,word+(u,));queue.append(n)
        if status!='closed':break
    # Crucially: negative evidence from a bounded prefix remains valid
    # even when the full original neural reachable graph is not closed.
    compare()
    statuses={}
    for name in charts:
        bad={consumer:r for (q,consumer),r in collisions.items() if q==name}
        closed=(status=='closed' and len(edges)==len(seen)*len(controls))
        per_consumer={}
        for consumer in (*consumers,'__transition__'):
            per_consumer[consumer]=dict(
                status='refuted' if consumer in bad else 'finite_closed' if closed else 'incomplete',
                witness=bad.get(consumer),total_checked_edges=len(edges),
                compared_same_chart_control_pairs=sum(compared[name].values()),
                unrestricted_float_source_certified=False)
        statuses[name]=dict(
            status='refuted' if bad else 'finite_closed' if closed else 'incomplete',
            witnessed_failures=bad,
            compatibility_matrix=per_consumer,
            distinct_fibres=len(fibre_sizes[name]),
            largest_fibre=max(fibre_sizes[name].values(),default=0),
            all_original_consumers_stable_on_explored_graph=(not bad and closed),
            unrestricted_float_source_certified=False)
    return dict(graph_status=status,seen_states=len(seen),checked_edges=len(edges),
                required_edges=len(seen)*len(controls),charts=statuses,
                source_owned_observations=True,original_checkpoint_authenticated=False,
                native_precision_certified=False,unrestricted_source_certified=False,
                agda_kernel_checked=False)


def replay_witness(witness,seeds,step,charts):
    """BIDI re-run from original seed IDs and words, never from saved labels.

    The original step is invoked anew; a forged or stale witness fails.
    No universal conclusion is inferred from successful counterexample replay.
    """
    seeds=tuple(seeds)
    chart=charts[witness['chart']]
    def reach(p):
        i=p['seed']
        if type(i) is not int or not 0<=i<len(seeds):
            raise ValueError('invalid original seed witness')
        h=seeds[i]
        for u in p['word']:
            _,h=step(h,u)
        return h
    a,b=reach(witness['left']),reach(witness['right'])
    if isinstance(a,bytes):
        check=lambda h:dict(kind='original_state_bytes',sha256=sha256(h).hexdigest(),length=len(h))
    else:
        check=lambda h:dict(kind='synthetic_fixture',sha256=sha256(repr(h).encode()).hexdigest())
    if check(a)!=witness['left_state'] or check(b)!=witness['right_state']:
        raise ValueError('source-state identity mismatch on replay')
    if chart(a)!=chart(b):
        raise ValueError('forged chart-fibre collision')
    obs_a,next_a=step(a,witness['control'])
    obs_b,next_b=step(b,witness['control'])
    if witness['kind']=='observation':
        aa,bb=dict(obs_a),dict(obs_b)
        name=witness['consumer']
        if name not in aa or name not in bb or aa[name]==bb[name]:
            raise ValueError('source observation separator failed replay')
        lhs,rhs=aa[name],bb[name]
    elif witness['kind']=='successor_chart':
        if chart(next_a)==chart(next_b):
            raise ValueError('source successor separator failed replay')
        lhs,rhs=chart(next_a),chart(next_b)
    else:raise ValueError('unknown witness kind')
    if sha256(repr(lhs).encode()).hexdigest()!=witness['left_value_sha256'] or (
        sha256(repr(rhs).encode()).hexdigest()!=witness['right_value_sha256']):
        raise ValueError('source observer/successor identity changed on replay')
    return dict(replayed=True,chart=witness['chart'],
                consumer=witness['consumer'],kind=witness['kind'],
                source_step_reexecuted=True)


def native_audit(model,controls,*,budget=1000,seconds=90):
    """Actual eval gfx803 native source, no artificial task labels.

    Caller MUST separately use x369_source_acceptance to authenticate the
    immutable source/checkpoint; this read-only audit does not do so.
    """
    from scripts.x369_source_future_acceptance import native_future_contract
    consumer='typed' if model.typed_consumer else 'raw'
    seeds,step,profile=native_future_contract(model,consumer)
    cache={}
    def p(h):
        if h not in cache:cache[h]=profile(h)
        return cache[h]
    charts={
        'constructed_semantic_six':lambda h:p(h)[0],
        'semantic_plus_cap_profile':p,
        'full_original_history_bytes':lambda h:h}
    consumers=TYPED_CONSUMERS if model.typed_consumer else RAW_CONSUMERS
    out=fibre_audit(seeds,controls,step,charts,consumers,
                    budget=budget,seconds=seconds)
    replays={}
    for chart_name,entry in out['charts'].items():
        for consumer_name,w in entry['witnessed_failures'].items():
            replays[chart_name,consumer_name]=replay_witness(w,seeds,step,charts)
    out['independent_source_separator_replays']={
        f'{chart}:{consumer}':v
        for (chart,consumer),v in replays.items()}
    out.update(model_consumer=consumer,source_name=type(model).__name__,
               seed_count=len(seeds),source_state_kind='original raw history bytes',
               source_chart_kind='actual rational decoded history/cap profile',
               original_checkpoint_authenticated=False)
    return out


def certificate():
    """Synthetic adverse controls; no GPU or neural-source acceptance."""
    controls=('tick',)
    def graph_step(s,u):
        return (('logits_0',s%2),('ids',s)),(s+1)%4
    charts={'parity':lambda s:s%2,'full':lambda s:s}
    out=fibre_audit((0,),controls,graph_step,charts,
                    ('logits_0','ids'),budget=4)
    assert out['graph_status']=='closed' and out['checked_edges']==4
    assert out['charts']['parity']['status']=='refuted'
    assert out['charts']['parity']['witnessed_failures']['ids']['kind']=='observation'
    assert out['charts']['full']['status']=='finite_closed'
    b=fibre_audit((0,),controls,graph_step,charts,
                  ('logits_0','ids'),budget=2)
    assert b['graph_status']=='state_budget' and b['charts']['parity']['status']=='incomplete'
    def unstable(s,u):
        return (('x',0),), (1 if s==0 else 2 if s==2 else (s+1)%4)
    c=fibre_audit((0,2),controls,unstable,{'coarse':lambda s:s%2},('x',),budget=4)
    assert c['charts']['coarse']['status']=='refuted'
    assert '__transition__' in c['charts']['coarse']['witnessed_failures']
    return dict(synthetic_closed=out['checked_edges'],negative_consumer=True,
                negative_transition=True,budget_refusal=True,
                original_native_tested=False,optimizer_steps=0)


if __name__=='__main__':
    print(json.dumps(certificate(),sort_keys=True))
