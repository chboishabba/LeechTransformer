"""Witness-first ORIGINAL recurrent-source consumer collision probe.

Source receipt 2026-11-28 found all 756 successors of 189 canonical
signed seeds were UNMARKED, while the breadth-first 1000-state graph
stopped before comparing two completed unmarked edges under the same
control. That graph does NOT establish consumer factorization.

Instead produce original depth-one successor histories from a small
number of canonical seeds/controls, group the unmarked successors in
the same proposed coarse fibre and immediately execute the SAME
original control on each. This yields a genuine one-step separating
witness without requiring global finite closure.

The original seed and each original path are stored, so replay starts
from the source-owned input bytes, never synthetic labels.
Original Leech-LILA: Anatolii Kornienko. DASHI extension.
"""
from hashlib import sha256
from itertools import combinations
from time import monotonic

def _hash(value):
    return sha256(repr(value).encode()).hexdigest()

def _state_id(h):
    if type(h) is not bytes:
        raise ValueError('actual original source history bytes required')
    return {'sha256':sha256(h).hexdigest(),'length':len(h)}

def _observe(step,h,u,fields):
    obs,n=step(h,u)
    named=dict(obs)
    if type(n) is not bytes or len(named)!=len(obs) or any(f not in named for f in fields):
        raise ValueError('source callback violated original named byte contract')
    return named,n

def _walk(seeds,step,original):
    h=seeds[original['seed']]
    for u in original['word']:
        _,h=step(h,u)
    return h

def witness_first_fibre_probe(seeds,controls,step,chart,fields,*,
                              seed_budget=4,seconds=90):
    seeds=tuple(seeds);controls=tuple(controls);fields=tuple(fields)
    if (not seeds or not controls or not fields or
        type(seed_budget) is not int or not 1<=seed_budget<=len(seeds) or
        not 0<seconds<=90 or any(type(s) is not bytes for s in seeds) or
        not callable(chart) or not callable(step)):
        raise ValueError('actual source original seeds, controls and chart required')
    if len(set(controls))!=len(controls) or len(set(fields))!=len(fields):
        raise ValueError('duplicate source controls or consumer names')
    deadline=monotonic()+seconds
    groups={};n_source_edges=0
    # Discover original depth-one successors only. Group by the
    # supplied actual source coarse chart, NOT inferred neural labels.
    for seed_index,h in enumerate(seeds[:seed_budget]):
        for first_control in controls:
            if monotonic()>=deadline:
                return _incomplete(groups,n_source_edges,'time_budget')
            _,s=_observe(step,h,first_control,fields)
            n_source_edges+=1
            c=chart(s)
            try:hash(c)
            except TypeError as exc:
                raise ValueError('original source chart must be hashable') from exc
            groups.setdefault(c,{})
            groups[c].setdefault(s,{'seed':seed_index,'word':[first_control]})
    # Cache each successor/control once, then compare across pairs.
    # This prioritizes same-fibre pairs missed by breadth-first search.
    comparisons=0;cache={}
    for coarse,items in groups.items():
        distinct=list(items)
        if len(distinct)<2:continue
        for u in controls:
            for h in distinct:
                if monotonic()>=deadline:
                    return _incomplete(groups,n_source_edges,'time_budget',comparisons)
                if (h,u) not in cache:
                    cache[h,u]=_observe(step,h,u,fields)
                    n_source_edges+=1
            base=distinct[0]
            obs0,n0=cache[base,u]
            for h in distinct[1:]:
                obs1,n1=cache[h,u];comparisons+=1
                differing=next((f for f in fields if obs0[f]!=obs1[f]),None)
                transition=chart(n0)!=chart(n1)
                if differing is None and not transition:continue
                kind='observation' if differing is not None else 'successor_chart'
                target=differing if differing is not None else '__transition__'
                left=obs0[differing] if differing else chart(n0)
                right=obs1[differing] if differing else chart(n1)
                witness={
                    'kind':kind,'consumer':target,'control':u,
                    'left':items[base],'right':items[h],
                    'left_state':_state_id(base),'right_state':_state_id(h),
                    'coarse_chart_fingerprint':_hash(coarse),
                    'left_value_sha256':_hash(left),
                    'right_value_sha256':_hash(right)}
                replay=independently_replay_witness(
                    witness,seeds,controls,step,chart,fields)
                if not replay['original_source_separator_replayed']:
                    raise AssertionError('original-source collision failed replay')
                return {
                    'status':'refuted','witness':witness,
                    'independent_replay':replay,'original_source_edges':n_source_edges,
                    'compared_same_fibre_source_pairs':comparisons,
                    'sampled_canonical_seed_count':seed_budget,
                    'source_fibre_count':len(groups),
                    'all_original_float_histories_covered':False,
                    'finite_closure_claim':False}
    return _incomplete(groups,n_source_edges,'sample_inconclusive',comparisons,
                       seed_budget)

def _incomplete(groups,edges,reason,comparisons=0,seed_budget=0):
    return {'status':reason,'witness':None,
            'original_source_edges':edges,
            'compared_same_fibre_source_pairs':comparisons,
            'source_fibre_count':len(groups),
            'sampled_canonical_seed_count':seed_budget,
            'all_original_float_histories_covered':False,
            'finite_closure_claim':False}

def independently_replay_witness(witness,seeds,controls,step,chart,fields):
    if witness['control'] not in controls:
        raise ValueError('invalid real source control')
    a=_walk(seeds,step,witness['left'])
    b=_walk(seeds,step,witness['right'])
    if _state_id(a)!=witness['left_state'] or _state_id(b)!=witness['right_state']:
        raise ValueError('source history changed')
    if chart(a)!=chart(b) or _hash(chart(a))!=witness['coarse_chart_fingerprint']:
        raise ValueError('source histories do not occupy same chart fibre')
    oa,na=_observe(step,a,witness['control'],fields)
    ob,nb=_observe(step,b,witness['control'],fields)
    if witness['kind']=='observation':
        k=witness['consumer']
        if k not in fields or oa[k]==ob[k]:
            raise ValueError('replayed source consumer not separated')
        left,right=oa[k],ob[k]
    elif witness['kind']=='successor_chart':
        left,right=chart(na),chart(nb)
        if left==right:raise ValueError('replayed original successor not separated')
    else:raise ValueError('unknown original source witness kind')
    if (_hash(left)!=witness['left_value_sha256'] or
        _hash(right)!=witness['right_value_sha256']):
        raise ValueError('original source changed during witness replay')
    return {'original_source_separator_replayed':True,
            'original_source_steps_reexecuted':True,
            'source_bytes_provenance_checked':True,
            'learned_source_universal_factorization_refuted_for_chart':True}

def native_witness_first(model,controls,*,seed_budget=4,seconds=90):
    from scripts.x369_indexed_residual_closure import native_source
    from scripts.x369_source_future_acceptance import native_future_contract
    from scripts.audit_x369_joint_action import MEANS
    from scripts.audit_x369_ordered_signed_task import TRIADS
    from scripts.x369_original_marked_seed_fabric import marked_seed_records
    from scripts.x369_original_source_consumer_coarse_fine_maxcut import marked_source_chart
    from scripts.x369_indexed_source_fibre_audit import RAW_CONSUMERS,TYPED_CONSUMERS
    seed,_,_=native_source(model)
    marks=marked_seed_records(MEANS,TRIADS,seed)
    if not marks['original_seed_marking_injective'] or len(marks['records'])!=189:
        raise ValueError('original source 189 marked input mapping not authenticated')
    mode='typed' if model.typed_consumer else 'raw'
    seeds,step,_=native_future_contract(model,mode)
    if set(seeds)!=set(marks['records']):
        raise ValueError('original native model seed ownership mismatch')
    consumers=TYPED_CONSUMERS if model.typed_consumer else RAW_CONSUMERS
    receipt=witness_first_fibre_probe(
        seeds,controls,step,marked_source_chart(marks),consumers,
        seed_budget=seed_budget,seconds=seconds)
    receipt.update(source_chart='original_189_seed_mark_only',
                   source_model='SourceGeometryOrderedLearner',
                   checkpoint_identity_checked_by_parent=True,
                   original_source_gpu_executed=True,
                   agda_kernel_checked=False)
    return receipt
