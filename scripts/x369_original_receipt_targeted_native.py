"""Target only the actual semantic chart collisions already in a native receipt.

Original Leech-LILA: Anatolii Kornienko; DASHI source weld.
Use the immutable parent source acceptance and original native callback.
Reopen every tested history from an original seed, not merely a byte injection.
"""
from collections import defaultdict, deque
from itertools import combinations
import json
from time import monotonic

from scripts.x369_original_receipt_multichart_weld import (
    RECEIPT_SHA256, digest, require, extract_edges, chart_values, FIELDS)
from scripts.x369_original_source_word_residual_weld import _encode_obs
from scripts.x369_source_future_acceptance import native_future_contract


def canonical(x):
    return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)


def verify_parent(receipt, current):
    for key in ('checkpoint_sha256','model_sha256','config','execution_contract'):
        require(canonical(receipt[key])==canonical(current[key]),
                'original native parent mismatch: '+key)
    old,new=receipt['source']['files'],current['source']['files']
    owners = ('scripts/x369_source_geometry.py','scripts/x369_source_future_acceptance.py',
              'scripts/x369_indexed_residual_closure.py','scripts/audit_x369_joint_action.py',
              'scripts/audit_x369_ordered_signed_task.py','scripts/x369_capped_semantic_geometry.py')
    require({k:v for k,v in old.items() if k.startswith('models/')} ==
            {k:v for k,v in new.items() if k.startswith('models/')} and
            all(k in old and old[k]==new.get(k) for k in owners),
            'original model or chart owner changed')


def targeted_compare(receipt, seeds, step, native_profile, *, seconds=90):
    require(0 < seconds <= 90, 'invalid cooperative source deadline')
    deadline=monotonic()+seconds
    _,states,edges=extract_edges(receipt)
    cfg=receipt['config']
    controls=tuple(tuple(u) for u in cfg['controls'])
    values={h:chart_values(b,cfg) for h,b in states.items()}
    names=('constructed_semantic_six','semantic_plus_cap_profile')
    roots={}
    rows=receipt['stages']['indexed_fibre']['original_arbitrary_word_source_residual']['exact_source_word_transport_receipts']
    for r in rows:
        i=r['original_seed_index']
        h=r['receipt']['original_initial_source_sha256']
        require(type(i) is int and 0<=i<len(seeds) and seeds[i]==states[h],
                'original native seed identity changed')
        roots[h]=i
    outgoing=defaultdict(list)
    for h,u in edges:
        outgoing[h].append(u)
    paths={h:(i,()) for h,i in roots.items()}
    todo=deque(roots)
    while todo:
        h=todo.popleft()
        for u in sorted(outgoing[h]):
            n=edges[h,u]['successor']
            if n not in paths:
                i,p=paths[h]
                paths[n]=(i,p+(u,))
                todo.append(n)

    cache={}
    def execute(h,u):
        key=(h,u)
        if key not in cache:
            if monotonic()>=deadline:
                raise TimeoutError('source step budget')
            obs,n=step(h,u)
            named=dict(obs)
            require(len(named)==len(obs) and set(named)==set(FIELDS),
                    'original consumer portfolio changed')
            cache[key]=(named,n)
        return cache[key]

    reopened={}
    def reopen(hid):
        if hid in reopened:
            return reopened[hid]
        require(hid in paths, 'receipt history has no original source path')
        seed,word=paths[hid]
        h=seeds[seed]
        for u in word:
            recorded=edges[digest(h),u]
            obs,n=execute(h,u)
            require(digest(n)==recorded['successor'] and
                    {f:digest(repr(v).encode()) for f,v in obs.items()}==recorded['hashes'],
                    'historical original source replay mismatch')
            if recorded['payload'] is not None:
                require(_encode_obs(obs)==recorded['payload'],
                        'historical exact consumer mismatch')
            h=n
        require(h==states[hid],'original history replay mismatch')
        p=native_profile(h)
        require(p[0]==values[hid][names[0]] and p==values[hid][names[1]],
                'offline/native exact chart mismatch')
        reopened[hid]=h
        return h

    results={}
    for name in names:
        groups=defaultdict(list)
        for h,v in values.items():
            groups[v[name]].append(h)
        pairs=[(a,b) for group in groups.values() for a,b in combinations(group,2)]
        comparisons=[];timed_out=False
        for a,b in pairs:
            try:
                left,right=reopen(a),reopen(b)
                for u in controls:
                    lo,ln=execute(left,u);ro,rn=execute(right,u)
                    lq,rq=chart_values(ln,cfg)[name],chart_values(rn,cfg)[name]
                    fields=[f for f in FIELDS if lo[f]!=ro[f]]
                    comparisons.append(dict(left=a,right=b,control=u,
                        left_seed=paths[a][0],left_word=paths[a][1],
                        right_seed=paths[b][0],right_word=paths[b][1],
                        differing_consumer_fields=fields,successor_chart_differs=lq!=rq,
                        left_successor_hex=ln.hex(),right_successor_hex=rn.hex(),
                        left_exact_observations=_encode_obs(lo),right_exact_observations=_encode_obs(ro),
                        independently_reopened_original_paths=True))
            except TimeoutError:
                timed_out=True
                break
        witnesses=[p for p in comparisons if p['differing_consumer_fields'] or p['successor_chart_differs']]
        results[name]=dict(
            status='refuted' if witnesses else 'time_budget' if timed_out else 'sample_inconclusive',
            original_same_chart_pairs=len(pairs),comparisons=len(comparisons),
            comparisons_required=len(pairs)*len(controls),
            comparisons_complete=len(comparisons)==len(pairs)*len(controls),
            witnesses=witnesses,all_comparisons=comparisons,
            unrestricted_source_sufficiency_certified=False)
    return dict(status='targeted_native_receipt_comparison',
        charts=results,original_states_reopened=len(reopened),
        original_native_step_calls=len(cache),
        original_source_graph_closure_certified=False,
        training_provenance_authenticated=False)


def native_targeted(model, raw, current, *, seconds=90):
    require(digest(raw)==RECEIPT_SHA256,'original pinned receipt SHA mismatch')
    old=json.loads(raw)
    verify_parent(old,current)
    seeds,step,profile=native_future_contract(model,'raw')
    result=targeted_compare(old,seeds,step,profile,seconds=seconds)
    result.update(parent_receipt_file_sha256=digest(raw),
                  checkpoint_sha256=current['checkpoint_sha256'],
                  frozen_model_sha256=current['model_sha256'],
                  execution_buffers_sha256=current['execution_contract']['buffers_sha256'],
                  source_head=current['source']['head'],
                  source_and_checkpoint_identity_verified_by_parent=True,
                  unrestricted_native_source_sufficient=False,
                  agda_kernel_checked=False)
    return result
