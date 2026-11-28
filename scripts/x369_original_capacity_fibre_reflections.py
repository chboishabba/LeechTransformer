"""Exact cap-preserving signed fibre reflections and optional native stress test.

Original Leech-LILA: Anatolii Kornienko. DASHI geometry extension.
A constructed alias is an ambient finite-float history, NOT a learned
reachable state unless an original path has separately been witnessed.
"""
import argparse
from collections import defaultdict, deque
import json
from pathlib import Path
from time import monotonic

from scripts.x369_targeted_receipt_capacity_maxcut import union_receipts, CAP
from scripts.x369_original_receipt_multichart_weld import chart_values, digest, require, FIELDS
from scripts.x369_original_receipt_targeted_native import verify_parent
from scripts.x369_original_source_word_residual_weld import _encode_obs
from scripts.x369_source_future_acceptance import native_future_contract


def reflect_triads(h,config):
    """Flip full signed triads on semantic-invisible coordinates.

    Expert zero: only unused slots j>=9. Other experts: every triad.
    Each reflection is an involutive exact float-bit isometry; six,
    signed residual cross terms and all branch energies are unchanged.
    """
    e,w,dt=config['experts'],config['width'],config['dtype']
    require(dt in ('float32','float64') and w>=9 and w%3==0,
            'paired triad source required')
    size=4 if dt=='float32' else 8
    q=chart_values(h,config)[CAP]
    for i in range(e):
        for b in range(2):
            for start in range(9 if i==0 else 0,w,3):
                pos=[size*((2*i+b)*w+j)+size-1 for j in range(start,start+3)]
                if not any(any(h[k-size+1:k]) or h[k]&127 for k in pos):
                    continue
                other=bytearray(h)
                for k in pos:other[k]^=128
                other=bytes(other)
                require(other!=h and chart_values(other,config)[CAP]==q,
                        'capacity reflection changed original chart')
                reopen=bytearray(other)
                for k in pos:reopen[k]^=128
                require(bytes(reopen)==h,'capacity reflection not involutive')
                yield (i,b,start),other


def candidate_pairs(states,config):
    for h,b in sorted(states.items()):
        for reflection,other in reflect_triads(b,config):
            yield h,reflection,other


def certificate(raw,target_raw):
    old,_,states,_,_,_=union_receipts(raw,target_raw)
    cfg=old['config']
    n=0;anchors=set();ambient=set();known=0
    for h,reflection,other in candidate_pairs(states,cfg):
        n+=1;anchors.add(h)
        key=digest(other)
        if key in states:
            require(states[key]==other,'source hash collision')
            known+=1
        else:ambient.add(key)
    return dict(status='capacity_preserving_fibre_reflections',
        observed_histories=len(states),anchors_with_nontrivial_reflection=len(anchors),
        exact_reflection_pairs=n,distinct_aliases_not_in_receipt=len(ambient),
        aliases_with_recorded_source_path=known,
        chart_preserved_exactly=True,bit_involution_verified=True,
        constructed_alias_reachability_not_certified=True,
        original_gpu_model_newly_replayed=False,
        consumer_descent_certified=False)


def native_probe(model,raw,target_raw,current,*,seconds=90,anchor_budget=4):
    require(0<seconds<=90 and type(anchor_budget) is int and 1<=anchor_budget<=16,
            'invalid bounded capacity fibre scope')
    deadline=monotonic()+seconds
    old,new,states,edges,_,_=union_receipts(raw,target_raw)
    verify_parent(old,current);verify_parent(new,current)
    cfg=old['config'];controls=tuple(tuple(u) for u in cfg['controls'])
    seeds,step,profile=native_future_contract(model,'raw')
    roots={}
    for r in old['stages']['indexed_fibre']['original_arbitrary_word_source_residual']['exact_source_word_transport_receipts']:
        i=r['original_seed_index'];h=r['receipt']['original_initial_source_sha256']
        require(seeds[i]==states[h],'native source seed changed')
        roots[h]=i
    outgoing=defaultdict(list)
    for h,u in edges:outgoing[h].append(u)
    paths={h:(i,()) for h,i in roots.items()};todo=deque(roots)
    while todo:
        h=todo.popleft()
        for u in sorted(outgoing[h]):
            n=edges[h,u]['successor']
            if n not in paths:
                i,word=paths[h];paths[n]=(i,word+(u,));todo.append(n)
    cache={}
    def execute(h,u):
        if (h,u) not in cache:
            if monotonic()>=deadline:raise TimeoutError('native capacity probe budget')
            obs,n=step(h,u);named=dict(obs)
            require(len(named)==len(obs) and set(named)==set(FIELDS),'raw consumer mismatch')
            cache[h,u]=(named,n)
        return cache[h,u]
    def reopen(h):
        i,word=paths[h];b=seeds[i]
        for u in word:
            row=edges[digest(b),u];obs,n=execute(b,u)
            require(digest(n)==row['successor'] and
                    {f:digest(repr(v).encode()) for f,v in obs.items()}==row['hashes'] and
                    (row['payload'] is None or _encode_obs(obs)==row['payload']),
                    'historical native edge replay mismatch')
            b=n
        require(b==states[h] and profile(b)==chart_values(b,cfg)[CAP],
                'native anchor or chart replay mismatch')
        return b
    # Prefer the fourth expert: exact finite receipt projection found
    # either of its branch energies separates the entire observed sample.
    candidates=defaultdict(list)
    for h,ref,other in candidate_pairs(states,cfg):candidates[h].append((ref,other))
    selected=sorted(candidates,key=lambda h:(not any(r[0]==cfg['experts']-1 for r,_ in candidates[h]),len(paths[h][1]),h))[:anchor_budget]
    rows=[];timed_out=False
    for h in selected:
        try:
            b=reopen(h)
            opts=sorted(candidates[h],key=lambda p:(p[0][0]!=cfg['experts']-1,p[0]))
            # One geometric alias per anchor, four common source controls.
            ref,other=opts[0];k=digest(other)
            reachable=k in paths
            if reachable:require(reopen(k)==other,'alias original path mismatch')
            require(profile(other)==profile(b),'native capacity chart mismatch')
            for u in controls:
                lo,ln=execute(b,u);ro,rn=execute(other,u)
                fields=[f for f in FIELDS if lo[f]!=ro[f]]
                chart_diff=profile(ln)!=profile(rn)
                rows.append(dict(anchor_sha256=h,anchor_path=paths[h],
                    reflection=ref,alias_sha256=k,alias_hex=other.hex(),
                    alias_reachable_by_recorded_path=reachable,control=u,
                    differing_consumer_fields=fields,successor_chart_differs=chart_diff,
                    left_exact_observations=_encode_obs(lo),right_exact_observations=_encode_obs(ro),
                    left_successor_hex=ln.hex(),right_successor_hex=rn.hex()))
        except TimeoutError:
            timed_out=True;break
    witnesses=[r for r in rows if r['differing_consumer_fields'] or r['successor_chart_differs']]
    return dict(status='capacity_fibre_ambient_probe',
        original_receipt_sha256=digest(raw),targeted_receipt_sha256=digest(target_raw),
        original_checkpoint_sha256=current['checkpoint_sha256'],
        original_model_sha256=current['model_sha256'],
        source_head=current['source']['head'],
        selected_anchors=len(selected),comparisons=len(rows),time_budget_exhausted=timed_out,
        comparisons_required=len(selected)*len(controls),original_native_step_calls=len(cache),
        ambient_counterexamples=len(witnesses),
        reachable_counterexamples=sum(r['alias_reachable_by_recorded_path'] for r in witnesses),
        comparisons_data=rows,
        universal_ambient_chart_refuted=bool(witnesses),
        observed_reachable_chart_refuted=any(r['alias_reachable_by_recorded_path'] for r in witnesses),
        unrestricted_source_sufficiency_certified=False,
        training_provenance_authenticated=False,agda_kernel_checked=False)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--targeted',type=Path,required=True)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    result=json.dumps(certificate(a.receipt.read_bytes(),a.targeted.read_bytes()),indent=2,sort_keys=True)+'\n'
    if a.output:a.output.write_text(result)
    print(result,end='')


if __name__=='__main__':main()
