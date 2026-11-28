"""Original reachable signed-orbit capacity weld, not ambient injection.

Original Leech-LILA: Anatolii Kornienko. DASHI extension.
Reuse exact signed group and original paired history action. Numerical
stabilizers ignore zero sign; exact byte action retains it. Conjugated
words are executed from original seeds, never assumed to be equivariant.
"""
import argparse
from collections import Counter, defaultdict, deque
import json
from pathlib import Path
import struct
from time import monotonic

from models.geometric_reflection_fibre import GROUP, mul, inv
from scripts.x369_targeted_receipt_capacity_maxcut import union_receipts, CAP
from scripts.x369_original_receipt_multichart_weld import chart_values, digest, require, FIELDS
from scripts.x369_original_receipt_targeted_native import verify_parent
from scripts.x369_original_marked_seed_fabric import decode_canonical_original_seed_bytes
from scripts.x369_original_source_word_residual_weld import _encode_obs
from scripts.x369_source_future_acceptance import native_future_contract


def layout(cfg):
    e,w,dt=cfg['experts'],cfg['width'],cfg['dtype']
    require(dt in ('float32','float64') and w%3==0 and w>=9 and e>=2,
            'invalid source group layout')
    return e,w,4 if dt=='float32' else 8


def numeric_key(h,cfg):
    e,w,s=layout(cfg)
    require(len(h)==e*2*w*s,'source byte width')
    return tuple(0 if (v & ((1<<(8*s-1))-1))==0 else v
                 for v in (int.from_bytes(h[i:i+s],'little')
                           for i in range(0,len(h),s)))


def act_bytes(g,h,cfg):
    e,w,s=layout(cfg);k,b,a=g
    require(g in GROUP and len(h)==e*2*w*s,'invalid original signed action')
    out=bytearray()
    for i in range(e):
        for j in range(2):
            for t in range(w//3):
                for l in range(3):
                    c=(l+k)%3
                    if b:c=2-c
                    slot=((i*2+(j^b))*w+t*3+c)*s
                    v=bytearray(h[slot:slot+s])
                    if a:v[-1]^=128
                    out.extend(v)
    return bytes(out)


def conjugate(g,u):
    c,tag=u
    return GROUP.index(mul(mul(g,GROUP[c]),inv(g))),tag


def stabilizer(h,cfg):
    n=numeric_key(h,cfg)
    return tuple(i for i,g in enumerate(GROUP)
                 if numeric_key(act_bytes(g,h,cfg),cfg)==n)


def public_literal(u):
    c,tag=u
    return c in ((0,4,8) if tag in (0,1) else (0,10) if tag==2 else (0,1))


def source_paths(old,states,edges):
    roots={r['receipt']['original_initial_source_sha256']:r['original_seed_index']
           for r in old['stages']['indexed_fibre']['original_arbitrary_word_source_residual']
                       ['exact_source_word_transport_receipts']}
    outgoing=defaultdict(list)
    for h,u in edges:outgoing[h].append(u)
    paths={h:(i,()) for h,i in roots.items()};todo=deque(roots)
    while todo:
        h=todo.popleft()
        for u in sorted(outgoing[h]):
            n=edges[h,u]['successor']
            if n not in paths:
                i,w=paths[h];paths[n]=(i,w+(u,));todo.append(n)
    require(set(paths)==set(states),'source history has no recorded seed path')
    return roots,paths


def inventory(old,states,edges):
    cfg=old['config'];controls=tuple(tuple(u) for u in cfg['controls'])
    roots,paths=source_paths(old,states,edges)
    by_seed={i:states[h] for h,i in roots.items()}
    chart={h:chart_values(b,cfg)[CAP] for h,b in states.items()}
    stabs={h:stabilizer(b,cfg) for h,b in states.items()}
    checks=0
    for (h,u),row in edges.items():
        required={i for i in stabs[h] if conjugate(GROUP[i],u)==u}
        require(required<=set(stabs[row['successor']]),
                'observed input stabilizer not preserved')
        checks+=1
    aliases=[];nontrivial=0
    for h,b in sorted(states.items()):
        for i,g in enumerate(GROUP[1:],1):
            n=act_bytes(g,b,cfg)
            require(act_bytes(inv(g),n,cfg)==b,'signed byte action inverse')
            if numeric_key(n,cfg)==numeric_key(b,cfg):continue
            nontrivial+=1
            if chart_values(n,cfg)[CAP]!=chart[h]:continue
            seed,word=paths[h]
            seed_image=act_bytes(g,by_seed[seed],cfg)
            # Source seeds have canonical zero bytes. Decode after normalizing
            # signed zero only; compare actual seed values again on native run.
            e,w,s=layout(cfg)
            seed_image=b''.join(int(v).to_bytes(s,'little') for v in numeric_key(seed_image,cfg))
            mark=decode_canonical_original_seed_bytes(seed_image,
                experts=e,width=w,dtype=cfg['dtype'])
            require(mark is not None,'transformed source seed not canonical')
            transformed=tuple(conjugate(g,u) for u in word)
            scope=('fixed' if all(u in controls for u in transformed) else
                   'public_literal' if all(public_literal(u) for u in transformed)
                   else 'expanded_group')
            aliases.append(dict(anchor=h,group_index=i,seed=seed,word=word,
                transformed_seed_hex=seed_image.hex(),transformed_word=transformed,
                predicted_alias_hex=n.hex(),predicted_alias_sha256=digest(n),scope=scope))
    return dict(observed_histories=len(states),observed_transitions=checks,
        stabilizer_size_histogram=dict(Counter(map(len,stabs.values()))),
        all_observed_stabilizer_inclusions=True,
        nontrivial_numeric_orbit_actions=nontrivial,
        exact_cap_preserving_nontrivial_orbit_actions=len(aliases),
        scope_counts=dict(Counter(a['scope'] for a in aliases)),
        aliases=aliases,original_gpu_newly_replayed=False,
        conjugate_path_equivariance_certified=False,
        reachable_cap_descent_certified=False)


def certificate(raw,target_raw):
    old,_,states,edges,_,_=union_receipts(raw,target_raw)
    return inventory(old,states,edges)


def native_probe(model,raw,target_raw,current,*,seconds=90):
    require(0<seconds<=90,'invalid native orbit deadline')
    deadline=monotonic()+seconds
    old,new,states,edges,_,_=union_receipts(raw,target_raw)
    verify_parent(old,current);verify_parent(new,current)
    cfg=old['config'];controls=tuple(tuple(u) for u in cfg['controls'])
    invt=inventory(old,states,edges)
    roots,paths=source_paths(old,states,edges)
    seeds,step,profile=native_future_contract(model,'raw')
    by_bytes={b:i for i,b in enumerate(seeds)}
    require(len(by_bytes)==189,'original seed identity changed')
    for h,i in roots.items():require(seeds[i]==states[h],'original receipt seed mismatch')
    cache={}
    def execute(h,u):
        if (h,u) not in cache:
            if monotonic()>=deadline:raise TimeoutError('orbit probe budget')
            obs,n=step(h,u);named=dict(obs)
            require(len(named)==len(obs) and set(named)==set(FIELDS),'original consumer portfolio changed')
            cache[h,u]=(named,n)
        return cache[h,u]
    def run(seed,word,recorded):
        h=seeds[seed]
        for u in word:
            obs,n=execute(h,u)
            if recorded:
                row=edges[digest(h),u]
                require(digest(n)==row['successor'] and
                        {f:digest(repr(v).encode()) for f,v in obs.items()}==row['hashes'] and
                        (row['payload'] is None or _encode_obs(obs)==row['payload']),
                        'original path replay mismatch')
            h=n
        return h
    rows=[];timed_out=False
    for a in sorted(invt['aliases'],key=lambda a:({'fixed':0,'public_literal':1,'expanded_group':2}[a['scope']],a['anchor'],a['group_index'])):
        try:
            h=run(a['seed'],a['word'],True)
            require(h==states[a['anchor']],'anchor reopen mismatch')
            seed_image=bytes.fromhex(a['transformed_seed_hex'])
            require(seed_image in by_bytes,'transformed seed absent from actual 189 seeds')
            i=by_bytes[seed_image]
            n=run(i,a['transformed_word'],False)
            predicted=bytes.fromhex(a['predicted_alias_hex'])
            q,p=profile(h),profile(n)
            require(q==chart_values(h,cfg)[CAP] and p==chart_values(n,cfg)[CAP],
                    'native chart differs from exact offline chart')
            row=dict(a,transformed_seed=i,actual_transformed_history_hex=n.hex(),
                     actual_transformed_history_sha256=digest(n),
                     predicted_numeric_match=numeric_key(n,cfg)==numeric_key(predicted,cfg),
                     predicted_byte_match=n==predicted,
                     distinct_original_histories=n!=h,
                     same_full_cap=q==p,comparisons=[])
            # Preserve actual reachable path even if equivariance prediction
            # fails. Compare only genuine equal-chart, distinct endpoints.
            if n!=h and q==p:
                for u in controls:
                    lo,ln=execute(h,u);ro,rn=execute(n,u)
                    row['comparisons'].append(dict(control=u,
                        differing_consumer_fields=[f for f in FIELDS if lo[f]!=ro[f]],
                        successor_chart_differs=profile(ln)!=profile(rn),
                        left_exact_observations=_encode_obs(lo),right_exact_observations=_encode_obs(ro),
                        left_successor_hex=ln.hex(),right_successor_hex=rn.hex()))
            rows.append(row)
        except TimeoutError:
            timed_out=True;break
    witness=[r for r in rows if any(c['differing_consumer_fields'] or c['successor_chart_differs'] for c in r['comparisons'])]
    return dict(status='actual_conjugated_source_paths',
        checkpoint_sha256=current['checkpoint_sha256'],model_sha256=current['model_sha256'],
        source_head=current['source']['head'],
        candidates=len(invt['aliases']),completed_paths=len(rows),
        original_native_step_calls=len(cache),time_budget_exhausted=timed_out,
        paths=rows,reachable_counterexamples_by_scope=dict(Counter(r['scope'] for r in witness)),
        fixed_control_descent_refuted=any(r['scope']=='fixed' for r in witness),
        public_literal_descent_refuted=any(r['scope'] in ('fixed','public_literal') for r in witness),
        expanded_group_descent_refuted=bool(witness),
        unrestricted_source_sufficiency_certified=False,
        training_provenance_authenticated=False,agda_kernel_checked=False)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--targeted',type=Path,required=True)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    result=json.dumps(certificate(a.receipt.read_bytes(),a.targeted.read_bytes()),sort_keys=True,indent=2)+'\n'
    if a.output:a.output.write_text(result)
    print(result,end='')


if __name__=='__main__':
    main()
