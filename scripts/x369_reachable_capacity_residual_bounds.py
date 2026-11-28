"""Three native receipts -> scope-specific residual bounds and exact orbit section.

Original Leech-LILA: Anatolii Kornienko; DASHI source geometry extension.
A pinned native path receipt attests its endpoints; the transformed path's
intermediate steps are not retained and are NOT independently replayed here.
No finite graph closure or training lineage is inferred.
"""
import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path

from models.geometric_reflection_fibre import GROUP
from scripts.x369_targeted_receipt_capacity_maxcut import union_receipts, CAP
from scripts.x369_original_receipt_multichart_weld import chart_values, digest, require, FIELDS
from scripts.x369_original_receipt_targeted_native import verify_parent, canonical
from scripts.x369_original_source_word_residual_weld import _decode_obs, _encode_obs
from scripts.x369_reachable_capacity_orbit_weld import inventory, source_paths, act_bytes, numeric_key, layout
from scripts.x369_semantic_cap_geometric_section import encode as section_encode, reopen as section_reopen

ORBIT_SHA256='2f8a37e937f76e11857421d3f1cdce9d2acc08b1d765f9e7a8a4815780548d90'
SCOPES=('fixed','public_literal','expanded_group')


def receipt_union(raw,target_raw,orbit_raw):
    require(digest(orbit_raw)==ORBIT_SHA256,'pinned native orbit receipt SHA mismatch')
    old,target,states,edges,_,_=union_receipts(raw,target_raw)
    native=json.loads(orbit_raw)
    verify_parent(old,native);verify_parent(target,native)
    admission=native['stages']['admission']
    s=native['stages']['indexed_fibre']
    require(native['kind']=='new_source_acceptance' and native['complete'] and
            native['execution_complete'] and admission['status']=='admitted' and
            admission['checkpoint_digest_verified'] and admission['strict_state_dict_loaded'] and
            admission['native_gfx803_verified'] and
            admission['model_sha256']==native['model_sha256'] and
            canonical(admission['execution_contract'])==canonical(native['execution_contract']) and
            s['status']=='actual_conjugated_source_paths' and
            s['checkpoint_sha256']==native['checkpoint_sha256'] and
            s['model_sha256']==native['model_sha256'] and
            s['source_head']==native['source']['head'] and not s['time_budget_exhausted'],
            'native orbit receipt admission or identity mismatch')
    cfg=old['config'];controls=tuple(tuple(u) for u in cfg['controls'])
    roots,paths=source_paths(old,states,edges)
    roots_by_seed={i:states[h] for h,i in roots.items()}
    inv=inventory(old,states,edges)
    expected={(a['anchor'],a['group_index']):a for a in inv['aliases']}
    seen=set();counts=Counter();scope_counts=Counter()
    scopes={h:0 for h in states}
    size=len(next(iter(states.values())))

    def put(b,scope):
        require(len(b)==size,'native orbit history size mismatch')
        h=digest(b)
        require(h not in states or states[h]==b,'native history hash conflict')
        states[h]=b
        scopes[h]=min(scope,scopes.get(h,scope))
        return h

    def add(h,u,n,payload,scope,origin):
        nh=put(n,scope)
        require(set(payload)==set(FIELDS),'native orbit consumer fields')
        decoded=_decode_obs(payload)
        require(_encode_obs(decoded)==payload,'noncanonical native orbit observation')
        hashes={f:digest(repr(v).encode()) for f,v in decoded.items()}
        key=h,u
        if key in edges:
            r=edges[key]
            require(r['successor']==nh and r['hashes']==hashes and
                    (r['payload'] is None or r['payload']==payload),'native orbit edge conflict')
            r['payload']=payload;r['provenance'].append(origin)
        else:
            edges[key]=dict(successor=nh,hashes=hashes,payload=payload,provenance=[origin])
        return chart_values(n,cfg)[CAP]

    for i,r in enumerate(s['paths']):
        key=r['anchor'],r['group_index']
        require(key in expected and key not in seen,'unexpected or repeated conjugated path')
        a=expected[key];seen.add(key)
        require(all(canonical(r[k])==canonical(v) for k,v in a.items()),
                'original conjugate path inventory changed')
        h=states[r['anchor']]
        scope=SCOPES.index(a['scope'])
        # All eight observed transformed seeds are among the four retained
        # original roots. Refuse an unsupported seed in this pinned campaign.
        require(r['transformed_seed'] in roots_by_seed and
                roots_by_seed[r['transformed_seed']].hex()==a['transformed_seed_hex'],
                'transformed original seed mismatch')
        n=bytes.fromhex(r['actual_transformed_history_hex'])
        nh=put(n,scope)
        p=bytes.fromhex(a['predicted_alias_hex'])
        same=chart_values(h,cfg)[CAP]==chart_values(n,cfg)[CAP]
        require(nh==r['actual_transformed_history_sha256'] and
                r['predicted_numeric_match']==(numeric_key(n,cfg)==numeric_key(p,cfg)) and
                r['predicted_byte_match']==(n==p) and r['distinct_original_histories']==(n!=h) and
                r['same_full_cap']==same and same and n!=h,
                'native endpoint or chart mismatch')
        counts['numeric_prediction_matches']+=r['predicted_numeric_match']
        counts['byte_prediction_matches']+=r['predicted_byte_match']
        counts['numerically_distinct_anchor_alias']+=numeric_key(n,cfg)!=numeric_key(h,cfg)
        seen_controls=set();witness=False
        for j,c in enumerate(r['comparisons']):
            u=tuple(c['control'])
            require(u in controls and u not in seen_controls,'native orbit comparison coverage')
            seen_controls.add(u)
            lo,ro=c['left_exact_observations'],c['right_exact_observations']
            lq=add(r['anchor'],u,bytes.fromhex(c['left_successor_hex']),lo,0,['orbit',i,j,0])
            rq=add(nh,u,bytes.fromhex(c['right_successor_hex']),ro,scope,['orbit',i,j,1])
            diff=[f for f in FIELDS if lo[f]!=ro[f]]
            succ=lq!=rq
            require(diff==c['differing_consumer_fields'] and succ==c['successor_chart_differs'],
                    'native orbit separation mismatch')
            counts.update(diff);counts['consumer_separation']+=bool(diff)
            counts['successor_chart_separation']+=succ
            witness |= bool(diff) or succ
        require(seen_controls==set(controls),'missing common native controls')
        if witness:scope_counts[a['scope']]+=1
    require(seen==set(expected) and s['candidates']==s['completed_paths']==len(seen) and
            dict(scope_counts)==s['reachable_counterexamples_by_scope'] and
            s['fixed_control_descent_refuted']==bool(scope_counts['fixed']) and
            s['public_literal_descent_refuted']==bool(scope_counts['fixed']+scope_counts['public_literal']) and
            s['expanded_group_descent_refuted']==bool(scope_counts.total()),
            'native orbit campaign verdict mismatch')
    # Every recorded comparison uses the four fixed controls. Propagate
    # narrower admitted reachability if an endpoint acquires such a path.
    changed=True
    while changed:
        changed=False
        for (h,u),r in edges.items():
            require(u in controls,'unexpected recorded comparison control')
            n=r['successor']
            if scopes[h]<scopes[n]:
                scopes[n]=scopes[h];changed=True
    return cfg,states,edges,scopes,counts


def normalized(h,cfg):
    _,_,s=layout(cfg)
    return b''.join(v.to_bytes(s,'little') for v in numeric_key(h,cfg))


def fibre_orbit(h,cfg):
    """Distinct numerical signed-group images in the SAME exact cap fibre."""
    q=chart_values(h,cfg)[CAP]
    images={normalized(n,cfg) for g in GROUP
            for n in (act_bytes(g,h,cfg),) if chart_values(n,cfg)[CAP]==q}
    return tuple(sorted(images,key=lambda b:numeric_key(b,cfg)))


def orientation(h,cfg):
    orbit=fibre_orbit(h,cfg)
    return orbit[0],orbit.index(normalized(h,cfg)),len(orbit)


def encode(h,cfg):
    canon,mark,_=orientation(h,cfg)
    _,_,s=layout(cfg)
    signs=tuple(i for i in range(len(h)//s) if h[i*s:(i+1)*s]==(1<<(8*s-1)).to_bytes(s,'little'))
    return dict(schema=1,canonical_section=section_encode(canon,cfg),
                orientation_rank=mark,negative_zero_positions=list(signs),
                original_sha256=digest(h))


def reopen(packet):
    require(packet['schema']==1,'orbit section schema')
    section=packet['canonical_section'];cfg=section['config']
    canon=section_reopen(section);orbit=fibre_orbit(canon,cfg)
    mark=packet['orientation_rank']
    require(orbit[0]==canon and type(mark) is int and 0<=mark<len(orbit),
            'noncanonical orbit or rank')
    h=bytearray(orbit[mark]);_,_,s=layout(cfg)
    signs=packet['negative_zero_positions']
    require(type(signs) is list and signs==sorted(set(signs)) and
            all(type(i) is int and 0<=i<len(h)//s for i in signs),'invalid zero sign positions')
    for i in signs:
        require(h[i*s:(i+1)*s]==bytes(s),'zero sign on nonzero scalar')
        h[(i+1)*s-1]^=128
    h=bytes(h)
    require(digest(h)==packet['original_sha256'] and encode(h,cfg)==packet,'orbit section BIDI mismatch')
    return h


def exact_step(packet,u,step):
    """Exact fallback source dynamics on the geometric orbit packet.
    step is the admitted original callback; no invented quotient transition.
    """
    h=reopen(packet)
    obs,n=step(h,u)
    return obs,encode(n,packet['canonical_section']['config'])


def scope_bounds(states,edges,scopes,codes,marks,level):
    groups=defaultdict(list)
    for h in sorted(states):
        if scopes[h]<=level:groups[codes[h]].append(h)
    best=[];pair_witnesses=[];unresolved=0;collision_pairs=0
    for g in groups.values():
        pairs={}
        for a,b in combinations(g,2):
            collision_pairs+=1
            for h,u in edges:
                if h!=a or (b,u) not in edges:continue
                l,r=edges[a,u],edges[b,u]
                if l['payload'] is None or r['payload'] is None:continue
                fields=[f for f in FIELDS if l['payload'][f]!=r['payload'][f]]
                succ=codes[l['successor']]!=codes[r['successor']]
                if fields or succ:
                    pairs[a,b]=dict(left=a,right=b,control=u,differing_fields=fields,
                                    successor_chart_differs=succ)
                    break
            if (a,b) not in pairs:unresolved+=1
        pair_witnesses.extend(pairs.values())
        for k in range(len(g),len(best),-1):
            found=next((xs for xs in combinations(g,k)
                        if all(p in pairs for p in combinations(xs,2))),None)
            if found:best=list(found);break
    size=max(map(len,groups.values()))
    labels=len(best)
    return dict(scope=SCOPES[level],observed_states=sum(map(len,groups.values())),
        cap_fibres=len(groups),same_cap_pairs=collision_pairs,
        exact_witness_pairs=len(pair_witnesses),unresolved_pairs=unresolved,
        maximum_witnessed_clique=best,minimum_required_residual_labels_lower_bound=labels,
        fixed_width_residual_bits_lower_bound=(max(1,labels)-1).bit_length(),
        maximum_observed_fibre_size=size,
        sample_injective_rank=all(len(set(marks[h] for h in g))==len(g) for g in groups.values()),
        exact_sample_label_optimality=(labels==size),
        pair_witnesses=pair_witnesses,universal_source_sufficiency_certified=False)


def certificate(raw,target_raw,orbit_raw):
    cfg,states,edges,scopes,counts=receipt_union(raw,target_raw,orbit_raw)
    codes={h:chart_values(b,cfg)[CAP] for h,b in states.items()}
    orientation_rows={h:orientation(b,cfg) for h,b in states.items()}
    marks={h:v[1] for h,v in orientation_rows.items()}
    groups=defaultdict(list)
    for h,q in codes.items():groups[q].append(h)
    for h,b in states.items():
        require(reopen(encode(b,cfg))==b,'original orbit-section byte roundtrip')
    return dict(status='native_reachable_capacity_residual_bounds',
        native_orbit_receipt_sha256=digest(orbit_raw),
        observed_states=len(states),observed_edges=len(edges),
        exact_payload_edges=sum(r['payload'] is not None for r in edges.values()),
        digest_only_edges=sum(r['payload'] is None for r in edges.values()),
        native_counts=dict(counts),
        scope_bounds=[scope_bounds(states,edges,scopes,codes,marks,s) for s in range(3)],
        observed_cap_fibres_share_one_numeric_orbit=all(
            len({orientation_rows[h][0] for h in g})==1 for g in groups.values()),
        maximum_observed_fibre_orbit_size=max(v[2] for v in orientation_rows.values()),
        observed_orientation_ranks=sorted(set(marks.values())),
        all_observed_orbit_section_roundtrips=True,
        exact_group_orbit_rank_only_sample_injective=all(
            len({marks[h] for h in g})==len(g) for g in groups.values()),
        transformed_path_intermediates_independently_replayed=False,
        original_native_gpu_newly_replayed=False,
        universal_reachable_source_sufficiency_certified=False,
        finite_source_graph_closure_certified=False,training_provenance_authenticated=False,
        agda_kernel_checked=False)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--targeted',type=Path,required=True)
    p.add_argument('--orbit',type=Path,required=True)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    result=json.dumps(certificate(a.receipt.read_bytes(),a.targeted.read_bytes(),a.orbit.read_bytes()),sort_keys=True,indent=2)+'\n'
    if a.output:a.output.write_text(result)
    print(result,end='')


if __name__=='__main__':
    main()
