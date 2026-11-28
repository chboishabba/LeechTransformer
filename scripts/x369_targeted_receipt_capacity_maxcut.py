"""Exact union of original native and targeted receipts; minimum sampled cap projection.

Original Leech-LILA: Anatolii Kornienko. DASHI geometry extension.
No new native replay. The two immutable receipt files are evidence, not
signatures of training lineage. Minimality is for coordinate projections
on the observed finite carrier only, not an arbitrary consumer codec.
"""
import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path

from scripts.x369_original_receipt_multichart_weld import (
    RECEIPT_SHA256, digest, require, extract_edges, chart_values, analyse_chart, FIELDS)
from scripts.x369_original_receipt_targeted_native import verify_parent, canonical
from scripts.x369_original_source_word_residual_weld import _decode_obs, _encode_obs
from scripts.x369_semantic_cap_geometric_section import encode, reopen

TARGET_SHA256 = 'f694fc3c5a780bb70e3f45f9aa2ac3849b60b10d287e6766120df1ab5878a37a'
SIX = 'constructed_semantic_six'
CAP = 'semantic_plus_cap_profile'


def union_receipts(raw, target_raw):
    require(digest(raw)==RECEIPT_SHA256 and digest(target_raw)==TARGET_SHA256,
            'original or targeted pinned receipt SHA mismatch')
    old,new=json.loads(raw),json.loads(target_raw)
    verify_parent(old,new)
    admission=new['stages']['admission']
    s=new['stages']['indexed_fibre']
    require(new['kind']=='new_source_acceptance' and new['complete'] and
            new['execution_complete'] and admission['status']=='admitted' and
            admission['checkpoint_digest_verified'] and admission['strict_state_dict_loaded'] and
            admission['native_gfx803_verified'] and
            admission['model_sha256']==new['model_sha256'] and
            canonical(admission['execution_contract'])==canonical(new['execution_contract']) and
            s['status']=='targeted_native_receipt_comparison' and
            s['source_and_checkpoint_identity_verified_by_parent'] and
            s['parent_receipt_file_sha256']==RECEIPT_SHA256 and
            s['source_head']==new['source']['head'] and
            s['checkpoint_sha256']==old['checkpoint_sha256'] and
            s['frozen_model_sha256']==old['model_sha256'] and
            s['execution_buffers_sha256']==old['execution_contract']['buffers_sha256'],
            'targeted native admission or identity mismatch')
    _,states,edges=extract_edges(old)
    config=old['config']
    base_states=dict(states);base_edges=dict(edges)
    controls=tuple(tuple(u) for u in config['controls'])
    size=len(next(iter(states.values())))
    values={h:chart_values(b,config) for h,b in base_states.items()}
    roots={r['original_seed_index']:r['receipt']['original_initial_source_sha256']
           for r in old['stages']['indexed_fibre']['original_arbitrary_word_source_residual']
               ['exact_source_word_transport_receipts']}

    def path_endpoint(seed,word):
        require(seed in roots,'unknown original source seed')
        h=roots[seed]
        for u in word:
            key=(h,tuple(u))
            require(key in base_edges,'unrecorded original path')
            h=base_edges[key]['successor']
        return h

    def add(h,u,n,payload,origin):
        require(len(n)==size and set(payload)==set(FIELDS),'targeted edge shape/fields')
        decoded=_decode_obs(payload)
        require(_encode_obs(decoded)==payload,'noncanonical typed observation')
        hashes={f:digest(repr(v).encode()) for f,v in decoded.items()}
        nh=digest(n)
        require(nh not in states or states[nh]==n,'source hash collision')
        states[nh]=n
        key=(h,u)
        if key in edges:
            row=edges[key]
            require(row['successor']==nh and row['hashes']==hashes and
                    (row['payload'] is None or row['payload']==payload),
                    'targeted edge conflicts with original receipt')
            row['payload']=payload
            row['provenance'].append(origin)
        else:
            edges[key]=dict(successor=nh,hashes=hashes,payload=payload,provenance=[origin])
        return chart_values(n,config)

    counts=Counter()
    seen_by_chart={}
    input_pairs={}
    for name in (SIX,CAP):
        groups=defaultdict(list)
        for h,v in values.items():groups[v[name]].append(h)
        pairs={tuple(sorted((a,b))) for g in groups.values() for a,b in combinations(g,2)}
        expected={(p,u) for p in pairs for u in controls}
        table=s['charts'][name]
        seen=set();witnesses=[]
        require(table['original_same_chart_pairs']==len(pairs) and
                table['comparisons_required']==len(expected) and
                table['comparisons_complete'],'incomplete targeted pair campaign')
        for i,row in enumerate(table['all_comparisons']):
            a,b=row['left'],row['right'];u=tuple(row['control'])
            p=tuple(sorted((a,b)));key=(p,u)
            require(key in expected and key not in seen and
                    row['independently_reopened_original_paths'] and
                    path_endpoint(row['left_seed'],row['left_word'])==a and
                    path_endpoint(row['right_seed'],row['right_word'])==b,
                    'targeted pair/source path changed')
            seen.add(key)
            lo,ro=row['left_exact_observations'],row['right_exact_observations']
            lq=add(a,u,bytes.fromhex(row['left_successor_hex']),lo,['targeted',name,i,0])
            rq=add(b,u,bytes.fromhex(row['right_successor_hex']),ro,['targeted',name,i,1])
            diff=[f for f in FIELDS if lo[f]!=ro[f]]
            succ=lq[name]!=rq[name]
            require(diff==row['differing_consumer_fields'] and
                    succ==row['successor_chart_differs'],'targeted witness discrepancy')
            if diff or succ:witnesses.append(row)
            if name==SIX:
                counts.update(diff)
                counts['consumer_separation']+=bool(diff)
                counts['successor_chart_separation']+=succ
        require(seen==expected and table['comparisons']==len(seen) and
                table['witnesses']==witnesses and
                table['status']==('refuted' if witnesses else 'sample_inconclusive'),
                'targeted comparison coverage or verdict changed')
        seen_by_chart[name]=len(seen)
        input_pairs[name]=pairs
    return old,new,states,edges,counts,seen_by_chart


def coordinates(code):
    z,(v,energies)=code
    return z,tuple(v)+tuple(x for branch in energies for x in branch)


def minimum_projections(values):
    """Exact hitting set over coordinate differences, including zero-mask refusal."""
    groups=defaultdict(list)
    for h,code in values.items():
        z,v=coordinates(code);groups[z].append((h,v))
    masks=Counter();size=len(next(iter(values.values()))[1][0])+sum(
        len(b) for b in next(iter(values.values()))[1][1])
    for group in groups.values():
        for (_,a),(_,b) in combinations(group,2):
            mask=sum(1<<i for i,(x,y) in enumerate(zip(a,b)) if x!=y)
            masks[mask]+=1
    if 0 in masks:
        return dict(sample_injective=False,minimum_extra_coordinates=None,
                    minimal_subsets=[],pair_difference_masks=dict(masks))
    for k in range(size+1):
        solutions=[]
        for subset in combinations(range(size),k):
            m=sum(1<<i for i in subset)
            if all(mask&m for mask in masks):solutions.append(subset)
        if solutions:
            return dict(sample_injective=True,minimum_extra_coordinates=k,
                        minimal_subsets=solutions,pair_difference_masks=dict(masks),
                        same_six_pairs=sum(masks.values()),
                        maximum_six_fibre=max(map(len,groups.values())),
                        coordinate_count=size)
    raise AssertionError('full profile should separate its sample')


def certificate(raw,target_raw):
    old,new,states,edges,counts,covered=union_receipts(raw,target_raw)
    cfg=old['config']
    codes={h:chart_values(b,cfg) for h,b in states.items()}
    cap={h:v[CAP] for h,v in codes.items()}
    projection=minimum_projections(cap)
    charts={name:analyse_chart({h:v[name] for h,v in codes.items()},edges)
            for name in (SIX,CAP)}
    for subset in projection['minimal_subsets']:
        chart={h:(z,tuple(v[i] for i in subset))
               for h,c in cap.items() for z,v in (coordinates(c),)}
        require(len(set(chart.values()))==len(states),'projection injectivity mismatch')
    for h,b in states.items():
        require(reopen(encode(b,cfg))==b,'expanded receipt geometric inverse failed')
    return dict(status='verified_targeted_receipt_union',
        original_receipt_sha256=digest(raw),targeted_receipt_sha256=digest(target_raw),
        native_comparisons=covered,native_separation_counts=dict(counts),
        observed_states=len(states),observed_edges=len(edges),
        exact_payload_edges=sum(r['payload'] is not None for r in edges.values()),
        digest_only_edges=sum(r['payload'] is None for r in edges.values()),
        charts=charts,minimum_sample_injective_cap_projection=projection,
        coordinate_order=[f'v_{i}' for i in range(6)]+[
            f'expert_{i}_branch_{j}_energy' for i in range(cfg['experts']) for j in range(2)],
        all_observed_histories_reopen=True,
        projection_minimality_relative_to_sample_and_cap_coordinates_only=True,
        full_source_sufficiency_certified=False,
        finite_source_graph_closure_certified=False,
        original_native_gpu_newly_replayed=False,
        training_provenance_authenticated=False,agda_kernel_checked=False)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--targeted',type=Path,required=True)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    result=json.dumps(certificate(a.receipt.read_bytes(),a.targeted.read_bytes()),
                      sort_keys=True,indent=2)+'\n'
    if a.output:a.output.write_text(result)
    print(result,end='')


if __name__=='__main__':
    main()
