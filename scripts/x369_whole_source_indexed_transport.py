"""Whole source exact recurrent transport, with independently indexed SLR context.

Original Leech-LILA: Anatolii Kornienko. DASHI extension. SLR native
query-context capsule/expansion is the production authority, not Python.
No invented Stage13 values, source semantics, or changes to learned weights.
"""
import argparse
from dataclasses import dataclass
from collections import Counter
import json
from pathlib import Path
import subprocess
from time import monotonic

from scripts.x369_reachable_capacity_residual_bounds import (
    receipt_union, encode, reopen, exact_step, ORBIT_SHA256)
from scripts.x369_original_receipt_multichart_weld import digest, require, FIELDS, RECEIPT_SHA256, chart_values
from scripts.x369_targeted_receipt_capacity_maxcut import TARGET_SHA256
from scripts.x369_original_receipt_targeted_native import verify_parent
from scripts.x369_original_source_word_residual_weld import _encode_obs, _decode_obs
from scripts.x369_source_future_acceptance import native_future_contract
from scripts.x369_stage13_pants_orbit_source_ports import attach, reopen_ports, INDEX_ROLES


@dataclass(frozen=True)
class SLRContext:
    raw: bytes
    raw_sha256: str
    capsule_digest: str
    family_ref: str
    query_ref: str
    consumer_ref: str
    admission: str


def read_context(raw,*,expected_sha,expected_capsule,family,query,admission='archival'):
    """Reopen original SLR export bytes, not a Python semantic evaluator.

    Archived file admission only binds file identity. A live read uses
    the native SLR current-replay command with its expected capsule digest.
    """
    require(type(raw) is bytes and digest(raw)==expected_sha,'SLR export byte identity')
    require(admission in ('archival','native_current_replay'),'unknown SLR context admission')
    j=json.loads(raw)
    env=j['capsule_envelope'];body=env['capsule']
    require(j['schema']=='slr.query-context-expanded.v1' and
            env['schema']=='slr.query-context-capsule-envelope.v1' and
            body['schema']=='slr.query-context-capsule.v1' and
            j['exact_current_replay'] is True and
            j['capsule_digest']==env['capsule_digest']==expected_capsule and
            j['family']['family_ref']==j['cohort']['family_ref']==body['family_ref']==family and
            j['cohort']['query_ref']==body['query_ref']==query and
            j['cohort']['consumer_ref']==body['consumer_ref'] and
            all(v['creates_semantic_authority'] is False and v['writes_database'] is False
                for v in (j,env,body)) and body['world_truth_paid'] is False,
            'SLR exact context identity or authority boundary')
    return SLRContext(raw,expected_sha,expected_capsule,family,query,body['consumer_ref'],admission)


def live_context(slr_cli,*,family,query,evidence_json,expected_capsule,seconds=90):
    """Call the existing production Rust/PG read-only replay owner."""
    require(0<seconds<=90,'SLR replay time budget')
    p=subprocess.run([str(slr_cli),'relational-family','expand-query-context-capsule',
                      family,query,evidence_json,expected_capsule],
                      check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=seconds)
    return read_context(p.stdout,expected_sha=digest(p.stdout),expected_capsule=expected_capsule,
                        family=family,query=query,admission='native_current_replay')


@dataclass(frozen=True)
class IndexedJoint:
    neural_section: dict
    context: SLRContext
    indices: dict


def pack_joint(h,config,context,indices):
    require(isinstance(context,SLRContext) and set(indices)==set(INDEX_ROLES) and
            all(v is not None for v in indices.values()),'five independently supplied source indices required')
    # No index is calculated from history hash, rank, token clock or
    # physical time. A repeated neural byte state may have new indices.
    return IndexedJoint(encode(h,config),context,dict(indices))


def reopen_joint(p):
    c=p.context
    require(read_context(c.raw,expected_sha=c.raw_sha256,expected_capsule=c.capsule_digest,
            family=c.family_ref,query=c.query_ref,admission=c.admission)==c,
            'SLR context changed')
    require(set(p.indices)==set(INDEX_ROLES) and all(v is not None for v in p.indices.values()),
            'missing independent index')
    return reopen(p.neural_section),c,dict(p.indices)


def joint_step(p,u,step,context_transport,index_transport):
    """Exact original inference; semantic context has its OWN transition.

    The source-owned callbacks govern context and indices independently,
    not from neural logits or inferred physical time. The caller may
    hold context fixed or supply actual native reviewed re-entry.
    """
    require(callable(context_transport) and callable(index_transport),'source transport owners required')
    h,c,i=reopen_joint(p)
    obs,n=step(h,u)
    nc=context_transport(c,u)
    ni=index_transport(i,c,nc,u)
    q=pack_joint(n,p.neural_section['canonical_section']['config'],nc,ni)
    require(reopen_joint(q)==(n,nc,dict(ni)),'joint source projection mismatch')
    return obs,q


def stage13_view(p,source_stage,source_pants):
    """Attach real independent source readings to this indexed state.

    Unlike a history-only reader, these callbacks can distinguish two
    equal histories with different semantic context/index coordinates.
    They must provide actual owned values, not a derived default.
    """
    h,c,i=reopen_joint(p)
    require(callable(source_stage) and callable(source_pants),'actual semantic reader ports absent')
    def read(x):
        r=source_stage(x,c,i)
        require(r['indices']==i,'Stage13 and independent index mismatch')
        return r
    result=attach(h,p.neural_section['canonical_section']['config'],read,
                  lambda x:source_pants(x,c,i))
    require(reopen_ports(result)[0].source_bytes==h,'original source view projection')
    return result


def load_receipts(raw,target,orbit):
    return receipt_union(raw,target,orbit)


def offline(raw,target,orbit):
    cfg,states,edges,scopes,_=load_receipts(raw,target,orbit)
    packets={h:encode(b,cfg) for h,b in states.items()}
    require(all(reopen(packets[h])==b for h,b in states.items()),'original history inverse')
    exact=0;unavailable=0
    for (h,u),r in edges.items():
        if r['payload'] is None:
            unavailable+=1;continue
        def recorded(b,v):
            require(b==states[h] and v==u,'recorded control/source mismatch')
            return tuple(_decode_obs(r['payload']).items()),states[r['successor']]
        obs,p=exact_step(packets[h],u,recorded)
        require(_encode_obs(dict(obs))==r['payload'] and p==packets[r['successor']] and
                reopen(p)==states[r['successor']],'exact recorded transport failed')
        exact+=1
    return dict(status='exact_retained_source_recorded_transport',
                histories=len(states),edges=len(edges),exact_recorded_edge_squares=exact,
                digest_only_edges_without_payload=unavailable,
                all_histories_reopened=True,original_gpu_newly_replayed=False,
                closed_source_graph_certified=False,smaller_chart_sufficiency_certified=False)


def native_probe(model,raw,target,orbit,current,*,seconds=90):
    """Actually run full geometric conjugation on the authenticated original source.

    Reopen both source paths through the retained section at each step,
    verify old recorded comparison observations/successors, and preserve
    actual native source callbacks. This tests the positive full fallback,
    not another search for negative reduced-chart examples.
    """
    require(0<seconds<=90,'native source transport time budget')
    deadline=monotonic()+seconds
    cfg,states,edges,_,_=load_receipts(raw,target,orbit)
    for b in (raw,target,orbit):verify_parent(json.loads(b),current)
    native=json.loads(orbit)['stages']['indexed_fibre']
    seeds,step,chart=native_future_contract(model,'raw')
    cache={};rows=[]
    def execute(packet,u):
        h=reopen(packet)
        key=(h,u)
        if key not in cache:
            if monotonic()>=deadline:raise TimeoutError('native retained source budget')
            obs,n=exact_step(packet,u,step)
            cache[key]=(obs,n)
        obs,p=cache[key];n=reopen(p)
        require(chart(n)==chart_values(n,cfg)['semantic_plus_cap_profile'],
                'native successor chart identity')
        if (digest(h),u) in edges:
            old=edges[digest(h),u];named=dict(obs)
            require(digest(n)==old['successor'] and
                    {f:digest(repr(v).encode()) for f,v in named.items()}==old['hashes'] and
                    (old['payload'] is None or _encode_obs(named)==old['payload']),
                    'original recorded native comparison changed')
        return obs,p
    def run(seed,word):
        p=encode(seeds[seed],cfg)
        for u in word:_,p=execute(p,tuple(u))
        return p
    exhausted=False
    for r in native['paths']:
        try:
            a=run(r['seed'],r['word'])
            b=run(r['transformed_seed'],r['transformed_word'])
            require(digest(reopen(a))==r['anchor'] and
                    digest(reopen(b))==r['actual_transformed_history_sha256'],
                    'retained source path endpoint changed')
            comparisons=[]
            for c in r['comparisons']:
                u=tuple(c['control'])
                lo,ln=execute(a,u);ro,rn=execute(b,u)
                require(_encode_obs(dict(lo))==c['left_exact_observations'] and
                        _encode_obs(dict(ro))==c['right_exact_observations'] and
                        reopen(ln).hex()==c['left_successor_hex'] and
                        reopen(rn).hex()==c['right_successor_hex'],
                        'native exact retained transport witness mismatch')
                comparisons.append(dict(control=u,left_next_sha256=digest(reopen(ln)),
                    right_next_sha256=digest(reopen(rn)),all_original_readouts_preserved=True))
            rows.append(dict(anchor=r['anchor'],alias=r['actual_transformed_history_sha256'],
                             scope=r['scope'],comparisons=comparisons))
        except TimeoutError:
            exhausted=True;break
    return dict(status='native_exact_retained_source_transport',paths_required=len(native['paths']),
        paths_completed=len(rows),comparisons_completed=sum(len(r['comparisons']) for r in rows),
        original_native_step_calls=len(cache),time_budget_exhausted=exhausted,rows=rows,
        all_declared_paths_preserved=len(rows)==len(native['paths']) and not exhausted,
        smaller_chart_sufficiency_certified=False,training_provenance_authenticated=False,
        actual_slr_semantic_ports_certified=False,agda_kernel_checked=False)


def main():
    p=argparse.ArgumentParser()
    for name in ('receipt','targeted','orbit'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--output',type=Path)
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--context-expanded',type=Path)
    mode.add_argument('--slr-cli',type=Path)
    p.add_argument('--context-sha256')
    p.add_argument('--capsule-digest')
    p.add_argument('--family')
    p.add_argument('--query')
    p.add_argument('--evidence-json')
    p.add_argument('--indices',type=Path,help='independently governed index record')
    p.add_argument('--history-sha256',help='actual recorded neural state to retain with context')
    a=p.parse_args()
    raw,target,orbit=(x.read_bytes() for x in (a.receipt,a.targeted,a.orbit))
    out=offline(raw,target,orbit)
    if a.context_expanded or a.slr_cli:
        if not all((a.capsule_digest,a.family,a.query,a.indices,a.history_sha256)):
            p.error('context ingress requires capsule/family/query/independent indices/history')
        if a.context_expanded:
            if not a.context_sha256 or a.evidence_json is not None:
                p.error('archived context requires exact file SHA only, not live evidence')
            ctx=read_context(a.context_expanded.read_bytes(),expected_sha=a.context_sha256,
                    expected_capsule=a.capsule_digest,family=a.family,query=a.query)
        else:
            if a.evidence_json is None or a.context_sha256 is not None:
                p.error('live native context requires evidence JSON and expected capsule digest')
            ctx=live_context(a.slr_cli,family=a.family,query=a.query,
                    evidence_json=a.evidence_json,expected_capsule=a.capsule_digest)
        cfg,states,_,_,_=load_receipts(raw,target,orbit)
        require(a.history_sha256 in states,'unknown original recorded neural history')
        indices=json.loads(a.indices.read_text())
        h=states[a.history_sha256]
        q=pack_joint(h,cfg,ctx,indices)
        require(reopen_joint(q)==(h,ctx,indices),'actual joint source reopens')
        out['independent_slr_context_ingress']=dict(
            raw_sha256=ctx.raw_sha256,capsule_digest=ctx.capsule_digest,
            family_ref=ctx.family_ref,query_ref=ctx.query_ref,consumer_ref=ctx.consumer_ref,
            admission=ctx.admission,original_neural_history_sha256=digest(h),
            exact_source_and_context_reopened=True,independent_indices=indices,
            source_stage13_semantics_certified=False,creates_semantic_authority=False)
    elif any((a.context_sha256,a.capsule_digest,a.family,a.query,a.evidence_json,
              a.indices,a.history_sha256)):
        p.error('context scope supplied without native or archived context')
    result=json.dumps(out,sort_keys=True,indent=2)+'\n'
    if a.output:
        with a.output.open('x') as out:out.write(result)
    print(result,end='')


if __name__=='__main__':main()
