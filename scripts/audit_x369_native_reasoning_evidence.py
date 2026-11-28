"""Reopen native Rust evidence and preserve independent triadic face debt.

No new semantic authority. Exact native context hashes, independent
pair locators and both directional support deltas are checked before
projecting a triangle. Three pair receipts never supply a triadic
face, a transitive licence, or a proof of world truth.
"""
import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
from scripts.x369_whole_source_indexed_transport import read_context
from scripts.x369_slr_signed_support_stage import source_support_views,reopen_support
from scripts.x369_original_receipt_multichart_weld import require


def canonical(v):
    return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def native_digest(domain,v):
    # SLR relational_family.rs context_digest, length-prefixed,
    # sorted JSON. Unsupported serialization differences fail closed.
    d=domain.encode();b=canonical(v)
    h=hashlib.sha256()
    for x in (d,b):h.update(len(x).to_bytes(8,'big'));h.update(x)
    return 'sha256:'+h.hexdigest()


def delta(left,right,missing=False):
    if missing or left is None or right is None:
        return dict(status='missing_authenticated_pair' if missing else 'unavailable_answer',
            left_support=left,right_support=right,lost=None,gained=None,
            opposed_signs=None,bounded_signed_support_equal=None)
    keys=('positive','negative')
    require(all(type(v[k]) is bool for v in (left,right) for k in keys) and
            all(set(v)==set(keys) for v in (left,right)),'native signed support type')
    lost={k:left[k] and not right[k] for k in keys}
    gained={k:right[k] and not left[k] for k in keys}
    return dict(status='preserved' if left==right else 'different',left_support=left,
        right_support=right,lost=lost,gained=gained,
        opposed_signs=(left['positive'] and right['negative']) or (left['negative'] and right['positive']),
        bounded_signed_support_equal=left==right)


def reopen_native(context):
    j=json.loads(context.raw);env=j['capsule_envelope'];c=env['capsule'];co=j['cohort'];f=j['family']
    require(native_digest('slr.context.capsule.v1',c)==context.capsule_digest and
            native_digest('slr.context.family.v1',f)==c['family_digest'] and
            native_digest('slr.context.full.v1',dict(family=f,cohort=co))==c['full_readback_digest'],
            'native context digest mismatch')
    require(co['shared_transaction_snapshot'] is True and co['derives_transitive_licences'] is False and
            co['formal_consumer_equivalence_paid'] is False and co['all_live_semantic_readings_covered'] is False and
            co['source_independence_paid'] is False and co['world_truth_paid'] is False,
            'native query-only scope')
    sources={r['selector_ref']:r for r in c['sources']}
    observations={r['selector_ref']:r['observation'] for r in f['projection']['sources']}
    require(len(sources)==len(c['sources'])==len(observations)==len(f['projection']['sources']) and
            set(sources)==set(observations)==set(co['selector_refs']),'native source inventory')
    for k,o in observations.items():
        s=sources[k]
        require(native_digest('slr.context.observation.v1',o)==s['observation_digest'] and
                s['observation_ref']==o['observation_ref'] and s['source_revision_ref']==o['source_revision_ref'],
                'native observation identity')
    views=source_support_views(context)
    pairs={frozenset((r['left_selector_ref'],r['right_selector_ref'])):r for r in co['pairs']}
    capsule={r['pair_ref']:r for r in c['pairs']}
    require(len(pairs)==co['pair_count']==len(co['pairs'])==len(capsule)==len(c['pairs']) and
            set(pairs)=={frozenset(x) for x in combinations(sources,2)},'explicit complete pair inventory')
    auth=0
    for key,p in pairs.items():
        a,b=p['left_selector_ref'],p['right_selector_ref'];q=capsule[p['pair_ref']]
        proof=p['authenticated_pair']
        if proof is None:
            require(q['proof_binding'] is None and q['proof_readback_digest'] is None and q['formal_premise'] is None,
                    'missing native proof changed')
            expected=delta(None,None,missing=True)
            require(p['forward']==p['backward']==expected,'missing pair must stay missing')
        else:
            auth+=1;binding=proof['binding']
            require(q['proof_binding']==binding and q['formal_premise']==proof['authenticated_formal_premise'] and
                    native_digest('slr.context.proof.v1',proof)==q['proof_readback_digest'],
                    'native proof readback binding')
            require(binding['schema']=='slr-reciprocal-source-query-binding-v1' and
                    binding['proof_scope']=='bounded-source-proposition' and
                    binding['family_ref']==context.family_ref and binding['consumer_ref']==context.consumer_ref and
                    binding['query_ref']==context.query_ref and binding['left_selector_ref']==a and
                    binding['right_selector_ref']==b and
                    all(binding[k]==p[k] for k in ('forward_comparison_ref','backward_comparison_ref')),
                    'same native pair/query proof coordinates')
            for side,k in (('left',a),('right',b)):
                require(binding[side+'_source_revision_ref']==sources[k]['source_revision_ref'] and
                        binding[side+'_observation_ref']==sources[k]['observation_ref'],'native proof observation identity')
            left=p['forward']['left_support'];right=p['forward']['right_support']
            require(p['forward']==delta(left,right) and p['backward']==delta(right,left),
                    'native reciprocal support delta')
    require(auth==co['authenticated_pair_count'],'native proof count')
    return c,sources,pairs,views


def triangle(context,selectors):
    c,sources,pairs,views=reopen_native(context)
    require(len(selectors)==3 and len(set(selectors))==3 and all(x in sources for x in selectors),
            'three independently addressed source selectors')
    edges=[]
    for a,b in combinations(selectors,2):
        p=pairs[frozenset((a,b))]
        keys=[(p['pair_ref'],d,s) for d in ('forward','backward') for s in ('left','right')]
        row=[views[k] for k in keys]
        edges.append(dict(selectors=[a,b],pair_ref=p['pair_ref'],proof_binding_present=p['authenticated_pair'] is not None,
            directional=[dict(key=list(k),selector=v.selector_ref,trit=v.value,residual=v.residual,
                source_support=reopen_support((v.value,v.residual)),authenticated=v.authenticated)
                for k,v in zip(keys,row)],
            bounded_pair_support_available=all(v.authenticated for v in row)))
    return dict(selectors=list(selectors),edges=edges,all_three_pair_receipts_available=all(e['bounded_pair_support_available'] for e in edges),
        face_status='independent_face_evidence_required',face_value=None,transitive_licence_created=False,
        semantic_authority_created=False,world_truth_paid=False)


def audit(context):
    c,s,p,v=reopen_native(context)
    # Enumerate only existing source selectors; do not manufacture
    # a third source to make the current two-selector capsule triadic.
    return dict(status='native_reasoning_evidence_readback',capsule_digest=context.capsule_digest,
        source_admission=context.admission,source_selectors=len(s),explicit_pair_records=len(p),
        authenticated_pair_records=sum(r['authenticated_pair'] is not None for r in p.values()),
        support_coordinates=len(v),available_authenticated_support_coordinates=sum(r.authenticated for r in v.values()),
        triangle_candidates=len(s)*(len(s)-1)*(len(s)-2)//6 if len(s)>=3 else 0,
        source_digest_reopened=True,face_evidence_inferred=False,source_proof_independently_verified_by_python=False,
        creates_semantic_authority=False,world_truth_paid=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--context-expanded',type=Path,required=True)
    p.add_argument('--context-sha256',required=True);p.add_argument('--capsule-digest',required=True)
    p.add_argument('--family',required=True);p.add_argument('--query',required=True)
    p.add_argument('--triangle',nargs=3);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();c=read_context(a.context_expanded.read_bytes(),expected_sha=a.context_sha256,
        expected_capsule=a.capsule_digest,family=a.family,query=a.query)
    r=audit(c)
    if a.triangle:r['triangle']=triangle(c,a.triangle)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(r,f,indent=2,sort_keys=True);f.write('\n')
    print(r['status'],r['triangle_candidates'],r['authenticated_pair_records'])

if __name__=='__main__':main()
