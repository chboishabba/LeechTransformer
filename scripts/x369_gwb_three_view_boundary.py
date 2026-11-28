"""Actual GWB three-view native source -> retained dialectic/trialectic boundary.

Reuse SLR's original exact native office handoff (not Python semantics),
the independently executed golden year/day consumer and native context
capsules. An unlicensed candidate never becomes an observed support.
This is an actual bounded semantic example, not original neural Stage13.
"""
import argparse
import json
from pathlib import Path
from scripts.audit_x369_native_reasoning_evidence import reopen_native,triangle,native_digest
from scripts.x369_native_stage13_evidence_manifest import load_context
from scripts.x369_original_receipt_multichart_weld import require,digest
from scripts.x369_slr_signed_support_stage import encode_support,reopen_support

EDITIONS=('en','simple','es')
LANES=('year_support','day_support')


def bridge(handoff,contexts):
    require(handoff['schema']=='slr.gwb-office-referee-handoff.v1' and
        handoff['candidate_only'] is True and handoff['formal_premise_paid'] is False and
        handoff['world_truth_paid'] is False,'native exact handoff remains candidate-only')
    views={x['edition']:x for x in handoff['views']}
    require(len(handoff['views'])==3 and set(views)==set(EDITIONS),'three actual editions')
    require(set(contexts)==set(LANES) and contexts[LANES[0]].query_ref!=contexts[LANES[1]].query_ref,
        'independently declared year/day consumer queries')
    selectors=[views[e]['selector_ref'] for e in EDITIONS]
    require(len(set(selectors))==3,'distinct native selectors')
    require(len({x['source_revision_ref'] for x in views.values()})==3,'distinct source revisions')
    by_selector={v['selector_ref']:v for v in views.values()}
    out={}
    for lane,ctx in contexts.items():
        cap,sources,pairs,_=reopen_native(ctx)
        require(ctx.family_ref==handoff['family_ref'] and ctx.consumer_ref==handoff['consumer_ref'] and
                set(sources)==set(selectors),'same actual native family scope')
        obs={s['selector_ref']:s['observation'] for s in json.loads(ctx.raw)['family']['projection']['sources']}
        for k,v in by_selector.items():
            require(v['candidate_only'] is True and v['formal_premise_paid'] is False and
                v['world_truth_paid'] is False and v['observation']==obs[k] and
                v['source_revision_ref']==sources[k]['source_revision_ref'] and
                v['parser_run_ref'] in obs[k]['parser_or_producer_ref'] and
                'sha256:'+v['source_sha256'] in obs[k]['provenance_refs'] and
                'syntax:predicate-span:'+':'.join(map(str,v['predicate_span'])) in obs[k]['native_metadata_refs'],
                'golden handoff exact native observation identity')
            require(reopen_support(encode_support(v[lane]))==v[lane],'candidate signed inverse')
        t=triangle(ctx,selectors);edge_rows=[]
        for edge in t['edges']:
            p=pairs[frozenset(edge['selectors'])]
            a,b=p['left_selector_ref'],p['right_selector_ref']
            cand=[by_selector[a][lane],by_selector[b][lane]]
            expected=[p['forward']['left_support'],p['forward']['right_support']]
            paid=p['authenticated_pair'] is not None
            matches=expected==cand if paid else None
            # Candidate missingness may itself be an exact authenticated
            # answer, but is not availability of positive/negative support.
            edge_rows.append(dict(pair_ref=p['pair_ref'],left=a,right=b,candidate_support=cand,
                native_support=expected if paid else None,candidate_agrees_with_native=matches,
                actual_pair_binding_present=paid,
                both_supports_available=edge['bounded_pair_support_available'],
                candidate_promoted=False))
        out[lane]=dict(capsule_digest=ctx.capsule_digest,query_ref=ctx.query_ref,
            source_admission=ctx.admission,boundary=t,
            boundary_digest=native_digest('x369.trialectic.boundary.v1',t),edges=edge_rows,
            all_direct_pairs_bound=all(r['actual_pair_binding_present'] for r in edge_rows),
            all_direct_candidate_answers_match=all(r['candidate_agrees_with_native'] is True for r in edge_rows),
            independent_face_evidence_required=True)
    return dict(status='actual_GWB_three_view_candidate_boundary_readback',
        family_ref=handoff['family_ref'],editions=list(EDITIONS),selectors=selectors,
        lanes=out,independent_trialectic_faces=2,face_answers_inferred=False,
        source_independence_paid=False,original_neural_stage13_identified=False,
        semantic_authority_created=False,world_truth_paid=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--manifest-sha256',required=True)
    p.add_argument('--slr-cli',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    raw=a.manifest.read_bytes();require(digest(raw)==a.manifest_sha256,'source manifest identity')
    m=json.loads(raw);root=a.manifest.parent
    require(m['schema']=='x369.gwb-three-view-boundary.v1','actual GWB source manifest')
    h=m['handoff'];b=(root/h['path']).read_bytes();require(digest(b)==h['sha256'],'source handoff bytes')
    r=bridge(json.loads(b),{k:load_context(root,v,a.slr_cli) for k,v in m['contexts'].items()})
    r['manifest_sha256']=a.manifest_sha256
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(r,f,sort_keys=True,indent=2);f.write('\n')
    print(r['status'],r['independent_trialectic_faces'])

if __name__=='__main__':main()
