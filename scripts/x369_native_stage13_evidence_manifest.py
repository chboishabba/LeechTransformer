"""Independent StageAtlas 12+1 evidence manifest over native SLR contexts.

Original StageAtlasZeroToEleven, TernaryThirteenOriginalAxisValuationBridge
and existing signed-support consumer own geometry. Native SLR owns source
answers. A manifest binds addressed source observations, not a proof
that they interpret the original neural axes. No padding missing slots.
"""
import argparse
import json
from pathlib import Path
from scripts.x369_whole_source_indexed_transport import read_context,live_context
from scripts.audit_x369_native_reasoning_evidence import reopen_native,native_digest
from scripts.x369_slr_signed_support_stage import reopen_support
from scripts.x369_original_receipt_multichart_weld import digest,require

STAGES=('seed','counter','resonance','overflow')
AXES=tuple(f'atlas-{i}' for i in range(12))+('centre',)


def expected_address(i):
    return dict(revolution=i//4,phase=STAGES[i%4]) if i<12 else dict(revolution=None,phase='centre')


def load_context(root,spec,cli=None):
    p=root/spec['path'];raw=p.read_bytes()
    c=read_context(raw,expected_sha=spec['sha256'],expected_capsule=spec['capsule_digest'],
        family=spec['family'],query=spec['query'])
    if cli:
        # Use the native Rust current readback; don't silently replace
        # an archived revision or accept a different capsule digest.
        req=root/spec['evidence_json']
        require(digest(req.read_bytes())==spec['evidence_sha256'],'native request identity')
        c=live_context(cli,family=c.family_ref,query=c.query_ref,evidence_json=req.read_text(),
                       expected_capsule=c.capsule_digest,seconds=80)
    return c


def compile_axes(manifest,contexts):
    require(manifest['schema']=='x369.stage13-native-evidence-manifest.v1' and
        isinstance(manifest['axes'],list) and len(manifest['axes'])==13 and
        [r['axis'] for r in manifest['axes']]==list(AXES),'complete original StageAtlas addresses')
    reopened={k:(c,*reopen_native(c)) for k,c in contexts.items()}
    used=set();rows=[];held=[]
    for i,r in enumerate(manifest['axes']):
        require(r['address']==expected_address(i),'original stage revolution/phase address')
        b=r['binding']
        if b is None:
            rows.append(dict(axis=r['axis'],address=r['address'],status='unbound_independent_source',
                source=None,trit=None,residual=None))
            held.append(r['axis']);continue
        require(isinstance(b['interpretation_source_ref'],str) and bool(b['interpretation_source_ref']),
                'independently declared axis interpretation source')
        c,cap,sources,pairs,views=reopened[b['context']]
        key=tuple(b['coordinate']);require(key in views,'actual native source coordinate')
        v=views[key]
        # Forward/backward copies of the same answer are not independent
        # coordinates. Preserve direction, but disallow counting the
        # same pair-selector-query answer twice as independent axes.
        identity=(c.family_ref,c.query_ref,v.pair_ref,v.selector_ref)
        require(identity not in used,'duplicate bounded answer assigned to independent axes')
        used.add(identity)
        require(b['selector_ref']==v.selector_ref and b['comparison_ref']==v.comparison_ref and
            b['observation_ref']==sources[v.selector_ref]['observation_ref'],'actual source identity')
        avail=v.authenticated and v.residual not in ('missing','both')
        status=('native_current_bounded_support' if c.admission=='native_current_replay' else
                'archival_bounded_support') if avail else 'held_missing_or_conflicting_support'
        if not avail or c.admission!='native_current_replay':held.append(r['axis'])
        rows.append(dict(axis=r['axis'],address=r['address'],status=status,trit=v.value,
            residual=v.residual,source_support=reopen_support((v.value,v.residual)),
            source=dict(context=b['context'],family=c.family_ref,query=c.query_ref,
                capsule_digest=c.capsule_digest,coordinate=list(key),selector=v.selector_ref,
                observation=b['observation_ref'],comparison=v.comparison_ref,
                interpretation_source_ref=b['interpretation_source_ref']),
            authenticated_native_readback=v.authenticated))
    return dict(status='native_stage13_bounded_support_available' if not held else 'held_independent_axis_evidence',
        axes=rows,held_axes=held,bound_axis_count=len(used),all_axes_current_bounded_support=not held,
        centre_independently_addressed=rows[-1]['source'] is not None,
        source_axis_interpretation_proved=False,original_neural_stage13_identified=False,
        semantic_authority_created=False,world_truth_paid=False)


def check_frontier(context,frontier):
    cap,sources,pairs,views=reopen_native(context)
    require(frontier['schema']=='slr.signed-query-frontier.v1' and
        frontier['family_ref']==context.family_ref and frontier['query_ref']==context.query_ref and
        frontier['consumer_ref']==context.consumer_ref and frontier['capsule_digest']==context.capsule_digest and
        frontier['exact_current_replay'] is True and frontier['candidate_only'] is True and
        all(frontier[k] is False for k in ('formal_consumer_equivalence_paid','world_truth_paid','writes_database','executes_acquisition')),
        'native signed frontier scope')
    require(frontier['structural_residuals']==cap['residuals'] and frontier['existing_queues']==cap['queues'],
        'native structural debt changed')
    expected={}
    for p in pairs.values():
        f=p['forward']
        if f['status']=='missing_authenticated_pair':reason='independent_pair_evidence_required'
        elif f['status']=='unavailable_answer':reason='source_consumer_coverage_required'
        elif f['bounded_signed_support_equal'] is False:reason='signed_support_discriminator_required'
        elif f['bounded_signed_support_equal'] is True:continue
        else:raise ValueError('unknown native pair debt')
        k=native_digest('slr.signed.query.debt.v1',[context.family_ref,context.query_ref,p['pair_ref'],reason])
        refs=[r['residual_ref'] for r in cap['residuals'] if r['comparison_ref'] in
            (p['forward_comparison_ref'],p['backward_comparison_ref'])]
        expected[k]=dict(obligation_ref=k,reason=reason,pair_ref=p['pair_ref'],
            left_selector_ref=p['left_selector_ref'],right_selector_ref=p['right_selector_ref'],
            forward_comparison_ref=p['forward_comparison_ref'],backward_comparison_ref=p['backward_comparison_ref'],
            forward=f,backward=p['backward'],existing_residual_refs=refs,
            requires_reviewed_reentry=True,candidate_only=True,paid=False)
    for s in cap['inconsistent_selector_refs']:
        k=native_digest('slr.signed.query.debt.v1',[context.family_ref,context.query_ref,s,'inconsistent_selector'])
        expected[k]=dict(obligation_ref=k,reason='inconsistent_selector_answer',selector_ref=s,
            requires_reviewed_reentry=True,candidate_only=True,paid=False)
    actual={r['obligation_ref']:r for r in frontier['obligations']}
    require(len(actual)==len(frontier['obligations']) and actual==expected,'native exact signed debt mismatch')
    return dict(obligations=len(actual),independent_pair_proof_debt=sum(
        r['reason']=='independent_pair_evidence_required' for r in actual.values()),paid=False)


def check_reentry(parent,child,receipt):
    _,ps,pp,_=reopen_native(parent);_,cs,cp,_=reopen_native(child)
    t=receipt['transition']
    require(receipt['schema']=='slr.reviewed-signed-reentry.v1' and
        receipt['shared_transaction_snapshot'] is True and receipt['accepted_published_review_reopened'] is True and
        receipt['candidate_only'] is True and all(receipt[k] is False for k in
        ('transfers_prior_pair_licences','formal_consumer_equivalence_paid','world_truth_paid','writes_database','executes_acquisition')),
        'native reviewed reentry scope')
    require(t['old_family_ref']==parent.family_ref and t['new_family_ref']==child.family_ref and
        parent.query_ref==child.query_ref and parent.consumer_ref==child.consumer_ref,'actual parent/child scope')
    old,new=t['old_selector_ref'],t['new_selector_ref']
    require(old in ps and new not in ps and set(cs)==(set(ps)-{old})|{new},'literal source replacement')
    child_raw=json.loads(child.raw)
    require(t['transition_ref'] in child_raw['family']['published_parent_transition_refs'],'native child publication lineage')
    affected=[];retained=[]
    for keys,p in cp.items():
        changed=new in keys
        expected=t['new_comparison_refs'] if changed else t['retained_comparison_refs']
        require(all(p[k] in expected for k in ('forward_comparison_ref','backward_comparison_ref')),
            'native replacement comparison ownership')
        (affected if changed else retained).append(p['pair_ref'])
        # A retained structural comparison is not a transferable proof.
        # Only the independently rebound child proof readback counts.
    require(sorted(affected)==sorted(receipt['affected_pair_refs']) and
        sorted(retained)==sorted(receipt['retained_structural_pair_refs']),'native reentry pair inventory')
    debt=check_frontier(child,receipt['frontier'])
    return dict(status='reviewed_reentry_source_readback',transition_ref=t['transition_ref'],
        affected_pairs=affected,retained_structural_pairs=retained,child_debt=debt,
        source_currently_replayed=child.admission=='native_current_replay',
        prior_pair_licences_transferred=False,world_truth_paid=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--manifest-sha256',required=True)
    p.add_argument('--slr-cli',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    raw=a.manifest.read_bytes();require(digest(raw)==a.manifest_sha256,'manifest identity')
    m=json.loads(raw)
    contexts={k:load_context(a.manifest.parent,v,a.slr_cli) for k,v in m['contexts'].items()}
    r=compile_axes(m,contexts)
    if 'reviewed_reentry' in m:
        b=m['reviewed_reentry'];raw=(a.manifest.parent/b['path']).read_bytes()
        require(digest(raw)==b['sha256'],'native reentry receipt identity')
        r['reviewed_reentry']=check_reentry(contexts[b['parent']],contexts[b['child']],json.loads(raw))
    r['manifest_sha256']=a.manifest_sha256
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as out:json.dump(r,out,sort_keys=True,indent=2);out.write('\n')
    print(r['status'],r['bound_axis_count'],len(r['held_axes']))

if __name__=='__main__':main()
