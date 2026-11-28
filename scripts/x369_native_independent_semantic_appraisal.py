"""Native bounded semantic query -> independent Stage13 or trialectic appraisal.

Reuse SLR licensed-semantic (historical source answer + referee binding).
The request schema below is a DASHI consumer contract, NOT an existing
source theorem or an independently authenticated axis interpretation.
A face is never calculated from three pair answers.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from scripts.audit_x369_native_reasoning_evidence import canonical,native_digest,triangle
from scripts.x369_native_stage13_evidence_manifest import AXES,expected_address
from scripts.x369_whole_source_indexed_transport import read_context
from scripts.x369_slr_signed_support_stage import encode_support,reopen_support
from scripts.x369_original_receipt_multichart_weld import require

FACE=('reciprocalFace','coerciveFace','underdeterminedFace')
SCHEMA='x369.independent-semantic-appraisal.v1'


def sha(raw):return hashlib.sha256(raw).hexdigest()


def digest(v):return 'sha256:'+sha(canonical(v))


def pinned(root,spec):
    raw=(root/spec['path']).read_bytes()
    require(sha(raw)==spec['sha256'],'pinned native source bytes')
    return raw


def demand(request,root):
    require(request['schema']=='x369.independent-semantic-demand.v1','semantic demand schema')
    kind=request['kind'];src=request['source']
    require(kind in ('axis','face'),'independent appraisal kind')
    for k in ('revision','observation','consumer','interpretation_source_ref'):
        require(isinstance(src[k],str) and bool(src[k].strip()),'literal source coordinate '+k)
    # Bind the declared interpretation bytes, not merely a human label.
    interpretation=pinned(root,request['interpretation'])
    require(bool(interpretation),'empty interpretation declaration')
    if kind=='axis':
        axis=request['axis'];require(axis in AXES,'original StageAtlas address')
        geometry=dict(axis=axis,address=expected_address(AXES.index(axis)))
    else:
        spec=request['context'];raw=pinned(root,spec)
        ctx=read_context(raw,expected_sha=spec['sha256'],expected_capsule=spec['capsule_digest'],
            family=spec['family'],query=spec['query'])
        boundary=triangle(ctx,request['selectors'])
        geometry=dict(capsule_digest=ctx.capsule_digest,consumer_ref=ctx.consumer_ref,
            family_ref=ctx.family_ref,query_ref=ctx.query_ref,selectors=boundary['selectors'],
            boundary_digest=native_digest('x369.trialectic.boundary.v1',boundary),
            all_three_pair_receipts_available=boundary['all_three_pair_receipts_available'])
    scope=dict(kind=kind,geometry=geometry,source_revision_ref=src['revision'],
        observation_ref=src['observation'],consumer_ref=src['consumer'],
        interpretation_source_ref=src['interpretation_source_ref'],
        interpretation_sha256=request['interpretation']['sha256'])
    query='query:x369:'+native_digest('x369.independent.semantic.demand.v1',scope)
    return scope,query


def verify(receipt,scope,query):
    require(receipt['schema']=='slr.authenticated-bounded-source-semantic-query.v1' and
        receipt['formal_premise_paid'] is True and all(receipt[k] is False for k in
        ('common_current_authority_snapshot','creates_semantic_authority','all_live_semantic_readings_covered',
         'governed_identity_paid','world_truth_paid','writes_database')),'native historical bounded scope')
    b=receipt['binding'];c=receipt['candidate_receipt'];answer=receipt['answer']
    require(b['schema']=='slr-source-semantic-query-binding-v1' and b['proof_scope']=='bounded-source-predicate-query'
        and b['query_ref']==query and all(b[k]==scope[k] for k in
        ('source_revision_ref','observation_ref','consumer_ref')),'exact native predicate query binding')
    require(all(isinstance(b[k],str) and bool(b[k]) for k in ('statement_digest','evidence_ref')) and
        isinstance(receipt['reopen_refs'],list) and bool(receipt['reopen_refs']),'native proof readback')
    require(c['schema']=='slr.source-semantic-candidate-fragments.v1' and
        c['original_transport']['source_coordinate_meet']['source_revision_ref']==scope['source_revision_ref'],
        'actual source candidate revision')
    rows=[r for r in c['rows'] if r['original_predicate_transport']['checked_observation']['observation_ref']==scope['observation_ref']]
    require(len(rows)==1 and b['candidate_receipt_sha256']==digest(c) and
        b['predicate_row_sha256']==digest(rows[0]) and b['answer_sha256']==digest(answer),
        'native candidate, predicate or answer digest mismatch')
    require(answer['schema']==SCHEMA and answer['query_ref']==query and
            answer['kind']==scope['kind'] and answer['geometry']==scope['geometry'],
            'independent typed appraisal scope')
    require(set(answer)=={'schema','query_ref','kind','geometry','value'},'no authority fields in answer')
    value=answer['value']
    if scope['kind']=='axis':
        trit,residual=encode_support(value)
        require(reopen_support((trit,residual))==value,'exact independent signed support')
        result=dict(value=value,trit=trit,residual=residual,available=residual not in ('missing','both'))
    else:
        require(value is None or value in FACE,'original triadic face vocabulary')
        result=dict(value=value,available=value is not None,
            complete_boundary_and_face=value is not None and scope['geometry']['all_three_pair_receipts_available'],
            face_inferred_from_edges=False)
    return dict(scope=scope,query_ref=query,evidence_ref=b['evidence_ref'],statement_digest=b['statement_digest'],
        answer_sha256=b['answer_sha256'],candidate_receipt_sha256=b['candidate_receipt_sha256'],
        predicate_row_sha256=b['predicate_row_sha256'],result=result,
        historical_bounded_answer=True,common_current_authority_snapshot=False,
        source_interpretation_independently_proved_by_adapter=False,world_truth_paid=False)


def native_read(root,request,query,cli):
    s=request['source'];loc=request['referee'];batch=request['batches']
    pinned(root,batch)
    for k in ('run','doc','snapshot','digest'):
        require(isinstance(s[k],str) and bool(s[k]),'literal source input '+k)
    for k in ('evidence','repo','sha','path','thread'):
        require(isinstance(loc[k],str) and bool(loc[k]),'native referee locator '+k)
    p=subprocess.run([str(cli),'relational-family','licensed-semantic',s['revision'],
        str((root/batch['path']).resolve()),s['run'],s['doc'],s['snapshot'],s['digest'],
        s['observation'],s['consumer'],query,loc['evidence'],loc['repo'],loc['sha'],loc['path'],loc['thread']],
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,timeout=90)
    r=json.loads(p.stdout)
    require(r['binding']['evidence_ref']==loc['evidence'],'native evidence identity')
    return r


def assemble(rows):
    seen=set();slots={};face=None
    for r in rows:
        require(r['evidence_ref'] not in seen,'one referee evidence cannot pay distinct demands')
        seen.add(r['evidence_ref'])
        s=r['scope']
        if s['kind']=='face':
            require(face is None,'duplicate independent face');face=r
        else:
            k=s['geometry']['axis']
            require(k in AXES and k not in slots,'duplicate original axis')
            slots[k]=r
    held=[k for k in AXES if k not in slots or not slots[k]['result']['available']]
    return dict(axes=[slots.get(k) for k in AXES],held_axes=held,independent_face=face,
        all_13_historical_bounded_answers_available=not held,
        independent_face_available=bool(face and face['result']['available']),
        common_current_authority_snapshot=False,original_neural_stage13_identified=False,
        source_independence_paid=False,world_truth_paid=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--request',type=Path,required=True)
    p.add_argument('--request-sha256',required=True);p.add_argument('--receipt',type=Path);p.add_argument('--receipt-sha256')
    p.add_argument('--slr-cli',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    raw=a.request.read_bytes();require(sha(raw)==a.request_sha256,'request bytes changed')
    root=a.request.parent;q=json.loads(raw);scope,query=demand(q,root)
    require(not(a.receipt and a.slr_cli) and bool(a.receipt)==bool(a.receipt_sha256),'choose native replay or archive')
    if a.slr_cli or a.receipt:
        if a.slr_cli:r=native_read(root,q,query,a.slr_cli)
        else:r=json.loads(pinned(root,dict(path=str(a.receipt.resolve()),sha256=a.receipt_sha256)))
        out=verify(r,scope,query)
        out.update(status='native_historical_bounded_appraisal_readback',
            native_referee_reopened_this_run=bool(a.slr_cli),native_receipt=r)
    else:
        out=dict(status='independent_appraisal_demand_only',scope=scope,query_ref=query,
            formal_premise_paid=False,world_truth_paid=False)
    out['request_sha256']=a.request_sha256
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(out,f,sort_keys=True,indent=2);f.write('\n')
    print(out['status'],query)

if __name__=='__main__':main()
