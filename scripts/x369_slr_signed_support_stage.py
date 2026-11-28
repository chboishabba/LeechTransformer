"""Production SLR bounded signed-support -> retained sophontic chart.

Native Rust readback is authority. This is an exact consumer adapter,
NOT a semantic evaluator, proof referee, or source of new pair licences.
Four observed support patterns and missing evidence remain distinct.
"""
from dataclasses import dataclass
import json
from scripts.x369_original_receipt_multichart_weld import require
from scripts.x369_whole_source_indexed_transport import reopen_joint

# This is a chosen support projection, not a truth valuation of a proposition.
# In particular both, neither and unobserved cannot all become neutral.
SUPPORT={
    None:(0,'missing'),
    (False,False):(0,'neither'),
    (True,False):(1,'positive_only'),
    (False,True):(-1,'negative_only'),
    (True,True):(0,'both'),
}
INVERSE={v:k for k,v in SUPPORT.items()}


def encode_support(raw):
    if raw is None:return SUPPORT[None]
    require(isinstance(raw,dict) and set(raw)=={'positive','negative'} and
            all(type(v) is bool for v in raw.values()),'actual signed support schema')
    return SUPPORT[(raw['positive'],raw['negative'])]


def reopen_support(value):
    require(value in INVERSE,'invalid signed support residual')
    v=INVERSE[value]
    return None if v is None else dict(positive=v[0],negative=v[1])


@dataclass(frozen=True)
class BoundedSupportView:
    pair_ref: str
    comparison_ref: str
    side: str
    selector_ref: str
    value: int
    residual: str
    status: str
    authenticated: bool


def source_support_views(context):
    j=json.loads(context.raw)
    body=j['capsule_envelope']['capsule']
    require(body['query_ref']==context.query_ref and body['family_ref']==context.family_ref and
            body['world_truth_paid'] is False and body['creates_semantic_authority'] is False,
            'native scope/authority')
    pairs={r['pair_ref']:r for r in j['cohort']['pairs']}
    require(len(pairs)==len(j['cohort']['pairs']),'duplicate native pair')
    views={}
    inconsistent=set(j['cohort'].get('inconsistent_selector_refs',[]))
    for r in body['pairs']:
        cp=pairs[r['pair_ref']]
        for name in ('left_selector_ref','right_selector_ref','forward_comparison_ref','backward_comparison_ref'):
            require(cp[name]==r[name],'native pair coordinates differ')
        for direction in ('forward','backward'):
            d=r[direction];require(cp[direction]==d,'native direction readback differs')
            comparison=r[direction+'_comparison_ref']
            for side in ('left','right'):
                v=d[side+'_support']
                trit,tag=encode_support(v)
                require(reopen_support((trit,tag))==v,'signed support exact inverse')
                # The native backwards comparison exchanges selector roles.
                selector_side=side if direction=='forward' else ('right' if side=='left' else 'left')
                admitted=(cp['authenticated_pair'] is not None and
                          r.get('proof_binding') is not None and
                          d['status'] in ('preserved','different') and v is not None and
                          r[selector_side+'_selector_ref'] not in inconsistent)
                key=(r['pair_ref'],direction,side)
                require(key not in views,'duplicate source support coordinate')
                views[key]=BoundedSupportView(r['pair_ref'],comparison,side,
                    r[selector_side+'_selector_ref'],trit,tag,d['status'],admitted)
    return views


@dataclass(frozen=True)
class SupportStage:
    """Thirteen addressed evidence coordinates, plus full retained joint state.

    Slots are chosen by an independent governed axis manifest. This
    alone is not an interpretation of the original neural Stage13.
    """
    joint: object
    bindings: tuple
    coordinates: tuple


def attach_support_stage(joint,bindings):
    _,context,_=reopen_joint(joint)
    require(type(bindings) is tuple and len(bindings)==13 and len(set(bindings))==13,
            'independent thirteen distinct source bindings required')
    views=source_support_views(context)
    require(all(k in views for k in bindings),'axis references missing in native capsule')
    return SupportStage(joint,bindings,tuple(views[k] for k in bindings))


def reopen_stage(packet):
    require(attach_support_stage(packet.joint,packet.bindings)==packet,'native support view mismatch')
    return tuple(reopen_support((v.value,v.residual)) for v in packet.coordinates)


def projection_gate(packet):
    values=reopen_stage(packet)
    held=[i for i,v in enumerate(packet.coordinates) if not v.authenticated or
          v.residual in ('both','missing')]
    return dict(status='held_independent_source_support' if held else 'bounded_support_projection_available',
        support_projection=[v.value for v in packet.coordinates],retained_residuals=[v.residual for v in packet.coordinates],
        unavailable_or_conflicting_axes=held,exact_support_reopened=True,
        original_stage13_truth_interpretation_certified=False,
        creates_semantic_authority=False,world_truth_paid=False)


def audit(context):
    views=source_support_views(context)
    from collections import Counter
    return dict(capsule_digest=context.capsule_digest,source_admission=context.admission,
        bounded_support_coordinates=len(views),
        support_residual_histogram=dict(Counter(v.residual for v in views.values())),
        authenticated_coordinates=sum(v.authenticated for v in views.values()),
        at_least_thirteen_attested_view_coordinates=sum(v.authenticated for v in views.values())>=13,
        original_stage13_source_binding_certified=False,
        native_authority_created=False,world_truth_paid=False)


def main():
    import argparse
    from pathlib import Path
    from scripts.x369_whole_source_indexed_transport import read_context
    p=argparse.ArgumentParser()
    p.add_argument('--context-expanded',type=Path,required=True)
    p.add_argument('--context-sha256',required=True)
    p.add_argument('--capsule-digest',required=True)
    p.add_argument('--family',required=True);p.add_argument('--query',required=True)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    ctx=read_context(a.context_expanded.read_bytes(),expected_sha=a.context_sha256,
        expected_capsule=a.capsule_digest,family=a.family,query=a.query)
    r=audit(ctx);result=json.dumps(r,sort_keys=True,indent=2)+'\n'
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        with a.output.open('x') as out:out.write(result)
    print(result,end='')


if __name__=='__main__':main()
