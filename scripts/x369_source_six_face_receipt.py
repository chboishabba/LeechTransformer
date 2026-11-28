"""Actual source-input three-control cube: six face max-cut audit.

This is NOT the original indexed H07 cube's universal face theorem.
It realizes its six square-route SHAPE on actual source transitions and
tests whether each proposed native face commutes. No assumption that
noncommuting control actions, neural routing, or indexed time commute.
Original Leech-LILA: Anatolii Kornienko; DASHI extension.
"""
from hashlib import sha256
from itertools import permutations
import json


# Six actual original cube faces, two directions for each square.
# Third-axis prefix (when present) is applied BEFORE the square.
# x=0/1: yz; y=0/1: xz; z=0/1: xy.
FACE_PATHS = (
    ('xy0', ('x','y'), ('y','x')),
    ('xy1', ('z','x','y'), ('z','y','x')),
    ('xz0', ('x','z'), ('z','x')),
    ('xz1', ('y','x','z'), ('y','z','x')),
    ('yz0', ('y','z'), ('z','y')),
    ('yz1', ('x','y','z'), ('x','z','y')),
)

def fingerprint(x):
    if type(x) is bytes:return sha256(x).hexdigest()
    return sha256(repr(x).encode()).hexdigest()

def cube_faces(seed,axes,step,*,probe=None):
    """Runs source step, never supplied target labels or symbolic action.

    The optional same-control probe interrogates the original endpoint
    consumer at each face endpoint. It is NOT an extra cube edge.
    """
    if set(axes)!=set(('x','y','z')) or len(axes)!=3 or len(set(axes.values()))!=3:
        raise ValueError('three distinct original source controls required')
    seen={}
    def rollout(path):
        key=tuple(path)
        if key not in seen:
            h=seed;trace=[]
            for name in key:
                obs,h=step(h,axes[name])
                trace.append(dict(control=name,observation_sha256=fingerprint(obs)))
            seen[key]=(h,trace)
        return seen[key]
    faces=[]
    for name,left,right in FACE_PATHS:
        a,trace_a=rollout(left);b,trace_b=rollout(right)
        same=a==b
        row=dict(face=name,
                 left_path=list(left),right_path=list(right),
                 left_endpoint_sha256=fingerprint(a),
                 right_endpoint_sha256=fingerprint(b),
                 endpoint_bytes_equal=same,
                 left_edge_trace=trace_a,right_edge_trace=trace_b)
        if probe is not None:
            oa,_=step(a,probe);ob,_=step(b,probe)
            lhs,rhs=dict(oa),dict(ob)
            if set(lhs)!=set(rhs):raise ValueError('source consumer schema changes across face')
            row['probe_control']=repr(probe)
            row['consumer_equal']={k:lhs[k]==rhs[k] for k in lhs}
            row['left_consumer_sha256']={k:fingerprint(lhs[k]) for k in lhs}
            row['right_consumer_sha256']={k:fingerprint(rhs[k]) for k in rhs}
        faces.append(row)
    return dict(faces=faces,all_six_endpoints_commute=all(f['endpoint_bytes_equal'] for f in faces),
                no_source_commutation_assumed=True,
                source_operator_precision_certified=False,
                hyperfabric_9d_physical_dynamics_certified=False)


def native_face_audit(model,*,controls,seed_count=3):
    """Actual raw history-state source. Caller must authenticate checkpoint."""
    from itertools import product
    from scripts.x369_source_future_acceptance import native_future_contract
    from scripts.x369_source_acceptance import guard_controls
    from scripts.audit_x369_joint_action import MEANS
    from scripts.audit_x369_ordered_signed_task import TRIADS
    if type(seed_count)!=int or not 1<=seed_count<=189:
        raise ValueError('positive bounded original-source seed count')
    if len(controls)!=3 or len(set(controls))!=3:
        raise ValueError('three distinct controls')
    seeds,step,_=native_future_contract(model,'typed' if model.typed_consumer else 'raw')
    results=[]
    axes=dict(zip('xyz',controls))
    for i,h in enumerate(seeds[:seed_count]):
        result=cube_faces(h,axes,step,probe=controls[0])
        results.append(dict(seed=i,**result))
    return dict(source_seed_count=len(seeds),audited_seeds=len(results),
                original_native_source_states=True,
                face_comparisons=6*len(results),
                all_faces_equal=all(x['all_six_endpoints_commute'] for x in results),
                receipts=results,
                unrestricted_history_source_certified=False,
                full_h07_indexed_physical_face_certified=False,
                agda_kernel_checked=False)


def certificate():
    def plus(s,u):
        obs=(('value',s),)
        return obs,s+u
    good=cube_faces(0,{'x':1,'y':2,'z':3},plus,probe=1)
    assert good['all_six_endpoints_commute']
    assert len(good['faces'])==6
    def affine(s,u):
        obs=(('value',s),)
        if u=='x':return obs,s+1
        if u=='y':return obs,2*s
        if u=='z':return obs,s-3
        raise ValueError('control')
    bad=cube_faces(0,{'x':'x','y':'y','z':'z'},affine,probe='x')
    assert not bad['all_six_endpoints_commute']
    assert any(not all(x.values()) for x in (f['consumer_equal'] for f in bad['faces']))
    assert len({f['face'] for f in bad['faces']})==6
    return dict(faces=6,positive_additive=True,negative_affine=True,
                native_executed=False)


if __name__=='__main__':
    print(json.dumps(certificate(),sort_keys=True))
