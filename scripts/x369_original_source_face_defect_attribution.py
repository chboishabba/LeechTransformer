"""Decompose original six controlled-source faces against original task actions.

A source cube face has two REAL ordered recurrent endpoints. The
corresponding independent signed S3 task words have two CANONICAL
original-seed endpoints. Preserve their four exact histories via two
XOR residuals. This separates intrinsic noncommuting task actions from
learned recurrent source deviations without fabricating a commuting
square or overwriting model parameters.

For each face:
  actualLeft XOR actualRight
    = canonicalLeft XOR canonicalRight XOR deltaLeft XOR deltaRight.

That is an exact BIT-STORAGE identity, not a float-linear phase defect,
a sheaf transport equation or physical 9-DOF dynamics.

Original Leech-LILA: Anatolii Kornienko; DASHI extension.
"""
from hashlib import sha256
from scripts.x369_original_marked_seed_fabric import (
    signed_original_task_action,marked_seed_records)
from scripts.x369_original_source_signed_defect_transport import (
    _xor,source_step_exact,signed_action_of_control,
    original_source_defect_packet,reopen_source_defect)
from scripts.x369_source_six_face_receipt import FACE_PATHS

def source_word_endpoint(seed,original_seed,path,axes):
    """Independent signed source task action, never learned next step."""
    mean,residual=seed['mean'],seed['signed_residual']
    for name in path:
        g=signed_action_of_control(axes[name])
        mean,residual=signed_original_task_action(g,mean,residual)
    return original_seed(mean,residual)

def actual_word_endpoint(original,step,path,axes):
    """Execute actual original recurrent step in chronological order."""
    h=original;trace=[]
    for name in path:
        obs,h=source_step_exact(step,h,axes[name])
        trace.append(dict(source_control=name,
            original_consumer_sha256={
                k:sha256(repr(v).encode()).hexdigest()
                for k,v in obs.items()},
            original_successor_sha256=sha256(h).hexdigest()))
    return h,trace

def original_source_face_decomposition(initial,seed,axes,step,original_seed):
    if type(initial) is not bytes or set(axes)!=set(('x','y','z')):
        raise ValueError('original source face requires byte seed and three named axes')
    if len(set(axes.values()))!=3:
        raise ValueError('three distinct actual source control indices required')
    for control in axes.values():signed_action_of_control(control)
    faces=[]
    for face,left_path,right_path in FACE_PATHS:
        a,trace_a=actual_word_endpoint(initial,step,left_path,axes)
        b,trace_b=actual_word_endpoint(initial,step,right_path,axes)
        ca=source_word_endpoint(seed,original_seed,left_path,axes)
        cb=source_word_endpoint(seed,original_seed,right_path,axes)
        # Reuse the exact original byte defect constructor.
        left=original_source_defect_packet(a,ca,{},tuple(left_path))
        right=original_source_defect_packet(b,cb,{},tuple(right_path))
        if reopen_source_defect(left)!=a or reopen_source_defect(right)!=b:
            raise AssertionError('source endpoint residue not reversible')
        actual_diff=_xor(a,b)
        canonical_diff=_xor(ca,cb)
        defect_diff=_xor(
            bytes.fromhex(left['exact_xor_residual_hex']),
            bytes.fromhex(right['exact_xor_residual_hex']))
        if actual_diff!=_xor(canonical_diff,defect_diff):
            raise AssertionError('original face XOR defect identity failed')
        row=dict(
            face=face,left_path=left_path,right_path=right_path,
            original_left_endpoint_sha256=sha256(a).hexdigest(),
            original_right_endpoint_sha256=sha256(b).hexdigest(),
            canonical_left_endpoint_sha256=sha256(ca).hexdigest(),
            canonical_right_endpoint_sha256=sha256(cb).hexdigest(),
            original_endpoints_agree=a==b,
            independent_task_endpoints_agree=ca==cb,
            original_left_defect=left,original_right_defect=right,
            exact_source_face_xor_residual_hex=actual_diff.hex(),
            exact_task_noncommutation_xor_hex=canonical_diff.hex(),
            exact_source_defect_difference_xor_hex=defect_diff.hex(),
            exact_face_defect_attribution_reopens=True,
            original_left_path_trace=trace_a,
            original_right_path_trace=trace_b,
            original_source_history_preserved=True)
        faces.append(row)
    return dict(
        face_count=6,faces=faces,
        real_source_commuting_faces=sum(r['original_endpoints_agree'] for r in faces),
        independently_commuting_task_faces=sum(r['independent_task_endpoints_agree'] for r in faces),
        task_noncommuting_faces=sum(not r['independent_task_endpoints_agree'] for r in faces),
        source_noncommuting_faces=sum(not r['original_endpoints_agree'] for r in faces),
        all_original_source_endpoints_reopen=True,
        all_six_face_xor_decompositions_exact=True,
        independent_task_action_not_source_step=True,
        physical_six_cube_faces_certified=False,
        learned_source_equivariance_certified=False,
        unrestricted_history_source_certified=False)

def native_source_face_decomposition(model,controls,*,seed_count=2):
    from scripts.x369_indexed_residual_closure import native_source
    from scripts.x369_source_future_acceptance import native_future_contract
    from scripts.audit_x369_joint_action import MEANS
    from scripts.audit_x369_ordered_signed_task import TRIADS
    if type(seed_count) is not int or not 1<=seed_count<=189:
        raise ValueError('original marked seed count out of bounds')
    controls=tuple(controls)
    if len(controls)<3:
        raise ValueError('three genuine original source controls required')
    axes=dict(zip(('x','y','z'),controls[:3]))
    original_seed,_,_=native_source(model)
    marks=marked_seed_records(MEANS,TRIADS,original_seed)
    if not marks['original_seed_marking_injective'] or len(marks['records'])!=189:
        raise ValueError('original source marked seed ownership mismatch')
    seeds,step,_=native_future_contract(model,'typed' if model.typed_consumer else 'raw')
    if set(seeds)!=set(marks['records']):
        raise ValueError('original marked seed and consumer sources disagree')
    results=[]
    for i,h in enumerate(seeds[:seed_count]):
        result=original_source_face_decomposition(
            h,marks['records'][h],axes,step,original_seed)
        results.append(dict(original_seed_number=i,**result))
    return dict(
        original_seed_count=seed_count,
        source_face_checks=6*seed_count,
        actual_original_source_bytes=True,
        all_source_face_residuals_reopen=True,
        original_source_face_receipts=results,
        original_checkpoint_authenticated_by_caller_only=True,
        learned_equivariance_certified=False,
        Agda_kernel_checked=False)

def certificate():
    from models.geometric_reflection_fibre import GROUP
    means=((0,0,0),(1,0,0))
    delta=((0,0,0),(0,1,0))
    source_seed=lambda m,d:bytes(v+1 for v in m+d)
    marks=marked_seed_records(means,delta,source_seed)
    original=source_seed(means[1],delta[1])
    controls=((GROUP.index((1,0,0)),0),
              (GROUP.index((0,1,0)),1),
              (GROUP.index((0,0,1)),3))
    axes=dict(zip('xyz',controls))
    def learned_identity(h,u):
        return (('actual_consumer',h),),h
    result=original_source_face_decomposition(
        original,marks['records'][original],axes,learned_identity,source_seed)
    assert result['face_count']==6
    assert result['all_six_face_xor_decompositions_exact']
    assert result['source_noncommuting_faces']==0
    # On original independent task action, noncommuting axes expose the
    # intrinsic group commutator rather than neural source defects.
    def learned_task(h,u):
        return (('actual_consumer',h),),source_word_endpoint(
            {'mean':tuple(v-1 for v in h[:3]),
             'signed_residual':tuple(v-1 for v in h[3:6])},
            source_seed,('x',),{'x':u})
    matched=original_source_face_decomposition(
        original,marks['records'][original],axes,learned_task,source_seed)
    assert matched['all_six_face_xor_decompositions_exact']
    assert matched['real_source_commuting_faces']==matched['independently_commuting_task_faces']
    return dict(six_source_faces=6,
        exact_source_vs_task_face_residuals=True,
        independent_noncommutation_not_forced_equal=True,
        original_learned_native_model_executed=False,
        Agda_kernel_checked=False)

if __name__=='__main__':
    import json
    print(json.dumps(certificate(),sort_keys=True,indent=2))
