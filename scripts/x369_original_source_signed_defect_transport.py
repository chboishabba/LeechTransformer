"""Source-owned original recurrent MoE versus independent signed task action.

Compare ACTUAL, repeatable HistoryOrbitMoE transitions on exact raw history
bytes against the already-constructed original 189 signed seed mark/action.
Do not train, replay independent labels into the neural step, or assume
group equivariance of learned hidden-state evolution.

Every proposed group-transport endpoint is a CANONICAL ORIGINAL signed
seed byte image, not a claim that the learned transition must equal it.
The signed XOR residual is exact and reversible, including negative-zero
and any other floating-point bit distinctions. The two-step *total*
defect and interposed source transition are separately retained.

Original Leech-LILA: Anatolii Kornienko; DASHI extension.
"""
from hashlib import sha256

from scripts.x369_original_marked_seed_fabric import (
    marked_seed_records,mark_original_source_history,
    signed_original_task_action)

def _digest(data):
    if type(data) is not bytes:raise ValueError('raw original history bytes required')
    return sha256(data).hexdigest()

def _xor(a,b):
    if type(a) is not bytes or type(b) is not bytes or len(a)!=len(b):
        raise ValueError('equal-sized original source bytes required')
    return bytes(x^y for x,y in zip(a,b))

def source_step_exact(source_step,original,control):
    """A repeated SOURCE owner step, no imagined group/label action."""
    obs_a,actual_a=source_step(original,control)
    obs_b,actual_b=source_step(original,control)
    if type(actual_a) is not bytes or actual_a!=actual_b or obs_a!=obs_b:
        raise RuntimeError('original native source step not repeatable')
    named=dict(obs_a)
    if len(named)!=len(obs_a):
        raise ValueError('duplicate source observation field')
    return named,actual_a

def signed_action_of_control(control):
    """Native source control is (signed group index, literal tag)."""
    from models.geometric_reflection_fibre import GROUP
    if (not isinstance(control,tuple) or len(control)!=2 or
        type(control[0]) is not int or not 0<=control[0]<len(GROUP) or
        type(control[1]) is not int or not 0<=control[1]<4):
        raise ValueError('original (group_index,literal_tag) required')
    return GROUP[control[0]]

def original_source_defect_packet(actual_source,canonical_source,
                                  observations,control):
    """Raw original bytes = canonical prediction XOR retained defect."""
    delta=_xor(actual_source,canonical_source)
    recovered=_xor(canonical_source,delta)
    if recovered!=actual_source:
        raise AssertionError('original source byte recovery failed')
    return dict(
        control=control,original_source_endpoint_sha256=_digest(actual_source),
        predicted_canonical_endpoint_sha256=_digest(canonical_source),
        exact_xor_residual_hex=delta.hex(),
        signed_nonzero_residual_bytes=sum(v!=0 for v in delta),
        canonical_source_hex=canonical_source.hex(),
        actual_source_hex=actual_source.hex(),
        original_observation_sha256={
            k:sha256(repr(v).encode()).hexdigest()
            for k,v in observations.items()},
        exact_original_source_byte_reopen=True,
        learned_step_equals_independent_task_target=(delta==bytes(len(delta))),
        original_learned_equivariance_proved=False)

def reopen_source_defect(packet):
    canonical=bytes.fromhex(packet['canonical_source_hex'])
    residual=bytes.fromhex(packet['exact_xor_residual_hex'])
    recovered=_xor(canonical,residual)
    if _digest(recovered)!=packet['original_source_endpoint_sha256']:
        raise ValueError('forged original source residual or endpoint')
    return recovered

def marked_original_source_defects(marks,source_step,original_seed,
                                   controls,*,seed_budget=189,depth=2):
    """Source-first, bounded, independently replayable original defect ledger.

    For every selected original marked seed and each source control, carry
    the actual neural byte history, the independent 189-state task action,
    the canonical signed-seed prediction, and the exact XOR residual.

    Depth two measures actual F_v(F_u(h)) against the *composed* known
    target action and explicitly records intermediate source bytes.
    This is NOT an additive transport-cocycle identity: XOR is a
    bit-storage group, while learned float operations need not be linear.
    """
    from models.geometric_reflection_fibre import GROUP,IDENTITY
    if not marks['original_seed_marking_injective']:
        raise ValueError('ambiguous original source byte seed marks')
    entries=tuple(marks['records'].items())
    if (type(seed_budget) is not int or not 1<=seed_budget<=len(entries)
        or type(depth) is not int or depth not in (1,2)):
        raise ValueError('bounded original source seed/depth required')
    controls=tuple(controls)
    if not controls or len(set(controls))!=len(controls):
        raise ValueError('distinct nonempty original source controls required')
    for u in controls:signed_action_of_control(u)
    checked=0;nonzero=0;two_step_checked=0;two_step_nonzero=0
    first_nonzero=None;first_two_step_nonzero=None;rows=[]
    for seed_number,(initial,mark) in enumerate(entries[:seed_budget]):
        for u in controls:
            g=signed_action_of_control(u)
            m1,d1=signed_original_task_action(g,mark['mean'],
                                                mark['signed_residual'])
            canonical1=original_seed(m1,d1)
            observation,actual1=source_step_exact(source_step,initial,u)
            packet=original_source_defect_packet(actual1,canonical1,observation,u)
            if reopen_source_defect(packet)!=actual1:
                raise AssertionError('original source defect loss')
            checked+=1
            is_nonzero=not packet['learned_step_equals_independent_task_target']
            if is_nonzero:
                nonzero+=1
                if first_nonzero is None:
                    first_nonzero=dict(seed=seed_number,
                        original_initial_sha256=_digest(initial),
                        control=u,receipt=packet,
                        source_neural_action_mismatch=True)
            row=dict(seed=seed_number,first_control=u,
                original_initial_sha256=_digest(initial),first_defect=packet)
            if depth==2:
                # Follow SAME input twice: one actual learned history;
                # one independent original signed task history.
                observation2,actual2=source_step_exact(source_step,actual1,u)
                m2,d2=signed_original_task_action(g,m1,d1)
                canonical2=original_seed(m2,d2)
                packet2=original_source_defect_packet(
                    actual2,canonical2,observation2,u)
                row['second_defect']=packet2
                row['actual_intermediate_sha256']=_digest(actual1)
                row['composed_canonical_endpoint_sha256']=_digest(canonical2)
                row['two_step_original_byte_reopen_exact']=(
                    reopen_source_defect(packet2)==actual2)
                two_step_checked+=1
                if not packet2['learned_step_equals_independent_task_target']:
                    two_step_nonzero+=1
                    if first_two_step_nonzero is None:
                        first_two_step_nonzero=dict(
                            seed=seed_number,chronological_word=(u,u),
                            original_initial_sha256=_digest(initial),
                            original_intermediate_sha256=_digest(actual1),
                            original_second_sha256=_digest(actual2),
                            receipt=packet2)
            rows.append(row)
    return dict(original_marked_seeds_total=len(entries),
        original_marked_seeds_selected=seed_budget,
        controls=len(controls),depth=depth,
        checked_first_source_edges=checked,
        original_source_nonzero_canonical_defect_edges=nonzero,
        first_nonzero_original_neural_defect=first_nonzero,
        checked_two_step_paths=two_step_checked,
        nonzero_two_step_canonical_defects=two_step_nonzero,
        first_nonzero_two_step=first_two_step_nonzero,
        full_original_byte_defect_receipts=rows,
        original_source_history_bytes_reopened_exact=True,
        signed_action_known_independent_of_model=True,
        original_learned_equivariance_certified=False,
        source_float_space_universal_certified=False,
        original_source_checkpoint_authenticated_by_caller_only=True)

def native_marked_source_defects(model,controls,*,seed_budget=4,depth=2):
    from scripts.x369_indexed_residual_closure import native_source
    from scripts.x369_source_future_acceptance import native_future_contract
    from scripts.audit_x369_joint_action import MEANS
    from scripts.audit_x369_ordered_signed_task import TRIADS
    seed,_,_=native_source(model)
    marks=marked_seed_records(MEANS,TRIADS,seed)
    if not marks['original_seed_marking_injective'] or len(marks['records'])!=189:
        raise RuntimeError('native initial source seed labels are not unique')
    seeds,step,_=native_future_contract(model,'typed' if model.typed_consumer else 'raw')
    if set(seeds)!=set(marks['records']):
        raise RuntimeError('original source seed ownership changed')
    return marked_original_source_defects(marks,step,seed,controls,
                                         seed_budget=seed_budget,depth=depth)

def certificate():
    from models.geometric_reflection_fibre import GROUP
    means=((0,0,0),(-1,0,0))
    residuals=((0,0,0),(1,0,0))
    seed=lambda m,d:bytes((m[0]+1,m[1]+1,m[2]+1,d[0]+1,d[1]+1,d[2]+1))
    marks=marked_seed_records(means,residuals,seed)
    # Chosen zeroed group source: id and a single nonzero control.
    identity=(GROUP.index((0,0,0)),0)
    def unchanged(h,u):
        return (('actual_original_observation',h),),h
    result=marked_original_source_defects(
        marks,unchanged,seed,(identity,),seed_budget=4,depth=2)
    assert result['checked_first_source_edges']==4
    assert result['original_source_nonzero_canonical_defect_edges']==0
    # Source mismatch: alter the first original float-byte address only.
    def changed(h,u):
        return (('actual_original_observation',h),),bytes(((h[0]+1)%3,))+h[1:]
    bad=marked_original_source_defects(
        marks,changed,seed,(identity,),seed_budget=4,depth=2)
    assert bad['original_source_nonzero_canonical_defect_edges']==4
    assert bad['nonzero_two_step_canonical_defects']>0
    return dict(source_fixture_seeds=4,
        identity_canonical_transport_byte_reopen=True,
        changed_source_residual_detected=True,
        two_step_original_source_reopened=True,
        actual_native_neural_model_executed=False,
        agda_kernel_checked=False)

if __name__=='__main__':
    import json
    print(json.dumps(certificate(),sort_keys=True,default=str,indent=2))
