"""Original signed-seed ↔ Base369 chosen nine-trit marked subcarrier.

The original native_source(model) (not a new network) provides source seed
bytes for 7 source means × 27 signed residual triads. We attach a typed
label to precisely those 189 ORIGINAL INITIAL recurrent history bytes.
No synthetic stage13, pants signature, training label or future semantic
state is inferred. An unrecognized post-step history remains unadmitted,
with original bytes always retained.

Original Leech-LILA: Anatolii Kornienko; DASHI extension.
"""
from hashlib import sha256
from itertools import product

SOURCE_MARK_COUNT=189
INDEX_NAMES=('history','physical_time','refinement','frobenius','version')
MARK_TYPES=('original_mean','original_signed_residual','native_input9')

def _triad(v):
    v=tuple(v)
    if len(v)!=3 or any(type(x) is not int or x not in (-1,0,1) for x in v):
        raise ValueError('signed three-axis source trits required')
    return v

def decode_canonical_original_seed_bytes(h,*,experts,width,dtype):
    """REVERSE exact original signed_seed() storage, not learned decoding.

    Original signed_seed places m+d and m-d in expert0's paired
    first three channels and zeroes every other float. This accepts ONLY
    that canonical byte image. The seven source means are origin and
    signed coordinate axes; the 27 residual triads are independent.
    It does not interpret a post-transition history as a task seed.
    """
    import struct
    if type(h) is not bytes or dtype not in ('float32','float64'):
        raise ValueError('original raw bytes and exact float storage type required')
    if type(experts) is not int or type(width) is not int or (
        experts<2 or width<3 or width%3):
        raise ValueError('source original signed_seed shape required')
    scalar='f' if dtype=='float32' else 'd'
    count=experts*2*width
    expected_size=count*struct.calcsize(scalar)
    if len(h)!=expected_size:
        raise ValueError('original source history byte size mismatch')
    values=struct.unpack('='+str(count)+scalar,h)
    a=values[:3]
    b=values[width:width+3]
    from math import isfinite
    if not all(isfinite(x) for x in values):
        return None
    m=tuple((x+y)/2 for x,y in zip(a,b))
    d=tuple((x-y)/2 for x,y in zip(a,b))
    if any(z not in (-1,0,1) for z in (*m,*d)):
        return None
    m=tuple(map(int,m));d=tuple(map(int,d))
    if sum(x!=0 for x in m)>1:
        return None
    reconstructed=[0.0]*count
    for i in range(3):
        reconstructed[i]=float(m[i]+d[i])
        reconstructed[width+i]=float(m[i]-d[i])
    expected=struct.pack('='+str(count)+scalar,*reconstructed)
    if expected!=h:
        return None
    return dict(mean=m,signed_residual=d,native_input9=m+d+(0,0,0),
                original_canonical_signed_seed_bytes_exact=True,
                original_source_semantics_beyond_initial_seed=False)


def marked_seed_records(means,triads,original_seed):
    """Exact original seed byte map, with no heuristic inverse decoder.

    The seed producer MUST be the already-owned native_source(...).seed,
    or an explicitly declared deterministic test fixture. Byte collisions
    between differently labelled seeds are decisive chart obstructions.
    """
    if not callable(original_seed):raise ValueError('original seed producer required')
    mean=tuple(_triad(m) for m in means)
    delta=tuple(_triad(d) for d in triads)
    if not mean or not delta or len(set(mean))!=len(mean) or len(set(delta))!=len(delta):
        raise ValueError('distinct nonempty actual signed source labels required')
    by_bytes={};collisions=[]
    for i,(m,d) in enumerate(product(mean,delta)):
        original=original_seed(m,d)
        if type(original) is not bytes or not original:
            raise ValueError('actual source seed must return nonempty recurrent bytes')
        full=m+d+(0,0,0)
        if original in by_bytes:
            prior=by_bytes[original]
            if prior['native_input9']!=full:
                collisions.append(dict(first_mean=prior['mean'],
                    first_signed=prior['signed_residual'],
                    second_mean=m,second_signed=d,
                    original_source_sha256=sha256(original).hexdigest()))
            continue
        by_bytes[original]=dict(original_seed_index=i,
            original_source_sha256=sha256(original).hexdigest(),
            original_source_length=len(original),
            mean=m,signed_residual=d,native_input9=full,
            source_owner='original_native_source_seed_function',
            input_zero_padding_exact=True,
            other_369_hyperfabric_fibres_inferred=False,
            stage13_source_semantics_inferred=False)
    return dict(
        attempted_seed_count=len(mean)*len(delta),
        distinct_original_byte_seeds=len(by_bytes),
        original_source_byte_aliases=collisions,
        original_seed_marking_injective=not collisions and
           len(by_bytes)==len(mean)*len(delta),
        records=by_bytes,
        unrestricted_future_history_decoder_certified=False)

def mark_original_source_history(h,marks):
    """Only original marked seed byte histories get a chosen source label."""
    if type(h) is not bytes:raise ValueError('original state bytes required')
    receipt=marks['records'].get(h)
    if receipt is None:
        return dict(status='unmarked_source_history',
            original_source_sha256=sha256(h).hexdigest(),
            original_source_bytes_retained=True,
            original_native_nine_recovered=False)
    return dict(status='marked_original_source_seed',
        **receipt,
        original_native_nine_recovered=True,
        original_source_bytes_retained=True)

def seed_marker_future_audit(marks,controls,step,*,max_seeds=189):
    """Bounded native transitions; no target/action register in recurrence.

    Source input marks attach only to initial histories, not to arbitrary
    successors. We report whether a successor was also a genuine marked
    initial state, never assign the predecessor's triads to it.
    """
    if not marks['original_seed_marking_injective']:
        raise ValueError('original source seed alias prevents this chart')
    controls=tuple(controls)
    if not controls or len(set(controls))!=len(controls):
        raise ValueError('distinct nonempty original source controls required')
    if type(max_seeds) is not int or not 1<=max_seeds<=len(marks['records']):
        raise ValueError('seed budget out of original marked range')
    entries=tuple(marks['records'].items())[:max_seeds]
    transitions=0;unmarked=0;marked=0;witness=None
    for i,(h,receipt) in enumerate(entries):
        if mark_original_source_history(h,marks)['native_input9']!=receipt['native_input9']:
            raise AssertionError('original signed source seed lookup changed')
        for u in controls:
            observations,n=step(h,u)
            if type(n) is not bytes:
                raise ValueError('original controlled step must return bytes')
            transitions+=1
            found=mark_original_source_history(n,marks)
            if found['status']=='unmarked_source_history':
                unmarked+=1
                if witness is None:
                    witness=dict(seed=i,original_control=u,
                        old_source_sha256=sha256(h).hexdigest(),
                        successor_source_sha256=sha256(n).hexdigest(),
                        original_successor_bytes_retained=True)
            else:marked+=1
    return dict(seed_scope=max_seeds,controls=len(controls),
        source_steps=transitions,successors_also_marked=marked,
        successors_unmarked=unmarked,
        first_unmarked_successor=witness,
        source_original_marked_initials_proved=True,
        successor_native_semantics_automatically_inherited=False,
        original_all_futures_factorized=False)

def native_seed_marked_audit(model,controls,*,seed_budget=189):
    from scripts.x369_indexed_residual_closure import native_source
    from scripts.audit_x369_joint_action import MEANS
    from scripts.audit_x369_ordered_signed_task import TRIADS
    if len(MEANS)*len(TRIADS)!=SOURCE_MARK_COUNT:
        raise RuntimeError('original signed task carrier changed')
    seed,_,_=native_source(model)
    marks=marked_seed_records(MEANS,TRIADS,seed)
    dtype=str(next(model.parameters()).dtype).removeprefix('torch.')
    for h,original in marks['records'].items():
        decoded=decode_canonical_original_seed_bytes(h,
            experts=model.experts,width=model.width,dtype=dtype)
        if decoded is None or decoded['native_input9']!=original['native_input9'] or (
            decoded['mean']!=original['mean'] or
            decoded['signed_residual']!=original['signed_residual']):
            raise RuntimeError('original native signed_seed storage is not exact canonical input witness')
    if not marks['original_seed_marking_injective'] or (
        marks['distinct_original_byte_seeds']!=SOURCE_MARK_COUNT):
        raise RuntimeError('original native source seed bytes alias')
    # Use the SAME actual observer/step already checked repeatable by the
    # immutable original x369 native_future_contract owner.
    from scripts.x369_source_future_acceptance import native_future_contract
    seeds,step,_=native_future_contract(model,'raw')
    if set(seeds)!=set(marks['records']):
        raise RuntimeError('original native source seed ownership mismatch')
    out=seed_marker_future_audit(marks,controls,step,max_seeds=seed_budget)
    out.update(original_seed_cardinality=SOURCE_MARK_COUNT,
        original_full_fabric_cardinality=3**9,
        marked_source_seed_fraction='189/19683',
        checkpoint_authenticated_by_caller_only=True,
        nine_ternary_source_input_labels_verified=True,
        canonical_original_signed_seed_byte_decode_verified=True,
        original_stage13_interpreter_supplied=False,
        source_training_provenance_authenticated=False)
    return out

def certificate():
    means=((0,0,0),(-1,0,0))
    triads=((0,0,0),(1,0,0))
    def seed(m,d):
        return bytes((m[0]+1,d[0]+1))
    marks=marked_seed_records(means,triads,seed)
    assert marks['original_seed_marking_injective']
    assert marks['distinct_original_byte_seeds']==4
    h=seed(means[1],triads[1])
    assert mark_original_source_history(h,marks)['native_input9']==means[1]+triads[1]+(0,0,0)
    def step(h,u):return (('original_observer',h),),bytes(((h[0]+1)%3,h[1]))
    receipt=seed_marker_future_audit(marks,('step',),step,max_seeds=4)
    assert receipt['source_steps']==4
    assert receipt['successors_unmarked']>0
    return dict(exact_fixture_initials=4,
        source_byte_initial_label_bidi=True,
        unrelated_successor_not_assigned_source_semantics=True,
        original_native_model_executed=False,
        universal_source_semantic_decode=False,
        agda_kernel_checked=False)

if __name__=='__main__':
    import json
    print(json.dumps(certificate(),indent=2,sort_keys=True))
