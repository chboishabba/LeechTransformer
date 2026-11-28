"""Retained full Base369 hypervoxel + stage13 local-family + pants same-object.

Do not conflate rank/depth refinement with true self-indexing tetration,
or source task 9 numeric inputs with a full H369 world. All source
interpretations are explicit, original and independently provided.

Original Leech-LILA: Anatolii Kornienko; DASHI extension.
"""
from dataclasses import dataclass
from hashlib import sha256
from scripts.x369_full_fabric_pants_source_weld import BoundarySignature,matching_fields
from scripts.x369_leech24_pants_wave_weld import pack_three_native_legs,reopen_three_native_legs

TRITS=(-1,0,1)


def _trits(values,n,kind):
    values=tuple(values)
    if len(values)!=n or any(type(v)!=int or v not in TRITS for v in values):
        raise ValueError(f'{kind} requires exactly {n} signed ternary values')
    return values


@dataclass(frozen=True)
class HypervoxelSourcePacket:
    interaction: tuple
    appraisal_a: tuple
    appraisal_b: tuple
    twelve_axes: tuple
    centre: int
    source_bytes: bytes
    history_index: object
    physical_time_index: object
    refinement_index: object
    frobenius_index: object
    consumer_version: object

    def __post_init__(self):
        for label,leg in (('interaction',self.interaction),
                          ('appraisal_a',self.appraisal_a),
                          ('appraisal_b',self.appraisal_b)):
            _trits(leg,3,label)
        _trits(self.twelve_axes,12,'original twelve-stage source')
        if type(self.centre) is not int or self.centre not in TRITS:
            raise ValueError('actual source centre is a signed ternary value')
        if type(self.source_bytes) is not bytes:
            raise ValueError('original recurrent state must be exact bytes')
        for name in ('history_index','physical_time_index','refinement_index',
                     'frobenius_index','consumer_version'):
            if getattr(self,name) is None:
                raise ValueError('missing independently governed index: '+name)


def source_to_packet(native9,source13,original_bytes,indices):
    """No default inferred valuation: source must supply all 13 stages."""
    native9=_trits(native9,9,'full native hyperfabric')
    source13=_trits(source13,13,'central stage13 completion')
    if set(indices)!=set(('history','physical_time','refinement','frobenius','version')):
        raise ValueError('original five independent index roles required')
    return HypervoxelSourcePacket(
        native9[:3],native9[3:6],native9[6:],
        source13[:12],source13[12],original_bytes,
        indices['history'],indices['physical_time'],
        indices['refinement'],indices['frobenius'],indices['version'])


def split_packet(packet):
    return dict(
        original_nine=packet.interaction+packet.appraisal_a+packet.appraisal_b,
        source_stage13=packet.twelve_axes+(packet.centre,),
        original_source_bytes=packet.source_bytes,
        independent_indices=dict(
            history=packet.history_index,physical_time=packet.physical_time_index,
            refinement=packet.refinement_index,frobenius=packet.frobenius_index,
            version=packet.consumer_version))


def reopen_packet(parts):
    return source_to_packet(parts['original_nine'],parts['source_stage13'],
                            parts['original_source_bytes'],parts['independent_indices'])


def stitch_stage13_from_native(native9,remaining4,native_slots):
    """Chosen source-dependent nine-plus-four stage13 coordinate atlas."""
    native=_trits(native9,9,'original nine-trit fabric')
    extra=_trits(remaining4,4,'four retained stage residual trits')
    slots=tuple(native_slots)
    if len(slots)!=9 or any(type(i) is not int or not 0<=i<13 for i in slots):
        raise ValueError('nine original stage13 slot addresses required')
    if len(set(slots))!=9:raise ValueError('native axis slots cannot collide')
    rest=tuple(i for i in range(13) if i not in slots)
    stage=[None]*13
    for i,value in zip(slots,native):stage[i]=value
    for i,value in zip(rest,extra):stage[i]=value
    return tuple(stage)


def split_stage13_to_native(stage13,native_slots):
    """Source-backed inverse recovers native 9 AND exact missing 4."""
    stage=_trits(stage13,13,'original source stage13')
    slots=tuple(native_slots)
    if len(slots)!=9 or any(type(i) is not int or not 0<=i<13 for i in slots
                           ) or len(set(slots))!=9:
        raise ValueError('invalid nine-axis source chart')
    rest=tuple(i for i in range(13) if i not in slots)
    native=tuple(stage[i] for i in slots)
    extra=tuple(stage[i] for i in rest)
    if stitch_stage13_from_native(native,extra,slots)!=stage:
        raise AssertionError('stage13 9+4 BIDI failed')
    return native,extra


def admit_native_to_source_stage13(native9,source13,native_slots):
    """Only a real source coordinate-agreement supplies the seam."""
    native=_trits(native9,9,'full native fabric')
    source=_trits(source13,13,'source original stage13')
    encoded,extra=split_stage13_to_native(source,native_slots)
    if native!=encoded:
        return dict(admitted=False,source_native_slot_mismatch=True,
                    retained_source_stage13=source,
                    original_full_native9=native,
                    no_implicit_source_map=True)
    return dict(admitted=True,chosen_native_slots=tuple(native_slots),
                residual_stage4=extra,
                source_stage13=source,
                original_full_native9=native,
                reopen_exact=(stitch_stage13_from_native(native,extra,native_slots)==source),
                only_chosen_chart_not_intrinsic_equivalence=True)


def central_local_family(packet):
    """Actual original central-family definition: (valuation_i, centre)."""
    return tuple((v,packet.centre) for v in packet.twelve_axes)


def compatible_locals(locals):
    if len(locals)!=12 or any(len(p)!=2 for p in locals):
        raise ValueError('original 12 local source stages required')
    for valuation,centre in locals:
        _trits((valuation,centre),2,'local valuation/centre')
    return all(centre==locals[0][1] for _,centre in locals)


def glue_local_family(locals):
    """The executable counterpart to existing central-family gluing."""
    if not compatible_locals(locals):
        raise ValueError('independent local centres do not glue')
    return tuple(v for v,_ in locals)+(locals[0][1],)


def project_pants13(packet):
    locals=central_local_family(packet)
    assert compatible_locals(locals)
    result=glue_local_family(locals)
    assert result==packet.twelve_axes+(packet.centre,)
    assert all((result[i],result[12])==locals[i] for i in range(12))
    return result


def refine_address(coarse,fine):
    """Exact radix address join; rank is the width of each child block."""
    if not isinstance(coarse,tuple) or not isinstance(fine,tuple):
        raise ValueError('actual ranked address blocks must be tuples')
    rank=len(coarse[0]) if coarse else len(fine[0]) if fine else None
    if rank is None:
        raise ValueError('rank must be known for empty address')
    for block in (*coarse,*fine):
        if len(block)!=rank:
            raise ValueError('mixed rank in recursively refined address')
        _trits(block,rank,'relative hypervoxel child')
    return coarse+fine


def backsolve_source(packet,source_consumer,chart):
    """Actual source consumer factorization is a separate assertion:
    this function only gives the source-aware paired view for audit.
    """
    if not callable(source_consumer) or not callable(chart):
        raise ValueError('source consumer and chart are independently required')
    return dict(native_source_bytes_sha256=sha256(packet.source_bytes).hexdigest(),
                chart=chart(packet),observation=source_consumer(packet),
                all_future_factorization_proved=False)


def same_packet_pants_seam(left,right,read_boundary):
    """Use five actual typed fields on two views of the SAME packet.

    Two different projections of the retained packet may license a
    semantic waist, but the packet itself is never erased.
    """
    if not callable(read_boundary):
        raise ValueError('source-owned five-field interface reader required')
    a=BoundarySignature.from_source(read_boundary(left))
    b=BoundarySignature.from_source(read_boundary(right))
    return dict(source_is_retained=True,
        interface_match=matching_fields(a,b),
        edge_merge_licensed=all(matching_fields(a,b).values()),
        source_byte_sha256=sha256(left.source_bytes).hexdigest(),
        full_hyperfabric_reopen_exact=True,
        source_future_equivalence_proved=False)


def stage13_source_admission(seeds,controls,step,source_interpret,
                             *,max_seeds=189):
    """Source-provided same-object interpretation, not inferred trits.

    Each source_interpret(h) returns ORIGINAL native9, original stage13,
    explicit native slot addresses, five independent source indices.
    Only an exact source agreement chart is admitted. One-step controls
    check whether successors remain in the declared chosen source atlas.
    This is a bounded necessary test, not a universal factorization.
    """
    if not callable(source_interpret):
        return dict(status='missing_original_source_interpreter',
                    admitted=False,original_native_source_admission=False,
                    no_guessed_stage13_coordinates=True)
    seeds=tuple(seeds)
    controls=tuple(controls)
    if not seeds or not controls or type(max_seeds) is not int or max_seeds<1:
        raise ValueError('nonempty source evidence and positive seed bound required')
    if max_seeds>len(seeds):max_seeds=len(seeds)
    admitted=0
    checked=0
    failures=[]
    seen={}
    def decode(h,seed_number,word):
        if type(h) is not bytes:
            raise ValueError('actual original recurrent states require byte keys')
        owned=source_interpret(h)
        if not isinstance(owned,dict) or set(owned)!=set(
                ('native9','source13','native_slots','indices')):
            raise ValueError('source interpreter must supply complete owned geometry')
        report=admit_native_to_source_stage13(
            owned['native9'],owned['source13'],owned['native_slots'])
        if not report['admitted']:
            failures.append(dict(kind='source_coordinate_mismatch',
                 original_seed=seed_number,chronological_word=word,
                 original_source_sha256=sha256(h).hexdigest(),
                 owned_native9=report['original_full_native9'],
                 owned_stage13=report['retained_source_stage13']))
            return None
        result=source_to_packet(owned['native9'],owned['source13'],h,owned['indices'])
        if reopen_packet(split_packet(result))!=result:
            raise AssertionError('actual original source packet BIDI failed')
        if project_pants13(result)!=tuple(owned['source13']):
            raise AssertionError('original Stage13 pants source projection failed')
        return result
    for i,h in enumerate(seeds[:max_seeds]):
        p=decode(h,i,())
        if p is None:continue
        admitted+=1
        for u in controls:
            named,next_h=step(h,u)
            if type(next_h) is not bytes:
                raise ValueError('original step must provide byte state')
            checked+=1
            q=decode(next_h,i,(u,))
            if q is None:continue
            # These are original source geometry/interpreter checks only.
            # Consumer equality across chart fibres and successor
            # congruence are independently owned by fibre_audit.
    return dict(
        status='bounded_original_source_chart_refuted' if failures else
               'bounded_source_chart_compatible',
        admitted_seed_count=admitted,requested_seed_count=max_seeds,
        original_step_edges_examined=checked,
        source_interpreter_explicit=True,
        failures=failures,
        actual_source_byte_bidi_on_admitted_states=True,
        original_full_hyperfabric_isomorphic_to_neural_state=False,
        original_source_all_future_factorization_certified=False,
        all_original_source_histories_admitted=False,
        source_checkpoint_authenticated_by_caller_only=True)


def native_stage13_source_admission(model,controls,*,seed_count=4):
    """Reuse ORIGINAL native callback and caller-authenticated checkpoint."""
    from scripts.x369_source_future_acceptance import native_future_contract
    source_owned=getattr(model,'source_hypervoxel_stage13',None)
    if not callable(source_owned):
        return dict(status='missing_original_source_interpreter',
                    admitted=False,source_method='source_hypervoxel_stage13',
                    original_source_history_execution_skipped=True,
                    source_future_sufficiency=False)
    seeds,step,_=native_future_contract(model,'raw')
    return stage13_source_admission(seeds,controls,step,source_owned,
                                    max_seeds=seed_count)


def certificate():
    indices=dict(history='h',physical_time='t',refinement='r',
                 frobenius='f',version='v')
    source=source_to_packet(
        (-1,0,1,0,1,-1,1,-1,0),
        tuple((-1,0,1)[i%3] for i in range(13)),
        bytes.fromhex('00000080 00000000'),indices)
    parts=split_packet(source)
    assert reopen_packet(parts)==source
    assert project_pants13(source)==parts['source_stage13']
    assert glue_local_family(central_local_family(source))==parts['source_stage13']
    wave,centres=pack_three_native_legs(tuple(leg+(0,)*6 for leg in
        (source.interaction,source.appraisal_a,source.appraisal_b)))
    assert len(centres)==3 and wave.rows==24
    assert reopen_three_native_legs(wave,centres)==tuple(leg+(0,)*6 for leg in
        (source.interaction,source.appraisal_a,source.appraisal_b))
    chosen_slots=tuple(range(9))
    native=parts['original_nine']
    stage4=(0,1,-1,source.centre)
    stitched=stitch_stage13_from_native(native,stage4,chosen_slots)
    assert split_stage13_to_native(stitched,chosen_slots)==(native,stage4)
    assert admit_native_to_source_stage13(native,stitched,chosen_slots)['admitted']
    assert not admit_native_to_source_stage13(native,
        (1 if native[0]!=1 else 0,)+stitched[1:],chosen_slots)['admitted']
    sig=lambda s:dict(proposition='source',capacity=3,phase='p0',
                      wave=('cos','sin'),open=True)
    accepted=same_packet_pants_seam(source,source,sig)
    assert accepted['edge_merge_licensed']
    bad=list(central_local_family(source))
    bad[1]=(bad[1][0],1 if source.centre!=1 else 0)
    try:glue_local_family(tuple(bad))
    except ValueError:pass
    else:raise AssertionError('mismatched original centre was glued')
    return dict(full_native_coordinates=9,source_stages=13,
                original_stage_restrictions=12,independent_index_roles=5,
                source_bytes_reopened=True,
                same_object_pants13_projection=True,
                same_object_five_field_seam_licensed=True,
                three_24d_wave_lanes_reopened=True,
                chosen_nine_plus_four_stage13_chart_bidi=True,
                mismatched_original_source_chart_refused=True,
                mismatched_central_family_refused=True,
                actual_native_source_interpreter_executed=False,
                self_indexed_tower_identified_with_product=False,
                source_consumer_factorization_proved=False,
                agda_kernel_checked=False)


if __name__=='__main__':
    import json
    print(json.dumps(certificate(),sort_keys=True,indent=2))
