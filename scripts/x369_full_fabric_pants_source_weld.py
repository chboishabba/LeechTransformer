"""369 → original recurrent source → typed pants seam, exact BIDI.

Reuses original RelationalBranchCobordismGeometry five-coordinate
InterfaceMatch semantics and existing six-face source-byte endpoints.
Neither a neural logit nor an arbitrary signed coordinate is, by naming,
a pants phase/wave/capacity/openness. A SOURCE interpreter for all five
original observables must be supplied; without one, admission is blocked.

Original Leech-LILA: Anatolii Kornienko; DASHI extension.
"""
from dataclasses import dataclass
from hashlib import sha256
from collections.abc import Mapping

FIELDS=('proposition','capacity','phase','wave','open')


@dataclass(frozen=True)
class BoundarySignature:
    proposition: object
    capacity: object
    phase: object
    wave: object
    open: object

    @classmethod
    def from_source(cls,owned):
        if not isinstance(owned,Mapping) or set(owned)!=set(FIELDS):
            raise ValueError('actual source must supply exactly five typed pants fields')
        for field in FIELDS:
            if owned[field] is None:
                raise ValueError(f'missing source-owned pants boundary {field}')
        if type(owned['open']) is not bool:
            raise ValueError('source openness must be Boolean')
        if type(owned['capacity']) is not int or owned['capacity']<0:
            raise ValueError('source capacity must be nonnegative integer')
        return cls(*(owned[field] for field in FIELDS))

    def as_dict(self):
        return {field:getattr(self,field) for field in FIELDS}


def matching_fields(left,right):
    return {field: getattr(left,field)==getattr(right,field)
            for field in FIELDS}


def sha(data):
    if type(data) is not bytes:raise ValueError('exact source byte history required')
    return sha256(data).hexdigest()


def make_seam(left,right,signature_left,signature_right,*,provenance):
    """Before: two edges/two incidences. After: one signature seam/two
    incidences; BOTH original source byte states and original provenance
    remain. A signature match does NOT imply equality of source histories.
    """
    from scripts.x369_source_six_face_receipt import retain_byte_face,reopen_byte_face
    if type(left) is not bytes or type(right) is not bytes:
        raise ValueError('original source histories must be exact bytes')
    if not isinstance(provenance,tuple) or len(provenance)!=2 or any(
            type(p) is not str or not p for p in provenance):
        raise ValueError('two explicit original provenance receipts required')
    a=BoundarySignature.from_source(signature_left)
    b=BoundarySignature.from_source(signature_right)
    eq=matching_fields(a,b)
    packet=retain_byte_face(left,right)
    if reopen_byte_face(packet)!=(left,right):
        raise AssertionError('original source byte reopening failed')
    out=dict(
        status='admitted_selected_five_field_seam' if all(eq.values())
               else 'refused_unmatched_original_boundary',
        original_left_sha256=sha(left),original_right_sha256=sha(right),
        before_incidence=(('outer','outer_edge'),('inner','inner_edge')),
        after_incidence=(('outer','merged_edge'),('inner','merged_edge'))
                        if all(eq.values()) else None,
        before_edges=('outer_edge','inner_edge'),
        after_edges=('merged_edge',) if all(eq.values()) else None,
        original_provenance=provenance,merged_provenance=provenance
                            if all(eq.values()) else None,
        five_field_match=eq,merged_signature=a.as_dict()
                         if all(eq.values()) else None,
        retained_native_byte_packet=packet,
        bidirectional_reopen_exact=True,
        source_reader_future_descent_proved=False,
        actual_369_indexed_cube_equivalence_proved=False,
        original_leech_lattice_membership_proved=False,
        original_native_hip_certified=False)
    return out


def reopen_seam(receipt):
    """Always reopen original endpoints, including refused seams."""
    from scripts.x369_source_six_face_receipt import reopen_byte_face
    left,right=reopen_byte_face(receipt['retained_native_byte_packet'])
    if sha(left)!=receipt['original_left_sha256'] or sha(right)!=receipt['original_right_sha256']:
        raise ValueError('original source state SHA mismatch')
    return left,right


def weld_original_source_faces(seed,axes,step,signature_of,*,probe):
    """Apply original six routes, read original endpoints via same probe,
    then ask an EXPLICIT source-owned interpreter for five pants fields.
    No default logits→pants coercion. Probe observation is independent
    from the five-field seam and is NOT erased by a successful merge.
    """
    from scripts.x369_source_six_face_receipt import FACE_PATHS,cube_faces
    if type(seed) is not bytes or not callable(signature_of):
        raise ValueError('byte source and explicit owned pants interpreter required')
    faces=cube_faces(seed,axes,step,probe=probe)
    rows=[]
    for face_name,left_path,right_path in FACE_PATHS:
        def run(path):
            s=seed
            for part in path:
                _,s=step(s,axes[part])
            if type(s) is not bytes:raise ValueError('original source did not return bytes')
            return s
        left,right=run(left_path),run(right_path)
        observed_left,_=step(left,probe)
        observed_right,_=step(right,probe)
        a=signature_of(left,dict(observed_left))
        b=signature_of(right,dict(observed_right))
        receipt=make_seam(left,right,a,b,provenance=(
            'source:'+face_name+':'+','.join(left_path),
            'source:'+face_name+':'+','.join(right_path)))
        if reopen_seam(receipt)!=(left,right):
            raise AssertionError('source BIDI mismatch')
        if not isinstance(observed_left,(tuple,list,dict)) or not isinstance(observed_right,(tuple,list,dict)):
            raise ValueError('source observed readouts must be named')
        left_obs,right_obs=dict(observed_left),dict(observed_right)
        if set(left_obs)!=set(right_obs):
            raise ValueError('source consumer schema changes across seam')
        receipt['original_consumers_equal']={k:left_obs[k]==right_obs[k]
                                               for k in left_obs}
        receipt['face']=face_name
        receipt['source_control_paths']=(left_path,right_path)
        rows.append(receipt)
    return dict(face_count=6,rows=rows,
                admitted=sum(x['status']=='admitted_selected_five_field_seam' for x in rows),
                refused=sum(x['status']=='refused_unmatched_original_boundary' for x in rows),
                actual_original_source_interpreter_required=True,
                universal_full_hyperfabric_sufficiency=False,
                universal_original_model_equivariance=False)


def triple_pants_chain(states,signatures,provenance):
    """3-to-1 trivalent finite pants weld, two incidence merges.

    This is exactly the signature-controlled finite *seam*, not a
    proof of smooth cobordism, sheaf-local-family gluing or whole neural
    consumer equivalence. Reopen the original three BYTE histories.
    """
    from scripts.x369_source_six_face_receipt import retain_byte_face,reopen_byte_face
    if len(states)!=3 or len(signatures)!=3 or len(provenance)!=3:
        raise ValueError('three original source pants legs required')
    if any(type(v) is not bytes for v in states):
        raise ValueError('original source histories must be bytes')
    if any(type(p) is not str or not p for p in provenance):
        raise ValueError('all three source provenance labels required')
    if any(len(v)!=len(states[0]) for v in states):
        raise ValueError('equal-length source histories required for exact XOR BIDI')
    a,b,c=(BoundarySignature.from_source(s) for s in signatures)
    ab=matching_fields(a,b)
    bc=matching_fields(b,c)
    # Interface agreement is an equivalence relation on all five actual
    # source fields. Parenthesization of the merge cannot fix a mismatch.
    pairwise=(all(ab.values()),all(bc.values()),all(matching_fields(a,c).values()))
    admitted=all(pairwise)
    packet0=retain_byte_face(states[0],states[1])
    packet1=retain_byte_face(states[0],states[2])
    recovered=(states[0],reopen_byte_face(packet0)[1],reopen_byte_face(packet1)[1])
    if recovered!=tuple(states):
        raise AssertionError('three-legged source BIDI failed')
    return dict(
        status='admitted_three_leg_finite_seam' if admitted else 'refused_three_leg_boundary_mismatch',
        pairwise_match=pairwise,
        five_fields=(ab,bc,matching_fields(a,c)),
        merged_signature=a.as_dict() if admitted else None,
        edges_before=('leg_a','leg_b','leg_c'),
        edges_after=('merged_seam',) if admitted else None,
        actual_incidence_before=(('A','leg_a'),('B','leg_b'),('C','leg_c')),
        actual_incidence_after=(('A','merged_seam'),('B','merged_seam'),('C','merged_seam'))
                               if admitted else None,
        left_association_matches_right_association=admitted,
        ordered_provenance=tuple(provenance),
        retained_native_packets=(packet0,packet1),
        original_source_sha256=tuple(sha(s) for s in states),
        three_original_histories_reopen_exact=True,
        full_branch_channels_identified=False,
        path_dependent_consumer_equivalence_proved=False,
        arbitrary_local_sheaf_gluing_proved=False)


def reopen_triple_pants(packet):
    from scripts.x369_source_six_face_receipt import reopen_byte_face
    a,b=reopen_byte_face(packet['retained_native_packets'][0])
    a2,c=reopen_byte_face(packet['retained_native_packets'][1])
    if a!=a2:raise ValueError('incompatible parent centre history')
    result=(a,b,c)
    if tuple(map(sha,result))!=tuple(packet['original_source_sha256']):
        raise ValueError('three-legged original-source BIDI digest mismatch')
    return result


def certificate():
    def owns(label):
        return dict(proposition='source',capacity=2,phase=0,
                    wave=(1,0),open=True)
    a,b=bytes((0,128,255)),bytes((1,128,255))
    admitted=make_seam(a,b,owns(0),owns(1),provenance=('left','right'))
    assert admitted['status']=='admitted_selected_five_field_seam'
    assert reopen_seam(admitted)==(a,b)
    mismatch=dict(owns(0),phase=1)
    refused=make_seam(a,b,owns(0),mismatch,provenance=('left','right'))
    assert refused['status']=='refused_unmatched_original_boundary'
    assert reopen_seam(refused)==(a,b)
    return dict(selected_seam_admitted=True,phase_mismatch_refused=True,
        both_source_histories_reopen=True,
        source_native_executed=False,agda_kernel_checked=False)


if __name__=='__main__':
    import json
    print(json.dumps(certificate(),sort_keys=True))
