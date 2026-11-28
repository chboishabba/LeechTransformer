"""Three-legged pants view of existing Leech24 ambient wave chart.

Original Leech-LILA: Anatolii Kornienko; DASHI extension.
A 24D block is THREE existing punctured-nine Fourier wave8 lanes, each
with an independently retained source centre. This composes the already
constructed BIDI with the actual five-field typed pants seam criterion.
It does NOT identify E8^3 with the rootless Leech lattice, infer actual
moe recurrent-source ports, or manufacture InterfaceMatch from numerals.
"""
from dataclasses import dataclass
from hashlib import sha256
from scripts.check_x369_full_native_e8_leech_chart import (
    encode_native,decode_real,native_admission)
from scripts.x369_full_fabric_pants_source_weld import (
    BoundarySignature,matching_fields)
from scripts.certify_punctured_nine_exact import fourier_basis
import sympy as S


def pack_three_native_legs(legs,*,basis=None):
    """Full 3×9-trit native source -> 24 real wave coordinates + 3 centres."""
    if len(legs)!=3:raise ValueError('three native nine-site legs required')
    B=fourier_basis() if basis is None else basis
    pairs=tuple(encode_native(leg,B) for leg in legs)
    wave24=S.Matrix.vstack(*(wave for wave,_ in pairs))
    centres=tuple(centre for _,centre in pairs)
    return wave24,centres


def reopen_three_native_legs(wave24,centres,*,basis=None):
    if wave24.shape!=(24,1) or len(centres)!=3:
        raise ValueError('wave24 and all three independent centres required')
    B=fourier_basis() if basis is None else basis
    return tuple(native_admission(
        decode_real(wave24[8*j:8*(j+1),:],centres[j],B))
        for j in range(3))


def selected_wave_pants_seam(legs,signatures,provenance,*,basis=None):
    """Pants seam is licensed ONLY by matching original five-field
    interpreted branch signatures. Source-owned nine-trit legs and their
    centre data are never deleted by the interface quotient.
    """
    if len(signatures)!=3 or len(provenance)!=3 or any(
            type(v) is not str or not v for v in provenance):
        raise ValueError('three original signatures and source owners required')
    B=fourier_basis() if basis is None else basis
    wave,centres=pack_three_native_legs(legs,basis=B)
    originals=tuple(tuple(x) for x in legs)
    if reopen_three_native_legs(wave,centres,basis=B)!=originals:
        raise AssertionError('wave24 plus three-centre source BIDI failed')
    a,b,c=tuple(BoundarySignature.from_source(s) for s in signatures)
    ab,bc,ac=matching_fields(a,b),matching_fields(b,c),matching_fields(a,c)
    admitted=all(ab.values()) and all(bc.values()) and all(ac.values())
    return dict(status='matched_three_leg_wave_pants' if admitted
                else 'unmatched_original_interface',
                wave24=wave,centre3=centres,
                original_native_legs=originals,
                full_native_reopening_exact=True,
                five_fields=(ab,bc,ac),
                merged_five_field_signature=a.as_dict() if admitted else None,
                original_source_provenance=tuple(provenance),
                merged_source_provenance=tuple(provenance) if admitted else None,
                edge_merging_licensed=admitted,
                source_branch_channel_equality_proved=False,
                actual_leech_lattice_coset_membership_proved=False,
                original_neural_hidden_state_identified_with_chart=False,
                arbitrary_bundle_sheaf_gluing_proved=False)


def certificate():
    a=((0,)*9,(1,)*9,(-1,)*9)
    signature=dict(proposition='p',capacity=2,phase='0',wave=('a','b'),open=True)
    result=selected_wave_pants_seam(a,(signature,)*3,('interaction','appraisalA','appraisalB'))
    assert result['edge_merging_licensed']
    assert reopen_three_native_legs(result['wave24'],result['centre3'])==a
    altered=dict(signature,wave=('different','phase'))
    negative=selected_wave_pants_seam(a,(signature,altered,signature),('I','A','B'))
    assert not negative['edge_merging_licensed']
    assert reopen_three_native_legs(negative['wave24'],negative['centre3'])==a
    return dict(triads=3,ambient_wave_dimension=24,
                retained_centres=3,three_native_leg_bytes_elided=False,
                selected_seam_matched=True,
                mismatched_seam_refused=True,original_LILA_gpu_tested=False,
                actual_leech_glue_constructed=False,agda_kernel_checked=False)


if __name__=='__main__':
    import json
    print(json.dumps(certificate(),sort_keys=True))
