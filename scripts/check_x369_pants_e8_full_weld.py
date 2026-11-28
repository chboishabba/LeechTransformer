"""One deterministic source-free acceptance receipt for the X369 pants/E8 weld.

This DOES NOT import or execute original neural checkpoint and does not
turn chosen wave charts into a literal Leech lattice. Source ownership:
Anatolii Kornienko original Leech-LILA; DASHI extensions.
"""
import json
from scripts.x369_full_fabric_pants_source_weld import (
    certificate as byte_pants, make_seam, reopen_seam,
    triple_pants_chain,reopen_triple_pants)
from scripts.x369_leech24_pants_wave_weld import (
    certificate as wave_pants,selected_wave_pants_seam,
    reopen_three_native_legs)
from scripts.check_x369_full_native_e8_leech_chart import (
    fourier_basis)
import sympy as S
from scripts.x369_original_marked_seed_fabric import (
    certificate as marked_source_certificate)
from scripts.check_x369_original_marked_fabric_signed_graph import (
    certificate as signed_graph_certificate)
from scripts.x369_original_source_signed_defect_transport import (
    certificate as signed_defect_certificate)
from scripts.x369_original_source_word_residual_weld import (
    certificate as original_word_residual_certificate)
from scripts.x369_original_source_consumer_coarse_fine_maxcut import (
    certificate as source_consumer_coarse_fine_certificate)
from scripts.x369_original_source_minimal_residual_repair import (
    certificate as source_minimal_residual_certificate)
from scripts.x369_original_source_face_defect_attribution import (
    certificate as six_face_attribution_certificate)
from scripts.x369_hypervoxel_stage13_pants_same_object import (
    certificate as stage13_hypervoxel_certificate,
    stitch_stage13_from_native,split_stage13_to_native,
    admit_native_to_source_stage13)


def certify():
    B=fourier_basis()
    P=B*B.T
    ones=S.ones(9,1)
    e0=S.Matrix([[1]+[0]*8])
    if B.T*B!=S.eye(8) or P!=S.eye(9)-ones*ones.T/9:
        raise AssertionError('original pointed-nine wave geometry not exact')
    if P+ones*(e0-e0*P)!=S.eye(9):
        raise AssertionError('full original nine-signal inverse failed')
    byte=byte_pants()
    wave=wave_pants()
    stage13=stage13_hypervoxel_certificate()
    marked=marked_source_certificate()
    signed_graph=signed_graph_certificate()
    neural_defects=signed_defect_certificate()
    original_words=original_word_residual_certificate()
    source_consumers=source_consumer_coarse_fine_certificate()
    minimal_residual=source_minimal_residual_certificate()
    face_attribution=six_face_attribution_certificate()
    if not (marked['source_byte_initial_label_bidi']
            and signed_graph['every_full_native_P3_edge_preserved_by_monomial_proof']
            and neural_defects['two_step_original_source_reopened']
            and original_words['finite_original_arbitrary_word_bidi']
            and original_words['independent_actual_source_replay']
            and source_consumers['finite_original_coarse_mark_refuted']
            and source_consumers['source_negative_independently_replayed']
            and minimal_residual['minimum_extra_bits_per_fibre_worst_case']==1
            and face_attribution['exact_source_vs_task_face_residuals']):
        raise AssertionError('original source signed geometry residual weld incomplete')
    if not (stage13['same_object_pants13_projection']
            and stage13['chosen_nine_plus_four_stage13_chart_bidi']
            and stage13['mismatched_original_source_chart_refused']):
        raise AssertionError('native hypervoxel 9+4 Stage13 admission proof failed')
    native=(-1,0,1,0,1,-1,1,-1,0)
    extra=(0,1,-1,0)
    for slots in (tuple(range(9)),tuple(range(4,13)),
                  (0,2,4,6,8,10,12,1,3)):
        original13=stitch_stage13_from_native(native,extra,slots)
        if split_stage13_to_native(original13,slots)!=(native,extra):
            raise AssertionError('chosen hypervoxel source atlas BIDI failed')
        if not admit_native_to_source_stage13(native,original13,slots)['reopen_exact']:
            raise AssertionError('native source atlas not admitted')
    # Real native model source mapping remains a separate owner; this
    # is a selected finite nine-trit/13-stage chart construction.
    f={'proposition':'p','capacity':1,'phase':'p0',
       'wave':('dc','rotating'), 'open':True}
    histories=(bytes.fromhex('00000080'),bytes.fromhex('00000000'),
               bytes.fromhex('01000000'))
    triple=triple_pants_chain(histories,(f,f,f),('interaction','A','B'))
    if triple['status']!='admitted_three_leg_finite_seam':
        raise AssertionError('original five-field three-leg source seam failed')
    if reopen_triple_pants(triple)!=histories:
        raise AssertionError('original three-history BIDI failed')
    legs=((1,0,-1,0,0,0,0,0,0),
          (0,0,0,1,0,-1,0,0,0),
          (0,0,0,0,0,0,-1,0,1))
    joined=selected_wave_pants_seam(legs,(f,f,f),('interaction','A','B'),basis=B)
    if joined['status']!='matched_three_leg_wave_pants':
        raise AssertionError('full 369 typed-signature native wave seam failed')
    if reopen_three_native_legs(joined['wave24'],joined['centre3'],basis=B)!=legs:
        raise AssertionError('actual wave24 centre3 native BIDI failed')
    incompatible=dict(f,phase='different')
    failed=selected_wave_pants_seam(legs,(f,incompatible,f),('I','A','B'),basis=B)
    if failed['edge_merging_licensed']:
        raise AssertionError('original unmatched phase silently glued')
    return dict(kind='x369_pants_e8_source_free_exact',
        sympy_version=S.__version__,
        original_189_source_seed_marker_fixture_bidi=True,
        original_signed_source_group_matrices=signed_graph['source_9x9_signed_monomial_witnesses'],
        source_12_group_compositions=signed_graph['group_composition_matrices_checked'],
        source_nonwrapping_p3_nine_edges=signed_graph['full_nonwrapping_P3_nine_edges'],
        actual_source_two_step_xor_bidi_fixture=True,
        actual_source_arbitrary_word_bidi_fixture=True,
        source_original_mark_only_refuted_on_closed_fixture=True,
        source_original_consumer_negative_replayed=True,
        original_source_minimal_residual_one_bit_fixture=True,
        six_face_learned_vs_task_defect_attribution_fixture=True,
        original_heterogeneous_five_field_matching=True,
        three_actual_source_histories_reopened=True,
        original_source_byte_signed_zero_preserved=True,
        all_three_native_nine_trit_legs_reopened=True,
        original_stage13_hypervoxel_packet_reopened=True,
        chosen_nine_plus_four_stage13_inverse=True,
        source_stage13_coordinate_admission_checked=True,
        original_punctured_nine_fourier_inverse_matrix_exact=True,
        retained_centres_per_24d_block=3,
        unmatched_original_phase_refused=True,
        original_native_checkpoint_authenticated=False,
        learned_source_transition_or_consumer_factorization_proved=False,
        true_leech_lattice_gluing_cosets_proved=False,
        unrestricted_source_13_stage_sheaf_gluing_proved=False,
        physical_nine_dof_gravity_proved=False,
        agda_kernel_checked=False,
        original_neural_source_executed=False)


if __name__=='__main__':
    print(json.dumps(certify(),sort_keys=True,indent=2))
