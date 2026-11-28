"""Actual E8^3 direct-sum / Leech theta rootless obstruction.

Reuses original complete exact E8 roots and existing original theta owners.
A 24D AMBIENT block built from three E8 8D factors is not the Leech
lattice: E8^3 has 720 norm-two roots while Leech is rootless (its modular
theta prediction has q1=0). An actual Leech lattice needs verified
gluing/coset/membership, not 24=3*8 or 196560=240*819 arithmetic.
Original Leech-LILA: Anatolii Kornienko; DASHI extension.
"""
import json
import sympy as S
from scripts.check_x369_full_native_e8_leech_chart import roots_e8
from scripts.check_e8_leech_theta_finite_same_norm import (
    integer_e8_shell_counts,eisenstein_e4,leech_modular_prediction)


def triple_e8_norm_two_roots():
    roots=roots_e8()
    zero=S.zeros(8,1)
    return tuple(S.Matrix.vstack(*(root if i==slot else zero
                for i in range(3))) for slot in range(3) for root in roots)


def certificate():
    roots=triple_e8_norm_two_roots()
    if len(roots)!=720 or len(set(tuple(r) for r in roots))!=720:
        raise AssertionError('three E8 coordinate blocks must have 720 different norm-two roots')
    if any(r.shape!=(24,1) or (r.T*r)[0]!=2 for r in roots):
        raise AssertionError('original E8 ambient Euclidean norm changed')
    shell=integer_e8_shell_counts(2)
    e4=eisenstein_e4(2)
    leech=leech_modular_prediction(2)
    if shell!=e4 or shell[1]!=240 or leech[1]!=0 or leech[2]!=196560:
        raise AssertionError('original exact finite E8 / Leech modular coefficient owner diverged')
    if 3*shell[1]!=len(roots):
        raise AssertionError('E8^3 theta norm-two coefficient not the direct sum')
    return dict(three_e8_blocks=3,ambient_dimension=24,
        actual_e8_direct_sum_norm_two_roots=len(roots),
        e8_norm_two_root_count=shell[1],
        three_e8_direct_sum_q1=720,leech_modular_candidate_q1=0,
        leech_modular_candidate_q2=leech[2],
        existing_leech_actual_minimal_vector_enumeration=False,
        leech_original_rootless_theorem_required_for_isometry_obstruction=True,
        unglued_e8_cubed_is_not_rootless=True,
        leech_lattice_gluing_cosets_supplied_here=False,
        leech_lattice_membership_supplied_here=False,
        original_lila_e8_prior_equivariant_to_leech=False,
        agda_kernel_checked=False,training_steps=0)


if __name__=='__main__':
    print(json.dumps(certificate(),sort_keys=True))
