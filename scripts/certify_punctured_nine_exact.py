"""Exact symbolic certificate: punctured F3² nine-sheet / 8 Fourier modes.

Optional dependency: sympy. Run:
    python -m scripts.certify_punctured_nine_exact

This verifies exact 9x8 Fourier orthogonality, the rank-eight
zero-mean projector, the invertible puncture/dual bridge, and all
nine translations acting through the four 2D character rotations.
No floating tolerances, trained-model claims, analytic j equality
or Leech lattice group recognition are used here.

DASHI owners: OggSSPP2BalancedTernaryPuncturedPlaneExact,
Base369FaceSheetPunctureIncidenceIdentityExact, C3FourierConjugationExact.
"""
from __future__ import annotations
import json
import sympy as sp
from models.punctured_nine_wave_prior import SITES,FREQUENCY_PAIRS


def exact_phase(n:int):
    if n%3==0:
        return sp.Integer(1),sp.Integer(0)
    if n%3==1:
        return -sp.Rational(1,2),sp.sqrt(3)/2
    return -sp.Rational(1,2),-sp.sqrt(3)/2


def fourier_basis():
    return sp.Matrix([
        [sp.sqrt(2)/3 * component
         for k,l in FREQUENCY_PAIRS
         for component in exact_phase(k*x+l*y)]
        for x,y in SITES])


def translation_permutation(shift):
    permutation=sp.zeros(9)
    for i,(x,y) in enumerate(SITES):
        source=(((x-shift[0]+1)%3)-1,((y-shift[1]+1)%3)-1)
        permutation[i,SITES.index(source)]=1
    return permutation


def phase_representation(shift):
    rotation=sp.zeros(8)
    for index,(k,l) in enumerate(FREQUENCY_PAIRS):
        cosine,sine=exact_phase(k*shift[0]+l*shift[1])
        i=2*index
        rotation[i,i]=cosine
        rotation[i,i+1]=-sine
        rotation[i+1,i]=sine
        rotation[i+1,i+1]=cosine
    return rotation


def certify():
    assert len(SITES)==9 and len(FREQUENCY_PAIRS)==4
    basis=fourier_basis()
    identity=sp.eye(9)
    projector=identity-sp.ones(9,9)/9
    assert basis.T*basis == sp.eye(8), "not orthonormal"
    assert basis*basis.T == projector, "incorrect zero-mode complement"
    assert basis.T*sp.ones(9,1)==sp.zeros(8,1)
    # Let f(0)=0. Fourier f loses only the constant mode.
    # Decode g=P f; f(x)=g(x)-g(0) for the eight punctured sites.
    puncture_embedding=sp.zeros(9,8)
    for j in range(8):
        puncture_embedding[j+1,j]=1
    project_to_waves=basis.T*puncture_embedding
    recover=sp.Matrix([
        list((basis.row(j+1)-basis.row(0)))
        for j in range(8)])
    assert recover*project_to_waves == sp.eye(8), "puncture inverse failed"
    for shift in SITES:
        spatial=translation_permutation(shift)
        phase=phase_representation(shift)
        assert spatial*basis==basis*phase, f"translation failed: {shift}"
        assert phase.T*phase==sp.eye(8), f"rotation not orthogonal: {shift}"
    for left in SITES:
        for right in SITES:
            combined=((left[0]+right[0])%3,(left[1]+right[1])%3)
            assert (phase_representation(left)*phase_representation(right)
                    ==phase_representation(combined)), "group law failed"
    return {
        "exact_symbolic_certificate":True,
        "nine_sites":9,"centre_sites":1,"punctured_sites":8,
        "dual_nontrivial_modes":8,"conjugate_character_pairs":4,
        "fourier_orthonormal":True,"zero_mean_projector_exact":True,
        "puncture_to_wave_roundtrip_exact":True,
        "translation_matrices_checked":9,
        "ordered_group_compositions_checked":81,
        "qualified":"finite Fourier and punctured-index theorem; not Leech/j same-object"
    }


if __name__=="__main__":
    print(json.dumps(certify(),indent=2))
