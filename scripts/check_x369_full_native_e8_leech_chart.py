"""Full Base369 nine-trit fabric -> existing pointed-nine wave -> E8 chart.

Original Leech-LILA: Anatolii Kornienko; DASHI extension.
This is a source-labelled CHOSEN real coordinate chart, not a lattice
isomorphism between 3^9, E8 or Leech. The CHOSEN nine-site index-zero amplitude is retained; this is NOT
    the zero voxel of the original 27-point interaction geometry.
The 8D orthonormal zero-DC wave space admits E8 root REFLECTIONS, but the
action need not close on native {-1,0,+1}^9 points; an exact residual
and/or explicit domain admission is indispensable.
"""
from itertools import product,combinations
import json
import sympy as S

from scripts.certify_punctured_nine_exact import fourier_basis
from models.punctured_nine_wave_prior import SITES

TRITS=(-1,0,1)
ORDER=('interaction_x','interaction_y','interaction_z',
       'appraisal_A_x','appraisal_A_y','appraisal_A_z',
       'appraisal_B_x','appraisal_B_y','appraisal_B_z')
ONES=S.ones(9,1)


def native_vector(point):
    if len(point)!=9 or any(type(x) is not int or x not in TRITS for x in point):
        raise ValueError('native source requires nine signed ternary coordinates')
    return S.Matrix(point)


def encode_native(point,basis=None):
    """Both Fourier non-DC modes and ORIGINAL distinguished centre."""
    f=native_vector(point)
    B=fourier_basis() if basis is None else basis
    return B.T*f,f[0]


def decode_real(wave,centre,basis=None):
    """Exact real BIDI for any 9-position real amplitude signal."""
    B=fourier_basis() if basis is None else basis
    if wave.shape!=(8,1):raise ValueError('eight wave amplitudes required')
    g=B*wave
    return g+(centre-g[0])*ONES


def native_admission(signal):
    """No silent rounding or sign-guessing of a nonternary E8 image."""
    if signal.shape!=(9,1):raise ValueError('nine real positions required')
    result=tuple(S.simplify(x) for x in signal)
    if any(v not in (S.Integer(-1),S.Integer(0),S.Integer(1)) for v in result):
        raise ValueError('real E8 wave image leaves the native ternary fabric')
    return tuple(map(int,result))


def roots_e8():
    """Independent exact 112 integral + 128 half-integral root source."""
    result=[]
    for i,j in combinations(range(8),2):
        for a,b in product((-1,1),repeat=2):
            v=[S.Integer(0)]*8
            v[i],v[j]=a,b
            result.append(S.Matrix(v))
    for signs in product((-1,1),repeat=8):
        if sum(x<0 for x in signs)%2==0:
            result.append(S.Matrix([S.Rational(x,2) for x in signs]))
    if len(result)!=240 or any((r.T*r)[0]!=2 for r in result):
        raise AssertionError('not the native E8 root inventory')
    return tuple(result)


def reflect_wave(wave,root):
    """The exact E8 Weyl reflection in the chosen wave8 coordinates."""
    if wave.shape!=(8,1) or root.shape!=(8,1) or (root.T*root)[0]!=2:
        raise ValueError('E8 norm-two root and eight-dimensional wave required')
    return wave-root*(root.T*wave)[0]


def preserve_or_residual(point,root,basis=None):
    wave,centre=encode_native(point,basis)
    transformed=decode_real(reflect_wave(wave,root),centre,basis)
    original=native_vector(point)
    # Additive residual is full 9-real source, not a discarded 27-voxel.
    residual=transformed-original
    assert original+residual==transformed
    try:
        admitted=native_admission(transformed)
    except ValueError:
        admitted=None
    return dict(source=tuple(point),target=transformed,
                retained_real_residual=residual,ternary_image=admitted,
                native_fabric_closed=admitted is not None,
                source_centre_retained=centre)


def pack_lila384_native(points,basis=None):
    """48 native T3^9 packets -> existing 16*24=48*8 wave layout.

    The 48 independently retained real centre amplitudes are NOT part of
    the original 384 wave slots. Three 8D E8-ambient charts per 24D block
    are a possible PRIOR, not a 24D Leech lattice gluing certificate.
    """
    if len(points)!=48:
        raise ValueError('384D prior requires exactly 48 native fabric packets')
    B=fourier_basis() if basis is None else basis
    encoded=[encode_native(p,B) for p in points]
    return (S.Matrix.vstack(*(w for w,c in encoded)),
            tuple(c for w,c in encoded))


def reopen_lila384_native(wave384,centres48,basis=None):
    if wave384.shape!=(384,1) or len(centres48)!=48:
        raise ValueError('384 wave modes and 48 centre residuals required')
    B=fourier_basis() if basis is None else basis
    return tuple(native_admission(decode_real(
        wave384[8*i:8*(i+1),:],centres48[i],B))
        for i in range(48))


def centre_erasure_counterexample():
    """Two actual 384D-native source packets with the same wave384."""
    zeros=((0,)*9,)*48
    ones=((1,)*9,)*48
    a,ca=pack_lila384_native(zeros)
    b,cb=pack_lila384_native(ones)
    if a!=b or ca==cb:
        raise AssertionError('full nine centre information not independently needed')
    if reopen_lila384_native(a,ca)!=zeros or reopen_lila384_native(b,cb)!=ones:
        raise AssertionError('centre BIDI failed')
    return dict(equal_wave384=True,distinct_centre48=True,
                full_native_packets_each=48,full_native_trits_each=432)


def certificate():
    B=fourier_basis()
    if B.T*B!=S.eye(8) or B*B.T!=S.eye(9)-ONES*ONES.T/9:
        raise AssertionError('existing punctured-nine Fourier chart changed')
    roots=roots_e8()
    # Prove all 3^9 native packets AT ONCE by the 9x9 matrix
    # identity, rather than 19683 expensive symbolic reductions.
    e0=S.Matrix([[1]+[0]*8])
    reconstruction=B*B.T+ONES*(e0-e0*B*B.T)
    if reconstruction!=S.eye(9):
        raise AssertionError('all-point native BIDI matrix identity failed')
    points=3**9
    for prefix in product(TRITS,repeat=3):
        point=prefix+(0,)*6
        w,c=encode_native(point,B)
        if decode_real(w,c,B)!=native_vector(point):
            raise AssertionError('native fixture BIDI failure')
    # Pure coordinate Weyl root e0+e1: explicit nonclosure witness.
    seed=(1,0,0,0,0,0,0,0,0)
    r=next(root for root in roots if tuple(root)==(1,1,0,0,0,0,0,0))
    receipt=preserve_or_residual(seed,r,B)
    assert not receipt['native_fabric_closed']
    assert receipt['target'][1]==S.Rational(1,3)+S.sqrt(3)/9
    assert receipt['target'][2]==S.Rational(1,3)-S.sqrt(3)/9
    assert receipt['source_centre_retained']==1
    # A full 24D ambient Leech PRIOR can host 3 wave8 blocks, but
    # lattice gluing/coset membership, shared expert dynamics and
    # semantic readout are each independent obligations.
    layout=centre_erasure_counterexample()
    return dict(native_fabric_points=points,native_appraisal_fibre=729,
        lila384_wave_and_centre_bidi=layout,
        distinct_native_9d_real_wave_chart=True,exact_bidi_points=points,
        exact_bidi_witness='9x9 reconstruction identity for every real input',
        wave_dimensions=8,centre_residual_dimensions=1,
        e8_roots=len(roots),e8_integral_roots=112,e8_half_roots=128,
        explicit_weyl_native_nonclosure=dict(source=seed,
            source_root=tuple(map(str,r)),target_first_three=tuple(
                map(str,receipt['target'][:3]))),
        leech24_ambient_wave8_blocks=3,
        leech_lattice_gluing_certified=False,
        e8_to_leech_isometry_certified=False,
        original_neural_source_equivariance_certified=False,
        original_LILA_language_improvement_certified=False,
        agda_kernel_checked=False,training_steps=0)


if __name__=='__main__':
    print(json.dumps(certificate(),sort_keys=True))
