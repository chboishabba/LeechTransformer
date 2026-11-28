"""Source-specific 189-point marked-seed / full 369 fabric crosswalk.

Reuses the EXACT seed layout of scripts/x369_source_geometry.signed_seed:
  nine input values = (M3, D3, 0_3);
  history[e=0,b=0,:3] = M+D, history[e=0,b=1,:3]=M-D,
  all other expert slots and channels initially zero.
One-step original recurrent history is NOT a 9-trit lattice state.

Full fabric owner: DASHI.Foundations.Base369Ternary27HypervoxelFabricGeometryExact.
The source seed is a PROPER MARKED SUBSTRATUM of 27^3, not all 19683.
Original Leech-LILA: Anatolii Kornienko; DASHI extension.
"""
import json
from fractions import Fraction
from itertools import product

TRITS=(-1,0,1)
ZERO=(0,0,0)
MEANS=(ZERO,)+tuple(tuple(s if i==axis else 0 for i in range(3))
                     for axis in range(3) for s in (-1,1))
TRIADS=tuple(product(TRITS,repeat=3))

def packet(mean,residual):
    if tuple(mean) not in MEANS or tuple(residual) not in TRIADS:
        raise ValueError('not an original task seed')
    return tuple(mean)+tuple(residual)+ZERO

def reopen(packet9):
    if len(packet9)!=9 or tuple(packet9[6:])!=ZERO or tuple(packet9[:3]) not in MEANS or tuple(packet9[3:6]) not in TRIADS:
        raise ValueError('not a source-marked nine-slot state')
    return tuple(packet9[:3]),tuple(packet9[3:6])

def paired_initial(mean,residual):
    """Exact mathematical branch coordinates, not floating bytes."""
    h0=tuple(Fraction(a+b) for a,b in zip(mean,residual))
    h1=tuple(Fraction(a-b) for a,b in zip(mean,residual))
    return h0,h1

def unpair_initial(h0,h1):
    if len(h0)!=3 or len(h1)!=3:raise ValueError('expected original first triad')
    mean=tuple((a+b)/2 for a,b in zip(h0,h1))
    residual=tuple((a-b)/2 for a,b in zip(h0,h1))
    return mean,residual

def audit():
    seen=set()
    for m,d in product(MEANS,TRIADS):
        p=packet(m,d);seen.add(p)
        assert reopen(p)==(m,d)
        h=paired_initial(m,d)
        mm,dd=unpair_initial(*h)
        assert mm==tuple(m) and dd==tuple(d)
        assert all(x.denominator==1 for x in (*mm,*dd))
    assert len(seen)==189
    assert len(MEANS)==7 and len(TRIADS)==27
    for x in TRIADS:
        if x!=ZERO:
            try:reopen(ZERO+ZERO+x)
            except ValueError:pass
            else:raise AssertionError('third appraisal erased')
    for m in TRIADS:
        if m not in MEANS:
            try:reopen(m+ZERO+ZERO)
            except ValueError:pass
            else:raise AssertionError('non-source mean admitted')
    s=(0,0,0,1,0,0,0,0,0)
    assert reopen(s)==(ZERO,(1,0,0))
    # This is an ideal-real exact BIDI, not a physical torch rounding
    # guarantee, and not an unrestricted history sufficient statistic.
    return dict(source='SourceGeometryOrderedLearner.signed_seed',
                accepted_mean_states=7,accepted_signed_states=27,
                exact_source_seed_packets=len(seen),full_fabric_points=19683,
                marked_source_is_proper_subset=True,
                third_appraisal_forced_to_zero=True,
                signed_six_initial_semantic_coordinates_reopen=True,
                paired_branch_mean_difference_reopens=True,
                hidden_source_shape='batch x experts x 2 x width',
                neural_transition_factors_through_seed=False,
                original_hyperfabric_dynamics_instantiated=False,
                physical_fp_bidi_certified=False,
                agda_kernel_checked=False,training_steps=0)

if __name__=='__main__':
    print(json.dumps(audit(),sort_keys=True))
