"""Exact F3^2 Fourier coefficients on Leech's original nine-site ordering.

Each identity is checked by universal linear/bilinear basis coefficients
in Z[omega], omega^2+omega+1=0, not by floating point samples. This is
an exact finite reference for DASHI A03-A05, NOT an Agda kernel receipt
or trained-neural/Leech lattice/physical same-object proof.
"""
from __future__ import annotations
import json
from models.punctured_nine_wave_prior import SITES, PUNCTURE_SITES, FREQUENCY_PAIRS

ZERO=(0,0)
ONE=(1,0)
NINE=(9,0)

def e_add(a,b):
    return a[0]+b[0],a[1]+b[1]

def e_mul(a,b):
    x,y=a;u,v=b
    return x*u-y*v,x*v+y*u-y*v

def e_conj(a):
    return a[0]-a[1],-a[1]

def e_sum(values):
    result=ZERO
    for value in values:
        result=e_add(result,value)
    return result

def omega(n):
    return (ONE,(0,1),(-1,-1))[n%3]

def canon(n):
    return (n+1)%3-1

def add(x,y):
    return canon(x[0]+y[0]),canon(x[1]+y[1])

def neg(x):
    return canon(-x[0]),canon(-x[1])

def sub(x,y):
    return add(x,neg(y))

def chi(frequency,point,*,inverse=False):
    exponent=frequency[0]*point[0]+frequency[1]*point[1]
    return omega(-exponent if inverse else exponent)

def fourier_of_delta(point,frequency):
    return chi(frequency,point,inverse=True)

def encode_offsets(sheet):
    centre=sheet[SITES[0]]
    return centre,{u:sheet[u]-centre for u in PUNCTURE_SITES}

def decode_offsets(centre,offsets):
    return {u:centre+(0 if u==SITES[0] else offsets[u]) for u in SITES}

def translate_sheet(sheet,shift):
    return {u:sheet[sub(u,shift)] for u in SITES}

def translate_offsets(offsets,shift):
    """The induced nine-site action must re-gauge the moved centre."""
    full={SITES[0]:0,**offsets}
    return {u:full[sub(u,shift)]-full[neg(shift)] for u in PUNCTURE_SITES}

def centred_norm_times_nine(sheet):
    vals=list(sheet.values())
    return 9*sum(v*v for v in vals)-sum(vals)**2

def certify():
    assert SITES[0]==(0,0)
    assert len(SITES)==len(set(SITES))==9
    assert len(PUNCTURE_SITES)==8
    assert {tuple(x%3 for x in s) for s in SITES}=={
        (a,b) for a in range(3) for b in range(3)}
    assert len(FREQUENCY_PAIRS)==4
    assert {tuple(x%3 for x in k) for k in FREQUENCY_PAIRS}.isdisjoint({(0,0)})
    inversion=0
    for x in SITES:
        for y in SITES:
            got=e_sum(e_mul(chi(k,x),fourier_of_delta(y,k)) for k in SITES)
            assert got==(NINE if x==y else ZERO),('inverse',x,y,got)
            inversion+=1
    conv=0;product_conv=0
    for a in SITES:
        for b in SITES:
            for k in SITES:
                actual=fourier_of_delta(add(a,b),k)
                expected=e_mul(fourier_of_delta(a,k),fourier_of_delta(b,k))
                assert actual==expected,('convolution',a,b,k)
                conv+=1
                # 9 F(delta_a delta_b)(k) = sum_l F_a(l) F_b(k-l).
                lhs=tuple(9*v for v in fourier_of_delta(a,k)) if a==b else ZERO
                rhs=e_sum(e_mul(fourier_of_delta(a,l),
                      fourier_of_delta(b,sub(k,l))) for l in SITES)
                assert lhs==rhs,('pointwise',a,b,k)
                product_conv+=1
    parseval=0;reality=0
    for a in SITES:
        for b in SITES:
            got=e_sum(e_mul(fourier_of_delta(a,k),
                      e_conj(fourier_of_delta(b,k))) for k in SITES)
            assert got==(NINE if a==b else ZERO)
            parseval+=1
        for k in SITES:
            assert fourier_of_delta(a,neg(k))==e_conj(fourier_of_delta(a,k))
            reality+=1
    centre_recovery=0;translation_gauge=0;action_composition=0;norm_checks=0
    for basis in SITES:
        sheet={u:int(u==basis) for u in SITES}
        c,off=encode_offsets(sheet)
        assert decode_offsets(c,off)==sheet
        centre_recovery+=1
        for s in SITES:
            shifted=translate_sheet(sheet,s)
            _,shifted_off=encode_offsets(shifted)
            assert shifted_off==translate_offsets(off,s)
            assert centred_norm_times_nine(shifted)==centred_norm_times_nine(sheet)
            translation_gauge+=len(PUNCTURE_SITES)
            norm_checks+=1
            for t in SITES:
                assert translate_offsets(translate_offsets(off,s),t)==(
                    translate_offsets(off,add(s,t)))
                action_composition+=len(PUNCTURE_SITES)
    # A concrete no-go for the FALSE claim of naive puncture norm invariance.
    centre_delta={u:int(u==SITES[0]) for u in SITES}
    _,off=encode_offsets(centre_delta)
    naive_before=sum(v*v for v in off.values())
    naive_after=sum(v*v for v in translate_offsets(off,(1,0)).values())
    assert naive_before==8 and naive_after==1
    return {
        'schema':1,
        'kind':'x369_original_nine_sheet_exact_universal_coefficient_certificate',
        'ring':'Z[omega]/(omega^2+omega+1); exact integer pairs',
        'carrier':'models.punctured_nine_wave_prior.SITES, original order',
        'nine_sites':len(SITES),
        'inverse_fourier_matrix_coefficients_checked':inversion,
        'convolution_bilinear_coefficients_checked':conv,
        'pointwise_to_frequency_convolution_coefficients_checked':product_conv,
        'parseval_sesquilinear_coefficients_checked':parseval,
        'real_conjugate_basis_coefficients_checked':reality,
        'pointed_centre_recovery_basis_cases':centre_recovery,
        'translation_gauge_basis_coefficients_checked':translation_gauge,
        'induced_translation_group_basis_coefficients_checked':action_composition,
        'nine_sheet_centred_norm_translation_basis_cases':norm_checks,
        'punctured_eight_naive_norm_counterexample':{
            'before':naive_before,'after_translation_1_0':naive_after},
        'all_arithmetic_exact':True,
        'agda_kernel_verified':False,
        'original_neural_model_verified':False,
        'physical_cross_lane_identification_verified':False,
    }

if __name__=='__main__':
    print(json.dumps(certify(),indent=2,sort_keys=True))
