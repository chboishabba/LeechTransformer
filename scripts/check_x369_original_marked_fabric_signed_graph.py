"""Source original signed S3 action extends to a chosen full Base369 P3^9 graph.

This is the ORIGINAL group action on M3 and the mirror-odd D3,
plus its explicitly CHOSEN same-acting third three-axis block. The
extension is a signed monomial 9×9 matrix and therefore a genuine graph
automorphism of the nonwrapping ternary path-product graph P3^9.
It is not arbitrary palette S3 (which fails strict -1↔0↔+1 grid
adjacency), not the C3^9 periodic torus, and not original learned MoE
transition or an E8/Leech lattice automorphism.

Original Leech-LILA: Anatolii Kornienko; DASHI extension.
"""
from itertools import product
import sympy as S
from models.geometric_reflection_fibre import GROUP,mul,inv,IDENTITY,act
from scripts.x369_original_marked_seed_fabric import signed_original_task_action

def native_nine(point):
    point=tuple(point)
    if len(point)!=9 or any(type(x) is not int or x not in (-1,0,1) for x in point):
        raise ValueError('original full ternary nine-axis carrier required')
    return point

def full_signed_action(g,point):
    """Explicit extension: original M,D semantics, chosen extra block."""
    p=native_nine(point)
    if g not in GROUP:raise ValueError('original source group element required')
    mean,residual=signed_original_task_action(g,p[:3],p[3:6])
    last=act(g,p[6:9])
    return mean+residual+last

def monomial9(g):
    if g not in GROUP:raise ValueError('original source group element required')
    columns=[]
    for k in range(9):
        e=tuple(int(k==j) for j in range(9))
        columns.append(S.Matrix(full_signed_action(g,e)))
    return S.Matrix.hstack(*columns)

def strict_p3_nine_adjacent(a,b):
    a=native_nine(a);b=native_nine(b)
    return sum(x!=y for x,y in zip(a,b))==1 and sum(abs(x-y) for x,y in zip(a,b))==1

def certificate():
    matrices={g:monomial9(g) for g in GROUP}
    zero=S.zeros(9)
    for g,M in matrices.items():
        if M.T*M!=S.eye(9) or M.det() not in (-1,1):
            raise AssertionError('original signed action not orthogonal monomial')
        for i in range(9):
            if sum(M[j,i]!=0 for j in range(9))!=1:
                raise AssertionError('source signed action not a coordinate permutation')
            if any(M[j,i] not in (-1,0,1) for j in range(9)):
                raise AssertionError('source action not signed ternary')
        if matrices[inv(g)]*M!=S.eye(9):
            raise AssertionError('source graph action inverse failed')
    for g,h in product(GROUP,repeat=2):
        if matrices[g]*matrices[h]!=matrices[mul(g,h)]:
            raise AssertionError('source graph extension not a group action')
    # Signed coordinate-permutation maps preserve all P3 edges because
    # any original one-coordinate +/-1 step remains exactly one such
    # step. Check the two edge types on each of the nine axes for all
    # 12 operators; the monomial matrix makes this a universal proof.
    for g in GROUP:
        for axis in range(9):
            for low,high in ((-1,0),(0,1)):
                a=[0]*9;b=[0]*9
                a[axis]=low;b[axis]=high
                if not strict_p3_nine_adjacent(
                      full_signed_action(g,tuple(a)),
                      full_signed_action(g,tuple(b))):
                    raise AssertionError('P3^9 edge not preserved')
    # Exact nonwrapping versus periodic edge counts.
    p3_edges=9*2*3**8
    c3_edges=9*3**9
    if p3_edges!=118098 or c3_edges!=177147:
        raise AssertionError('original full nine-grid adjacency counts changed')
    m0=(0,0,0);d0=(1,0,-1);original=m0+d0+(0,0,0)
    if any(full_signed_action(g,original)[6:]!=(0,0,0) for g in GROUP):
        raise AssertionError('original 189-seed last block no longer zero')
    return dict(original_signed_group_order=len(GROUP),
        full_original_native_trit_states=3**9,
        full_nonwrapping_P3_nine_edges=p3_edges,
        periodic_C3_nine_edges=c3_edges,
        source_9x9_signed_monomial_witnesses=len(matrices),
        group_composition_matrices_checked=len(GROUP)**2,
        representative_grid_edges_checked=len(GROUP)*9*2,
        every_full_native_P3_edge_preserved_by_monomial_proof=True,
        original_189_initial_mark_subcarrier_preserved=True,
        source_appraisal_B_extension_is_chosen_not_original_model_action=True,
        palette_trit_value_S3_automorphism_claim=False,
        original_neural_recurrent_equivariance_proved=False,
        actual_leech_lattice_automorphism_proved=False,
        physical_nine_dof_metric_invariance_proved=False,
        agda_kernel_checked=False)

if __name__=='__main__':
    import json
    print(json.dumps(certificate(),sort_keys=True,indent=2))
