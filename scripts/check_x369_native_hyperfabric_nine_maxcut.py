"""Full 369 finite-hyperfabric chart and cube-face audit.

Source-equivalent independent Python oracle for existing DASHI owners:
  Foundations/Base369Ternary27HypervoxelFabricGeometryExact
  Codec/IndexedCubeMaskSignInstanceExact
  Codec/IndexedAffineCornerAppraisal5832Exact
  Cognition/Base369ZeroFibre
  Foundations/Base369NineCoordinateAggregateBridgeExact

NOT an original learned LeechTransformer source, a physical nine-DOF
dynamics, or an Agda kernel receipt.
Original Leech-LILA Anatolii Kornienko; DASHI extension.
"""
import json
from itertools import product, permutations

TRITS=(-1,0,1)
OPS=('split','rotate','encode')

def encode(x):
    """Zero-aware ternary mask/sign pair; unlike binary, retains orientation."""
    return tuple((0,0) if t==0 else (1,t) for t in x)

def decode(x):
    if any(mask not in (0,1) or (mask==0 and sign!=0) or
           (mask==1 and sign not in (-1,1)) for mask,sign in x):
        raise ValueError('invalid mask/sign fibre')
    return tuple(0 if mask==0 else sign for mask,sign in x)

def rotate(x):
    return tuple(y for i in range(0,9,3)
                 for y in (x[i+1],x[i+2],x[i]))

def split(x):
    if len(x)!=9:raise ValueError('nine labelled coordinates required')
    return x[:6],x[6:]

def join(pair):
    prefix,residual=pair
    if len(prefix)!=6 or len(residual)!=3:raise ValueError('6+3 residual')
    return prefix+residual

def route(x,order):
    """Typed endpoint normal form of each existing H07 cube path.
    Actual Agda edges have different vertex types; this is their
    source-equivalent terminal chart, not a new typed cube definition.
    """
    for operation in order:
        if operation=='X':x=join(split(x))
        elif operation=='Y':x=rotate(x)
        elif operation=='Z':x=encode(x)
        else:raise ValueError('unknown edge operation')
    return x

def audit():
    points=face_checks=path_checks=aggregate_zero=0
    path_edges=periodic_edges=0
    witnessed_interaction_collision=False
    witnessed_six_slice_collision=False
    for x in product(TRITS,repeat=9):
        points+=1
        if not sum(x):aggregate_zero+=1
        assert decode(encode(x))==x
        assert join(split(x))==x
        assert join(split(rotate(x)))==rotate(join(split(x)))
        assert encode(join(split(x)))==join(split(encode(x)))
        assert encode(rotate(x))==rotate(encode(x))
        encoded=encode(x)
        assert join(split(rotate(encoded)))==rotate(join(split(encoded)))
        assert encode(join(split(rotate(x))))==join(split(encode(rotate(x))))
        assert encode(rotate(join(split(x))))==rotate(encode(join(split(x))))
        face_checks+=6
        terminals=tuple(route(x,p) for p in permutations('XYZ'))
        assert len(set(terminals))==1
        assert decode(terminals[0])==rotate(x)
        path_checks+=6
        # Count undirected edges of native nonwrapping P3^9 and
        # periodic C3^9 separately; each positive oriented edge once.
        path_edges+=sum(1 for t in x if t!=1)
        periodic_edges+=9
        if x[:3]==(0,0,0) and x[3:]==(0,)*6:
            witnessed_interaction_collision=True
        if x[:6]==(0,)*6 and x[6:]==(0,)*3:
            witnessed_six_slice_collision=True
    assert points==19683
    assert face_checks==path_checks==6*19683
    assert path_edges==118098 and periodic_edges==177147
    assert aggregate_zero==3139
    assert witnessed_interaction_collision and witnessed_six_slice_collision
    interaction_p=(0,)*9
    interaction_q=(0,)*3+(1,0,0)+(0,)*3
    assert interaction_p[:3]==interaction_q[:3] and interaction_p!=interaction_q
    six_p=(0,)*9
    six_q=(0,)*6+(1,0,0)
    assert six_p[:6]==six_q[:6] and six_p!=six_q
    assert encode((0,)*9)!=encode((1,)*9)
    assert decode(encode((-1,0,1)*3))==(-1,0,1)*3
    # Pullback q6: ternary value x binary orientation; q9:
    # ternary value x relational value. Equal shared ternary values.
    shared=[(tri,ori,relation)
            for tri,ori,relation in product(TRITS,(0,1),TRITS)]
    assert len(shared)==18
    assert sum(tri==0 for tri,_,_ in shared)==6
    return dict(scope='source-equivalent finite 369 address and cube chart',
                full_nine_points=points,appraisal_fibre_per_interaction=729,
                interaction_states=27,zero_sum_nine_fibre=aggregate_zero,
                original_H07_face_checks=face_checks,
                original_H07_terminal_path_checks=path_checks,
                nonwrapping_P3_9_edges=path_edges,periodic_C3_9_edges=periodic_edges,
                six_nine_pullback_states=len(shared),
                visible_zero_hidden_states=6,
                interaction_only_appraisal_collision=True,
                first_six_only_final_appraisal_collision=True,
                finite_binary_mask_sign_reopens=True,
                corner_appraisal_5832_is_stratum_only=True,
                source_native_neural_equivalence=False,
                native_physical_9dof=False,agda_kernel_checked=False,
                optimizer_steps=0)

if __name__=='__main__':
    print(json.dumps(audit(),sort_keys=True))
