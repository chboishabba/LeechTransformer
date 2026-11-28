"""Exact necessary origin-ray constraints for NEW SourceGeometryOrderedLearner.
Original Leech-LILA: Anatolii Kornienko; DASHI geometric extension.
Not a fitted FairLearner checkpoint migration, native floating proof, or
sufficiency test. Uses exact rational interpretations of stored parameters.
"""
from fractions import Fraction as Q
from itertools import permutations
from models.geometric_reflection_fibre import GROUP, act, mul, inv
from models.geometric_paired_history import pair_action
from scripts.x369_r4_interval_certificate import fused_activation, point


def vec(t):
    return tuple(Q(float(x)) for x in t.detach().cpu().flatten().tolist())


def mat(t):
    return tuple(vec(row) for row in t.detach().cpu())


def dot(a,b):
    if len(a)!=len(b): raise ValueError("width mismatch")
    return sum((x*y for x,y in zip(a,b)),Q(0))


def mv(A,v):
    return tuple(dot(row,v) for row in A)


def sign(x):
    return int(x>0)-int(x<0)


def pair_g(g,h):
    return tuple(pair_action(g,p) for p in h)


def flatten(h):
    return tuple(x for p in h for b in p for x in b)


def conjugate(g,u):
    return GROUP.index(mul(mul(g,GROUP[u]),inv(g)))


def canonical(u,h):
    c=max((conjugate(g,u),flatten(pair_g(g,h))) for g in GROUP)
    mark=next(g for g in GROUP if (conjugate(g,u),flatten(pair_g(g,h)))==c)
    z=pair_g(mark,h)
    K=tuple(g for g in GROUP if conjugate(g,c[0])==c[0] and pair_g(g,z)==z)
    return c[0],z,K


def project_pair(p,K):
    rows=[pair_action(g,p) for g in K]
    return tuple(tuple(sum(x,Q(0))/len(K) for x in zip(*(r[b] for r in rows))) for b in range(2))


def eventual_orientation(x,h,w,b):
    d=len(x);a=dot(w[d:],h);c=dot(w[:d],x)
    if a or c:return sign(a or c)
    # Source fallback: C3 lexicographic orientation, reflected for branch 1.
    slope=(Q(0),)*d+h; intercept=x+(Q(0),)*d
    if b:
        slope=act((0,1,0),slope);intercept=act((0,1,0),intercept)
    def can(a):
        return max(tuple(zip(act((k,0,a),slope),act((k,0,a),intercept))) for k in range(3))
    p,n=can(0),can(1)
    return int(p>n)-int(p<n)


def branch_slope(x,h,orientation,b,first,last,transition,capped):
    d=len(x);s=eventual_orientation(x,h,orientation,b)
    # GELU(t*a+b)/t tends to max(a,0), including the zero slope.
    r=tuple(max(Q(0),s*dot(row[d:],h)) for row in first)
    y=tuple(s*v for v in mv(last,r))
    if capped:return y,(Q(0),)*d
    return y,mv(transition,h+(Q(0),)*d+y) if s else (Q(0),)*d


def possible_orders(bias,slopes,k):
    for order in permutations(range(len(bias)),k):
        outside=set(range(len(bias)))-set(order)
        if any(bias[i]<bias[j] for i,j in zip(order,order[1:])):continue
        if any(bias[i]<bias[j] for i in order for j in outside):continue
        if any(slopes[i]!=slopes[order[0]] for i in order):continue
        if any(slopes[j]>slopes[order[0]] for j in outside):continue
        yield order


def projected_label_slope(values,tables,K):
    out=[];off=0
    for table in tables:
        n=len(table[0]);v=values[off:off+n];off+=n
        out.extend(sum((v[table[GROUP.index(g)][i]] for g in K),Q(0))/len(K) for i in range(n))
    if off!=len(values):raise ValueError("label width mismatch")
    return tuple(out)


def ray_constraints(model,direction,control,tag):
    """One exact full-domain direction exclusion, all possible origin ties.
    Observation contract: world output, ordered IDs, ideal-real positive
    mixture weights and world raw label logits. Necessary only.
    direction is a nonzero [E,2,width] floating tensor, read as exact dyadics.
    """
    import torch
    from scripts.x369_source_geometry import SourceGeometryOrderedLearner
    from models.history_orbit_moe import HistoryOrbitMoE
    if type(model) is not SourceGeometryOrderedLearner or model.training:
        raise ValueError("original new eval architecture required")
    src=model.history.source;d=model.width;E=model.experts
    if (type(src) is not HistoryOrbitMoE or src.route_geometry!="c2_c3_residual"
        or src.transition_rank is not None or getattr(src,"route_probe_action",None) is not None
        or not 0<=control<len(GROUP) or not 0<=tag<4
        or tuple(direction.shape)!=(E,2,d) or not torch.isfinite(direction).all()
        or not direction.is_floating_point() or not direction.any()):
        raise ValueError("unsupported exact source or direction")
    if any(not torch.isfinite(p).all() for p in model.parameters()):raise ValueError("nonfinite weights")
    if src.history_cap is not None and not 0<float(src.history_cap)<float("inf"):raise ValueError("invalid cap")
    h=tuple(tuple(vec(b) for b in p) for p in direction)
    u,h,K=canonical(control,h)
    x=vec(model.control_embed.weight[tag*len(GROUP)+u])
    w=vec(src.c3_residual_weight)[d:];base=vec(src.router.weight)[d:]
    # Actual rotate3 is left roll; existing rational directed-score owner.
    z=tuple(v.lo for v in fused_activation(list(map(point,act((1,0,0),x))),list(map(point,w))))
    mean=[tuple((a+b)/2 for a,b in zip(*p)) for p in h]
    slopes=tuple(dot(base,tuple(map(abs,m)))+dot(z,m) for m in mean)
    bias=vec(src.expert_bias);orient=vec(src.orientation.weight)
    data=[]
    for e,p in enumerate(h):
        expert=src.experts[e];trans=src.transitions[e]
        y,t=zip(*(branch_slope(x,bh,orient,b,mat(expert[0].weight),
                  mat(expert[2].weight),mat(trans.weight),src.history_cap is not None)
                  for b,bh in enumerate(p)))
        data.append((flatten((project_pair(y,K),)),project_pair(t,K)))
    tables=[getattr(model,f"label_perm_{i}").detach().cpu().tolist() for i in range(3)]
    W=tuple(row[2*d:] for row in mat(model.readout.weight))
    orders=[]
    for order in possible_orders(bias,slopes,src.top_k):
        # Distinct algebraic biases: exp(bias) are linearly independent over
        # algebraic numbers (Lindemann-Weierstrass). Equal biases group first.
        # All expert slope vectors here are rational, so a zero mixture
        # requires the sum of projected slopes in EACH equal-bias group zero.
        groups={}
        for e in order:groups.setdefault(bias[e],[]).append(data[e][0])
        output_zero=all(not any(sum(col,Q(0)) for col in zip(*ys)) for ys in groups.values())
        nxt=tuple(data[e][1] if e in order else h[e] for e in range(E))
        label_zero=not any(projected_label_slope(mv(W,flatten(nxt)),tables,K))
        orders.append(dict(order=order,output_slope_zero=output_zero,
                           history_readout_slope_zero=label_zero))
    # The label test assumes output slope zero; both must hold.
    survives=any(v["output_slope_zero"] and v["history_readout_slope_zero"] for v in orders)
    return dict(control=control,canonical_control=u,tag=tag,stabilizer=len(K),
                possible_orders=orders,direction_excluded=not survives,
                necessary_only=True,native_float_certified=False)
