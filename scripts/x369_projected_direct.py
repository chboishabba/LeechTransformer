"""Direct orbit-block projected ridge solve; original Leech-LILA Anatolii Kornienko.
DASHI extension. CPU FP64 algebra on captured native features, no optimizer.
"""
import math
import torch


def label_orbits(table):
    """Every label permutation preserves these exact disjoint coordinate blocks."""
    if table.ndim!=2 or table.dtype!=torch.long or not len(table) or not table.shape[1]:
        raise ValueError('invalid label table')
    rows=table.tolist();n=table.shape[1];unseen=set(range(n));out=[]
    if any(sorted(row)!=list(range(n)) for row in rows):raise ValueError('not permutations')
    while unseen:
        block={min(unseen)}
        while True:
            more=block|{row[i] for row in rows for i in block}
            if more==block:break
            block=more
        out.append(sorted(block));unseen-=block
    return out


def solve_direct(X,Y,masks,table,B0,ridge=1e-3,max_iter=512,tol=1e-9):
    """Same projected ridge objective as the original CG, no rank truncation.
    Whiten feature space by (X^T X+ridge I)^-1/2, then solve invariant
    label-orbit blocks by Cholesky with residual refinement. The original
    unscaled normal residual controls acceptance, not the scaled residual.
    """
    from scripts.x369_translation_repair import projection_groups
    if not math.isfinite(ridge) or ridge<=0 or max_iter<1 or not 0<tol<1:raise ValueError('invalid solve scope')
    if X.dtype!=torch.float64 or Y.dtype!=X.dtype or B0.dtype!=X.dtype or X.device.type!='cpu':raise ValueError('CPU FP64 feature algebra required')
    if B0.shape!=(X.shape[1],Y.shape[1]) or not all(torch.isfinite(v).all() for v in (X,Y,B0)):raise ValueError('invalid data')
    groups=projection_groups(X,Y,masks,table);d=X.shape[1]
    blocks=label_orbits(table)
    # Full feature basis; even exact null directions retain the ridge term.
    _,s,Vh=torch.linalg.svd(X,full_matrices=(len(X)<d));s=torch.nn.functional.pad(s,(0,d-len(s)))
    V=Vh.T;sc=(s.square()+ridge).rsqrt();W=V*sc
    def normal(D):return ridge*D+sum((G@D@P for G,_,P,_,_ in groups),torch.zeros_like(D))
    rhs=sum((x.T@((y-x@B0@P)@P) for _,_,P,x,y in groups),torch.zeros_like(B0))
    rhs_gram=sum((H-G@B0@P for G,H,P,_,_ in groups),torch.zeros_like(B0))
    initial=float(rhs_gram.norm());D=torch.zeros_like(B0);info=[]
    # Form scaled Gram matrices from XW, not cancellation-prone W^T G W.
    scaled=[((x@W).T@(x@W),P) for _,_,P,x,_ in groups]
    regularizer=ridge*torch.diag(sc.square())
    factors=[]
    for block in blocks:
        ix=torch.tensor(block);k=len(block)
        H=torch.kron(regularizer.contiguous(),torch.eye(k,dtype=X.dtype))
        for G,P in scaled:
            Q=P[ix][:,ix].contiguous()
            H+=torch.kron(G.contiguous(),Q)
        H=(H+H.T)*.5
        L,err=torch.linalg.cholesky_ex(H)
        if int(err)!=0:raise RuntimeError('scaled projected ridge Cholesky failed')
        factors.append((ix,L));info.append(dict(labels=block,normal_dimension=d*k))
    def correction(R):
        out=torch.zeros_like(R)
        for ix,L in factors:
            z=torch.cholesky_solve((W.T@R[:,ix]).reshape(-1,1),L).reshape(d,len(ix))
            out[:,ix]=W@z
        return out
    # Use directly evaluated feature residual, avoiding Gram subtraction.
    def residual(D):
        B=B0+D
        return sum((x.T@((y-x@B@P)@P) for _,_,P,x,y in groups),torch.zeros_like(D))-ridge*D
    target=tol*max(1.,initial);r=rhs;steps=0
    for _ in range(min(max_iter,8)):
        if float(r.norm())<=target:break
        D=D+correction(r);r=residual(D);steps+=1
    res=float(r.norm());legacy=float((rhs_gram-normal(D)).norm())
    def loss(B):return sum(float(((x@B@P-y)**2).sum()) for _,_,P,x,y in groups)
    before=loss(B0);after=loss(B0+D);objective=after+ridge*float((D*D).sum())
    floor=sum(float(((y-y@P)**2).sum()) for _,_,P,_,y in groups)
    if not all(math.isfinite(v) for v in (res,legacy,before,after,objective,floor)):raise RuntimeError('nonfinite solve')
    return B0+D,dict(method='label_orbit_whitened_direct_cholesky',iterations=steps,normal_residual=res,
        gram_normal_residual=legacy,initial_residual=initial,converged=max(res,legacy)<=target,
        conditional_distance_bound=max(res,legacy)/ridge,original_sse=before,solved_sse=after,
        regularized_objective=objective,stabilizer_irreducible_sse=floor,ridge=ridge,groups=len(groups),rows=len(X),
        numerical_feature_rank=int((s>s.max()*max(X.shape)*torch.finfo(X.dtype).eps).sum()),
        feature_directions_retained=d,blocks=info,residual_tolerance=tol,rounding_certified=False)


def collision_bounds(X,Y,masks,table,rows):
    """Exact finite collision lower bound for any local projected consumer.
    Preserve byte identity and subgroup, use local one-hot targets only.
    No assumption that different keys can actually be separated by an affine map.
    """
    from collections import defaultdict,Counter
    from fractions import Fraction
    from hashlib import sha256
    if len(X)!=len(Y) or len(X)!=len(rows) or len(X)!=len(masks) or masks.shape[1]!=len(table):raise ValueError('collision scope')
    if not bool(((Y==0)|(Y==1)).all() and (Y.sum(1)==1).all()):raise ValueError('one-hot targets required')
    buckets=defaultdict(list)
    for i,(x,m) in enumerate(zip(X,masks)):
        buckets[(x.contiguous().view(torch.uint8).numpy().tobytes(),tuple(m.tolist()))].append(i)
    labels=Y.argmax(1).tolist();perms=table.tolist();floor=Fraction(0);raw_floor=Fraction(0);correct=0;raw_correct=0;witness=None;conflicts=0
    for (raw,mask),ids in buckets.items():
        c=Counter(labels[i] for i in ids);K=[p for p,m in zip(perms,mask) if m];n=len(ids)
        if not K:raise ValueError('empty stabilizer')
        floor+=n-Fraction(sum(sum(v*c.get(p[j],0) for j,v in c.items()) for p in K),len(K)*n)
        raw_floor+=n-Fraction(sum(v*v for v in c.values()),n);raw_correct+=max(c.values())
        correct+=max((v for j,v in c.items() if all(p[j]==j for p in K)),default=0)
        if len(c)>1:
            conflicts+=1
            if witness is None:
                a=ids[0];b=next(i for i in ids if labels[i]!=labels[a])
                witness=dict(left=rows[a],right=rows[b],local_labels=[labels[a],labels[b]],feature_sha256=sha256(raw).hexdigest(),stabilizer=list(mask))
    return dict(rows=len(rows),distinct_keys=len(buckets),conflicting_keys=conflicts,exact_sse_lower_bound=str(floor),ideal_projected_unique_correct_upper_bound=correct,unprojected_sse_lower_bound=str(raw_floor),deterministic_unique_correct_upper_bound=raw_correct,first_conflict=witness,scope='same stored feature bytes and stabilizer; projected bound is ideal arithmetic, unprojected bound applies to deterministic local consumer')
