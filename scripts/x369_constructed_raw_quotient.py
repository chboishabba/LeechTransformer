"""Consumer-relative constructed dense12 value quotient.
Original Leech-LILA: Anatolii Kornienko; DASHI extension.
Exact saturated-value reference, and actual-source numeric-value audit.
Not original real GELU equality, fitted history or IEEE-byte compression.
"""
from itertools import product

RAW_COUNTS=(1053,1449,1457,1,1449,1337,1457,1,1449,1337,1457,1)
INDEXED_COUNTS=(1053,1457,1457,9,1449,1337,1457,9,1449,1337,1457,9)


def _semantic_index(m,d):
    z=tuple(m)+tuple(d)
    if len(z)!=6 or any(x not in (-1,0,1) for x in z):raise ValueError("signed ternary pair required")
    return sum({0:0,1:1,-1:2}[x]*3**i for i,x in enumerate(z))


def _decode_index(n):
    if type(n)!=int or not 0<=n<729:raise ValueError("semantic index")
    z=tuple((0,1,-1)[(n//3**i)%3] for i in range(6))
    return z[:3],z[3:]


def encode_index(code,consumer):
    """Dense fixed-length code, zero special-cased; numeric values only."""
    m,d,mark=code;n=_semantic_index(m,d)
    if mark not in (0,1,-1) or (n==0 and mark!=0):raise ValueError("invalid mark")
    if consumer=="typed":return n
    if consumer=="raw":return 2*n-1+int(mark==-1) if n else 0
    if consumer=="reversible":return 3*n-2+(0,1,-1).index(mark) if n else 0
    raise ValueError("unknown consumer")


def decode_index(n,consumer):
    """Canonical section for lossy consumers; exact inverse for reversible."""
    limit={"typed":729,"raw":1457,"reversible":2185}[consumer]
    if type(n)!=int or not 0<=n<limit:raise ValueError("code index")
    if consumer=="typed":return (*_decode_index(n),0)
    if not n:return (*_decode_index(0),0)
    k=2 if consumer=="raw" else 3
    i,j=divmod(n-1,k)
    return (*_decode_index(i+1),(0,-1)[j] if k==2 else (0,1,-1)[j])


def step_index(n,g,consumer):
    from scripts.x369_dense_compact import step_constructed_code
    return encode_index(step_constructed_code(decode_index(n,consumer),g),consumer)


def _owner():
    from models.geometric_reflection_fibre import GROUP,mul,inv
    from models.geometric_paired_history import pair_action
    from scripts.x369_dense_compact import (terms,SPATIAL,T,
        encode_constructed_value_history,decode_constructed_value_history,
        step_constructed_value_history,step_constructed_code)
    return GROUP,mul,inv,pair_action,terms,SPATIAL,T,encode_constructed_value_history,decode_constructed_value_history,step_constructed_value_history,step_constructed_code


def reference_certificate():
    """All 2185 value states x12 signed controls. Exact integer arithmetic.

    Raw output is divided by the fixed positive first-expert mixture weight,
    then multiplied by 24 (2 for dyadic coefficients, 12 for Reynolds).
    The reference is exact for the saturated ReLU-equivalent gate contract.
    """
    G,mul,inv,pa,terms,spatial,T,encode,decode,step,code_step=_owner()
    def conj(g,u):return mul(mul(g,u),inv(g))
    atoms=[(tuple(int(2*x) for x in l),tuple(int(2*x) for x in w),c) for l,w,c in terms()]
    def local(h,u):
        k,b,a=u;i=spatial.index((k,b,0));sgn=(-1)**a;ys=[]
        for p in h:
            y=[0]*9
            for l,w,c in atoms:
                v=sum(x*z for x,z in zip(l,p))
                if v%2:raise ValueError("not admitted integer gate input")
                v//=2;s=sgn*c[i]
                if c==(1,0,0):
                    v=max(0,v+(T if s==1 else -T))-max(0,v+(T if s==-1 else -T))
                else:v=2*max(0,v+T*s)
                for j in range(9):y[j]+=w[j]*v
            ys.append(tuple(y))
        return tuple(ys)
    def observe(h,u):
        cc,ch=max((conj(g,u),pa(g,h)) for g in G)
        mark=next(g for g in G if (conj(g,u),pa(g,h))==(cc,ch))
        K=tuple(g for g in G if (conj(g,cc),pa(g,ch))==(cc,ch))
        y=local(ch,cc);terms=[pa(g,y) for g in K]
        p=tuple(tuple(sum(z[j][i] for z in terms)*(12//len(K)) for i in range(9)) for j in (0,1))
        out=pa(inv(mark),p)
        return out,(out,cc,mark,K,y,p)
    triads=tuple(product((-1,0,1),repeat=3))
    states={encode(m,d,k) for m,d,k in product(triads,triads,(0,-1,1)) if any(m+d) or k==0}
    if len(states)!=2185:raise AssertionError("value carrier")
    maps=[[{} for u in G] for _ in range(2)];edges=0
    for h in states:
        c=decode(h);r=encode_index(c,"raw")
        for family in ("typed","raw","reversible"):
            n=encode_index(c,family)
            if encode_index(decode_index(n,family),family)!=n:raise AssertionError("index roundtrip")
        if encode(*decode_index(encode_index(c,"reversible"),"reversible"))!=h:raise AssertionError("reversible section")
        for i,u in enumerate(G):
            nxt=step(h,u);nc=decode(nxt)
            if nxt not in states or nc!=code_step(c,u):raise AssertionError("successor closure")
            for family in ("typed","raw","reversible"):
                if encode_index(nc,family)!=step_index(encode_index(c,family),u,family):
                    raise AssertionError("index successor descent")
            for obs,table in zip(observe(h,u),maps):
                if table[i].setdefault(r,obs)!=obs:raise AssertionError("raw fibre separation")
            edges+=1
    counts=[[len(set(d.values())) for d in table] for table in maps]
    if tuple(counts[0])!=RAW_COUNTS or tuple(counts[1])!=INDEXED_COUNTS:
        raise AssertionError(("unexpected reference observation counts",counts))
    probe=G.index((0,1,0))
    if any(row[probe]!=1457 for row in counts):raise AssertionError("reflection must separate raw classes")
    return dict(states=len(states),edges=edges,typed_classes=729,raw_classes=1457,
        reversible_classes=2185,fixed_length_bits=(10,11,12),
        raw_single_control_counts=counts[0],indexed_single_control_counts=counts[1],
        separating_control=G[probe],byte_identity=False)


class RawValueAudit:
    """Collect same-pass actual source observations without changing state.

    No raw history is included in the observation. Python numeric equality
    deliberately merges signed zeros. Batch/device/runtime remain receipt scope.
    """
    def __init__(self):
        self.maps=[[{} for _ in range(12)] for _ in range(2)]
        self.rows=0
        self.weights=None

    def add(self,u,h,out):
        import torch
        from scripts.x369_dense_compact import decode_constructed_value_history
        if not 0<=u<12 or not torch.isfinite(out.output).all() or not torch.isfinite(out.weights).all():
            raise RuntimeError("nonfinite raw observation")
        if not (out.ids==out.ids.new_tensor([0,1])).all() or not (out.local_signs==1).all():
            raise RuntimeError("constructed route/orientation contract")
        if u%4==3 and (out.output!=0).any():raise RuntimeError('negative reflection must emit zero raw value')
        hv=h[:,0,:,:9].detach().cpu().tolist()
        def vals(x):return x.detach().flatten(1).cpu().tolist()
        cols=[vals(x) for x in (out.output,out.ids,out.local_signs,out.weights,
            out.raw_local_output,out.local_output)]
        marks=out.world_to_local.detach().cpu().tolist()
        controls=out.local_control.detach().cpu().tolist()
        stabilizers=out.stabilizer.detach().cpu().tolist()
        for j,p in enumerate(hv):
            if any(v!=int(v) for branch in p for v in branch):raise RuntimeError("noninteger input history")
            code=decode_constructed_value_history(tuple(tuple(int(v) for v in branch) for branch in p))
            r=encode_index(code,"raw")
            raw=tuple(tuple(c[j]) for c in cols[:4])
            if self.weights is None:self.weights=raw[3]
            if raw[3]!=self.weights or not self.weights[0]>0:raise RuntimeError('constant positive mixture contract')
            indexed=(raw,marks[j],controls[j],tuple(stabilizers[j]),tuple(cols[4][j]),tuple(cols[5][j]))
            for obs,table in zip((raw,indexed),self.maps):
                if table[u].setdefault(r,obs)!=obs:raise RuntimeError("actual raw value fibre separated")
            self.rows+=1

    def finish(self):
        if any(len(d)!=1457 for table in self.maps for d in table):
            raise RuntimeError("incomplete raw quotient carrier")
        counts=[[len(set(d.values())) for d in table] for table in self.maps]
        if tuple(counts[0])!=RAW_COUNTS or tuple(counts[1])!=INDEXED_COUNTS:
            raise RuntimeError("actual raw/indexed discriminator counts differ")
        return dict(rows=self.rows,raw_future_classes=1457,typed_future_classes=729,
            raw_single_control_counts=counts[0],indexed_single_control_counts=counts[1],
            separating_control=(0,1,0),numeric_value_only=True,byte_identity=False)



# Existing DASHI pointed 729 address order:
# scripts/check_monster_3b_pointed_720_729_affine_projector.py.
# Here the chosen basis change puts the actual X369 C3-fixed DC plane
# on the donor's first two coordinates. Address equality is not Monster
# representation equality.
def balanced3(x):
    return (x+1)%3-1


def pointed_chart(m,d):
    _semantic_index(m,d)  # existing admission / signed digit order
    return (m[0],d[0],balanced3(m[1]-m[0]),balanced3(m[2]-m[0]),
            balanced3(d[1]-d[0]),balanced3(d[2]-d[0]))


def pointed_unchart(x):
    if len(x)!=6 or any(v not in (-1,0,1) for v in x):raise ValueError("six balanced trits required")
    a,b,p,q,r,s=x
    return (a,balanced3(a+p),balanced3(a+q)),(b,balanced3(b+r),balanced3(b+s))


def pointed_index(m,d):
    x=pointed_chart(m,d)
    return _semantic_index(x[:3],x[3:])


def unpointed_index(n):
    x,y=_decode_index(n)
    return pointed_unchart(x+y)


def pointed_address(n):
    _decode_index(n)
    q,t=divmod(n,9)
    if q==0:return ("central",t)
    corner,fine=divmod(q-1,10)
    return ("noncentral",corner,fine,t)


def address_index(address):
    if len(address)==2 and address[0]=="central":
        t=address[1]
        if type(t)==int and 0<=t<9:return t
    if len(address)==4 and address[0]=="noncentral":
        c,f,t=address[1:]
        if all(type(v)==int for v in (c,f,t)) and 0<=c<8 and 0<=f<10 and 0<=t<9:
            return 9*(1+10*c+f)+t
    raise ValueError("invalid pointed address")


def pointed_action(g,x):
    """Direct skew product in the donor's 9 x 81 address coordinates.
    No inference that eight corners or ten sectors separately intertwine.
    """
    k,b,a=g
    if k not in range(3) or b not in (0,1) or a not in (0,1):raise ValueError("invalid signed control")
    if len(x)!=6 or any(v not in (-1,0,1) for v in x):raise ValueError("invalid chart")
    i=tuple((2-j-k)%3 if b else (j+k)%3 for j in range(3))
    inner=[];outer=[]
    for lane in range(2):
        t=x[lane];v=(0,)+tuple(x[2+2*lane:4+2*lane]);s=(-1)**(a+b*lane)
        inner.append(balanced3(s*(t+v[i[0]])))
        outer.extend(balanced3(s*(v[j]-v[i[0]])) for j in i[1:])
    return tuple(inner+outer)


def step_pointed_index(n,g):
    x,y=_decode_index(n)
    out=pointed_action(g,x+y)
    return _semantic_index(out[:3],out[3:])


def pointed729_certificate():
    """Existing appraisal/Heisenberg and pointed address weld.
    Exact coordinate and group checks; no kernel/Monster/fitted-state claim.
    """
    from collections import Counter
    from models.geometric_reflection_fibre import GROUP,act,mul
    triads=tuple(product((-1,0,1),repeat=3))
    z=tuple(m+d for m,d in product(triads,repeat=2))
    def source(g,v):
        return act(g,v[:3])+tuple((-1)**g[1]*x for x in act(g,v[3:]))
    fibres={};seen=set();hist=Counter();fixed={}
    for v in z:
        x=pointed_chart(v[:3],v[3:]);n=pointed_index(v[:3],v[3:])
        assert unpointed_index(n)==(v[:3],v[3:]) and address_index(pointed_address(n))==n
        assert (n<9)==(v[0]==v[1]==v[2] and v[3]==v[4]==v[5])
        fibres.setdefault(x[2:],set()).add(x[:2])
        for g in GROUP:
            u=source(g,v);y=pointed_action(g,x)
            assert pointed_unchart(y)==(u[:3],u[3:])
            assert step_pointed_index(n,g)==pointed_index(u[:3],u[3:])
            for h in GROUP:
                assert pointed_action(g,pointed_action(h,x))==pointed_action(mul(g,h),x)
        if v not in seen:
            orbit={source(g,v) for g in GROUP};seen.update(orbit);hist[len(orbit)]+=1
    assert len(fibres)==81 and all(len(f)==9 for f in fibres.values())
    assert dict(hist)=={1:1,2:2,4:1,6:24,12:48}
    for g in GROUP:
        fixed[g]=sum(source(g,v)==v for v in z)
    assert sum(fixed.values())==12*76
    R=(1,0,0)
    fixed_cosets=[u for u in fibres if pointed_action(R,(0,0)+u)[2:]==u]
    assert len(fixed_cosets)==9
    lift_counts={u:sum(pointed_action(R,t+u)==t+u for t in fibres[u]) for u in fixed_cosets}
    assert lift_counts[(0,0,0,0)]==9 and all(n==0 for u,n in lift_counts.items() if any(u))
    # In characteristic three, the invariant DC subspace has no equivariant
    # section of its quotient, even set-theoretically: eight nonzero fixed
    # quotient points have no fixed lift. Retain inner-fibre transport.
    return dict(semantic_states=729,central_states=9,outer_states=720,
        fibres=81,states_per_fibre=9,action_edges=729*12,composition_checks=729*144,
        signed_orbits=76,orbit_sizes=dict(sorted(hist.items())),
        rotation_fixed_cosets=9,nonzero_fixed_cosets_without_fixed_lift=8,
        equivariant_product_section_exists=False,actual_monster_action=False)


if __name__=="__main__":
    print(reference_certificate())
    print(pointed729_certificate())
