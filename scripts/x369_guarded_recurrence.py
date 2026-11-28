"""Composed ideal-real controlled source reference; not a native checkpoint.
Original Leech-LILA: Anatolii Kornienko; DASHI extension.
Exact rational input guard evaluation; symbolic successor may require a
separate certified guard decision before another step.
"""
import sympy as s
from fractions import Fraction as F
from scripts.x369_guard_geometry import frame_receipt,shape,flatten,transform
from scripts.x369_route_orientation_weld import route_scores,route_receipt,orientation
from scripts.x369_capped_semantic_geometry import cap_profile,exact_repair
from models.geometric_reflection_fibre import inv


def rational(v):
    v=s.sympify(v)
    if v.is_Rational is not True:raise ValueError('rational guard carrier required')
    return F(int(v.p),int(v.q))


def rational_history(h):return tuple(tuple(tuple(map(rational,b)) for b in e) for e in h)


def gelu(p):return p*(1+s.erf(p/s.sqrt(2)))/2


def controlled_step(h,u,embed,base,directed,bias,w,experts,cap=None):
    """Full source ordering; embeds indexed by canonical control, tag captured.
    experts[e]=(first,b1,last,b2,transition,bt), exact rational matrices.
    Strict routes only. Unknown/irrational guard decisions are rejected.
    No semantic target, fitted admission, native ties, or training.
    """
    E=len(h);W=len(h[0][0]);h=rational_history(h)
    f=frame_receipt(u,h);local=shape(f['history'],E,W);x=tuple(map(rational,embed[f['control']]))
    scores=route_scores(x,local,base,directed,bias);ids=tuple(sorted(range(E),key=lambda i:scores[i],reverse=True)[:2]);r=route_receipt(scores,ids)
    if not r['strict']:raise ValueError('native tie authority required')
    gap=s.Rational(-r['relative'][0]);weights=(1/(1+s.exp(-gap)),1/(1+s.exp(gap)))
    raw=[list(e) for e in local];y=[s.zeros(W,1),s.zeros(W,1)];signs=[];pre=[];energy=[]
    if cap is not None and rational(cap)<=0:raise ValueError('positive cap required')
    for slot,e in enumerate(ids):
        A,a,B,b,T,t=map(s.Matrix,experts[e]);ss=[]
        for j in range(2):
            old=local[e][j];k=orientation(x,old,w,j);ss.append(k)
            X=s.Matrix(x);H=s.Matrix(old);p=A*(k*X).col_join(k*H)+a;pre.append(tuple(p))
            yc=B*p.applyfunc(gelu)+b;v=(k*(T*(k*H).col_join(k*X).col_join(yc)+t)).applyfunc(s.simplify)
            n=s.simplify(v.dot(v));energy.append(n)
            if cap is not None:
                # Literal ideal-real norm floor; native double/cast separate.
                c=s.Rational(cap);floor=s.Rational(1,10**30);alpha=s.Min(1,c/s.Max(s.sqrt(n),floor));v=(alpha*v).applyfunc(s.simplify)
            raw[e][j]=tuple(v);y[j]+=weights[slot]*k*yc
        signs.append(tuple(ss))
    def project(v):
        rows=[flatten(transform(k,v)) for k in f['stabilizer']]
        return shape(tuple(s.simplify(sum(a)/len(rows)) for a in zip(*rows)),len(v),W)
    loc=project(raw);world=shape(flatten(transform(inv(f['mark']),loc)),E,W)
    world=tuple(world[e] if e in ids else h[e] for e in range(E))
    output=transform(inv(f['mark']),project((tuple(tuple(v) for v in y),)))[0:2]
    return dict(history=world,output=output,frame=f,ids=ids,relative=r['relative'],weights=weights,signs=tuple(signs),preactivation=tuple(pre),precap_energy=tuple(energy),local_history=loc,raw_local_history=tuple(tuple(e) for e in raw))


def current_code(h,out):
    """Deliberately strong candidate: current guard+coefficients+current outputs.
    Not asserted sufficient for the next control.
    """
    return (exact_repair(h,(0,)*6,range(len(h)),1)['defect'],cap_profile(h),out['frame']['maximizers'],out['ids'],out['relative'],out['signs'],out['preactivation'],out['precap_energy'],out['output'])


def certificate():
    W=12;E=4;I=s.eye(W);Z=s.zeros(W,1);zero=(0,)*W
    def e(*entries):return tuple(dict(entries).get(i,0) for i in range(W))
    anchor=e((0,1));p=e((0,F(3,5)),(9,F(4,5)));q=e((0,F(3,5)),(9,F(-4,5)))
    def history(v):return ((anchor,anchor),(zero,zero),(v,v),(zero,zero))
    a,b=history(p),history(q)
    u=(0,0,0);v=(0,0,1);emb={u:e((0,1)),v:e((9,1),(10,1),(11,1))}
    # C3 self channel on fourth triad only, zero on first control.
    base=zero;directed=e((9,1));bias=(3,1,1-F(1,4),-1);w=(1,)*W+zero
    first=s.zeros(1,2*W);last=s.zeros(W,1);T=I.row_join(s.zeros(W,2*W));params=(first,s.zeros(1,1),last,Z,T,Z);experts=(params,)*4
    args=(emb,base,directed,bias,w,experts,1)
    A=controlled_step(a,u,*args);B=controlled_step(b,u,*args)
    assert current_code(a,A)==current_code(b,B) and A['ids']==B['ids']==(0,1)
    assert cap_profile(rational_history(A['history']))==cap_profile(rational_history(B['history']))
    C=controlled_step(A['history'],v,*args);D=controlled_step(B['history'],v,*args)
    assert C['ids']==(0,2) and D['ids']==(0,1)
    # Stronger: one control, identical full receipt now, different guard after
    # the SAME control. Selected expert1 swaps triads 2 and 3, preserving norm.
    from scripts.check_x369_nonlinear_closure import refine,witness
    def state(j,sgn):
        t=e((0,F(3,5)),(j,sgn*F(4,5)))
        return ((anchor,anchor),(t,t),(zero,zero),(zero,zero))
    a,b=state(9,1),state(9,-1);states=(a,b,state(6,1),state(6,-1))
    swap=I.copy();swap[6:12,6:12]=s.zeros(3).row_join(s.eye(3)).col_join(s.eye(3).row_join(s.zeros(3)))
    rotated=(first,s.zeros(1,1),last,Z,swap.row_join(s.zeros(W,2*W)),Z)
    args2=({u:e((6,1),(7,1),(8,1))},zero,e((6,1)),(3,1,F(3,4),-1),w,(params,rotated,params,params),1)
    table={}
    for h in states:
        out=controlled_step(h,u,*args2);table[h]=(current_code(h,out),rational_history(out['history']))
    assert table[a][0]==table[b][0] and cap_profile(table[a][1])==cap_profile(table[b][1])
    chart=lambda h:(exact_repair(h,(0,)*6,range(E),1)['defect'],cap_profile(h))
    partitions,edges=refine(states,(u,),chart,lambda h,_:table[h])
    counts=[len(set(p.values())) for p in partitions]
    assert counts==[1,3,4] and witness(a,b,(u,),chart,edges)==(u,u)
    # Separate literal nonlinear branch check: all-zero history, full stabilizer,
    # bias-only forcing on a selected branch, exact GELU and cap before Reynolds.
    h=tuple((zero,zero) for _ in range(E));first=s.zeros(1,2*W);last=I[:,:1];T=s.zeros(W,2*W).row_join(I)
    nonlinear=(first,s.ones(1,1),last,Z,T,Z)
    R=controlled_step(h,u,emb,zero,zero,(3,2,1,0),w,(nonlinear,)*E,s.Rational(1,4))
    g=gelu(s.S.One);assert all(s.simplify(n-g*g)==0 for n in R['precap_energy'])
    assert all(s.simplify(sum(z*z for z in R['raw_local_history'][i][j])-s.Rational(1,16))==0 for i in (0,1) for j in (0,1))
    # Global-negation stabilizer annihilates both projected history and output.
    assert not any(z for row in flatten(R['history']) for z in row) and not any(z for row in R['output'] for z in row)
    return dict(current_receipt_collision=True,next_guard_separator=True,next_routes=(C['ids'],D['ids']),same_control_shortest_word=2,finite_partition_counts=counts,finite_edges=len(edges),nonlinear_precap_checks=4,active_cap_checks=4,stabilizer_zero=True,training_steps=0)


if __name__=='__main__':print(certificate())
