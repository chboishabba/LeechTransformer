"""Exact staged GELU/cap pullbacks. Original Leech-LILA: Anatolii Kornienko.
DASHI extension; ideal-real, fixed admitted guard, not native/fitted proof.
Interpolation nodes are algebraic probes, NOT legal source words.
"""
from itertools import combinations
import sympy as s
from scripts.x369_affine_future_profile import (
    rational_matrix, affine_lift, linear_form, energy_form, compile_future, encode)


def stage_nodes(base, directions):
    T=rational_matrix(base);D=tuple(map(rational_matrix,directions));n=T.rows
    if T.shape!=(n,n) or list(T[-1,:])!=[0]*(n-1)+[1] or any(
        M.shape!=T.shape or any(M[-1,:]) for M in D):
        raise ValueError('invalid homogeneous stage')
    yield ('zero',),T
    for i,M in enumerate(D):
        yield ('plus',i),T+M
        yield ('minus',i),T-M
    for i,j in combinations(range(len(D)),2):
        yield ('pair',i,j),T+D[i]+D[j]


def compile_stages(stages,forms):
    """Sufficient common span for arbitrary real coefficients at each stage.
    Every coefficient/guard must itself factor through code or be an input.
    No minimal actual-word claim: nodes overapproximate legal switches.
    """
    stages=tuple((name,base,tuple(ds)) for name,base,ds in stages)
    if len({x[0] for x in stages})!=len(stages):raise ValueError('duplicate stage')
    maps=[((name,)+key,T) for name,base,ds in stages for key,T in stage_nodes(base,ds)]
    c=compile_future(maps,forms)
    c['stage_sizes']={name:len(ds) for name,_,ds in stages}
    c['actual_word_witnesses']=False
    return c


def stage_matrix(c,name,alpha):
    alpha=tuple(map(s.sympify,alpha))
    if len(alpha)!=c['stage_sizes'][name]:raise ValueError('wrong coefficient count')
    M=c['transitions'];Z=M[(name,'zero')];out=Z.copy()
    for i,a in enumerate(alpha):
        P,N=M[(name,'plus',i)],M[(name,'minus',i)]
        out+=a*(P-N)/2+a*a*(P+N-2*Z)/2
    for i,j in combinations(range(len(alpha)),2):
        out+=alpha[i]*alpha[j]*(M[(name,'pair',i,j)]-M[(name,'plus',i)]-M[(name,'plus',j)]+Z)
    return out


def branch_stage(first,b1,last,b2,transition,bt,x,sign):
    """Literal selected-branch equations from PairedHistoryMoE.step.
    Fixed local activation and admitted sign; pre-cap update only.
    first reads (x,h), transition reads (h,x,y_c).
    """
    first,b1,last,b2,transition,bt,x=map(rational_matrix,(first,b1,last,b2,transition,bt,x))
    d=x.rows;k=first.rows
    if sign not in (-1,0,1) or x.cols!=1 or first.shape!=(k,2*d) or b1.shape!=(k,1) or last.shape!=(d,k) or b2.shape!=(d,1) or transition.shape!=(d,3*d) or bt.shape!=(d,1):
        raise ValueError('invalid selected branch')
    C=sign*first[:,d:];c=sign*first[:,:d]*x+b1
    L,M,N=transition[:,:d],transition[:,d:2*d],transition[:,2*d:]
    base=affine_lift(sign**2*L,sign**2*M*x+sign*(N*b2+bt))
    B=sign*N*last;ds=[]
    for j in range(k):
        T=s.zeros(d+1);T[:d,d]=B[:,j];ds.append(T)
    pre=tuple(linear_form(C[j,:],c[j]) for j in range(k))
    return dict(base=base,directions=tuple(ds),preactivation_forms=pre,
                output_matrix=sign*last,output_bias=sign*b2)


def cap_stage(d):
    return affine_lift(s.zeros(d),s.zeros(d,1)),(s.diag(*([1]*d+[0])),)


def assemble_selected(branches, width, active):
    """Assemble selected branch stages in flattened local [expert,branch,w].
    Active maps are supplied from the same source guard. Others are identity.
    Each selected branch has an independent radial scale; no recapping inactive.
    Frame/stabilizer are separate admitted linear stages, not inferred here.
    """
    if type(branches)!=int or branches<1 or type(width)!=int or width<1:
        raise ValueError('invalid carrier')
    n=branches*width;pre=s.eye(n+1);cap=s.eye(n+1)
    ds=[];scales=[];forms=[];energies=[];owners=[]
    for i,a in sorted(active.items()):
        if type(i)!=int or not 0<=i<branches or a['base'].shape!=(width+1,width+1):
            raise ValueError('invalid selected branch')
        lo=i*width;hi=lo+width;T=a['base'];pre[lo:hi,lo:hi]=T[:width,:width];pre[lo:hi,n]=T[:width,width]
        cap[lo:hi,lo:hi]=s.zeros(width);D=s.zeros(n+1);D[lo:hi,lo:hi]=s.eye(width);scales.append(D)
        H=D.copy();energies.append(H)
        for j,M in enumerate(a['directions']):
            D=s.zeros(n+1);D[lo:hi,lo:hi]=M[:width,:width];D[lo:hi,n]=M[:width,width]
            ds.append(D);owners.append((i,j))
        for H in a['preactivation_forms']:
            Q=s.zeros(n+1);ix=list(range(lo,hi))+[n]
            for j,r in enumerate(ix):
                for k,c in enumerate(ix):Q[r,c]=H[j,k]
            forms.append(Q)
    return dict(pre=(pre,tuple(ds)),cap=(cap,tuple(scales)),preactivation_forms=tuple(forms),
                cap_forms=tuple(energies),activation_owners=tuple(owners),cap_owners=tuple(sorted(active)))


def assembly_certificate():
    d=12;I=s.eye(d);z=s.zeros(d,1);x=s.ones(d,1)
    first=s.ones(1,2*d);last=s.eye(d)[:,:1];T=(I/2).row_join(I).row_join(I)
    active={i:branch_stage(first,s.zeros(1,1),last,z,T,z,x,sign) for i,sign in [(0,1),(1,-1),(4,0),(5,1)]}
    a=assemble_selected(8,d,active);h=s.Matrix([s.Rational(i%7-3,8) for i in range(96)]);X=h.col_join(s.ones(1,1))
    values=[s.Rational(i+1,3) for i in range(len(a['activation_owners']))]
    def apply(stage,coeff,X):
        B,D=stage;return (B+sum((c*M for c,M in zip(coeff,D)),s.zeros(B.rows)))*X
    P=apply(a['pre'],values,X);expected=h.copy()
    for k,(i,j) in enumerate(a['activation_owners']):
        assert j==0;lo=i*d;hi=lo+d;b=active[i];local=h[lo:hi,0].col_join(s.ones(1,1))
        expected[lo:hi,0]=apply((b['base'],b['directions']),[values[k]],local)[:d,0]
        assert (X.T*a['preactivation_forms'][k]*X)[0]==(local.T*b['preactivation_forms'][0]*local)[0]
        assert (P.T*a['cap_forms'][k]*P)[0]==expected[lo:hi,0].dot(expected[lo:hi,0])
    assert P==expected.col_join(s.ones(1,1))
    coeff=[s.Rational(1,i+2) for i in range(len(active))];C=apply(a['cap'],coeff,P)
    for j,i in enumerate(sorted(active)):expected[i*d:(i+1)*d,0]*=coeff[j]
    assert C==expected.col_join(s.ones(1,1))
    for i in set(range(8))-set(active):assert C[i*d:(i+1)*d,0]==h[i*d:(i+1)*d,0]
    return dict(history_width=96,selected_branches=4,inactive_branches=4,preactivation_checks=4,cap_energy_checks=4,independent_scales=4)


def certificate():
    d=3;I=s.eye(d);z=s.zeros(d,1)
    # One source hidden atom; all three orientation cases checked symbolically.
    first=s.Matrix([[1,0,0,1,0,0]]);b1=s.Matrix([0]);last=s.Matrix([1,0,0]);b2=z
    T=(I/2).row_join(s.zeros(d)).row_join(I);x=s.Matrix([1,0,0]);h=s.Matrix(s.symbols('h0:3'));g=s.Matrix([s.Symbol('g')])
    for sign in (-1,0,1):
        a=branch_stage(first,b1,last,b2,T,z,x,sign);X=h.col_join(s.ones(1,1))
        P=a['base']+g[0]*a['directions'][0]
        expected=sign*T*(sign*h).col_join(sign*x).col_join(last*g+b2)
        assert s.expand((P*X)[:d,0]-expected)==z
        assert (X.T*a['preactivation_forms'][0]*X)[0]==(first*(sign*x).col_join(sign*h)+b1)[0].expand()
    a=branch_stage(first,b1,last,b2,T,z,x,1);cb,cd=cap_stage(d)
    H=energy_form(I,z);L=linear_form([1,0,0]);forms=(L,H)+a['preactivation_forms']
    c=compile_stages([('gelu',a['base'],a['directions']),('cap',cb,cd)],forms)
    assert c['rank']==3
    h=s.Matrix([s.Rational(1,16),s.Rational(1,12),s.Rational(-1,16)]);assert h.dot(h)<s.Rational(1,16)
    p=(first*x.col_join(h)+b1)[0];g=p*(1+s.erf(p/s.sqrt(2)))/2
    v=(a['base']+g*a['directions'][0])*h.col_join(s.ones(1,1));v=v[:d,0]
    k=encode(c,h);k1=stage_matrix(c,'gelu',[g])*k
    assert (k1-encode(c,v)).applyfunc(s.simplify)==s.zeros(c['rank'],1)
    assert s.simplify((c['decoder']*k1)[1]-v.dot(v))==0
    # Force actual active cap by a sufficiently small threshold; p>1 implies v0>1/2.
    alpha=s.Rational(1,4)/s.sqrt(v.dot(v));k2=stage_matrix(c,'cap',[alpha])*k1
    assert (k2-encode(c,alpha*v)).applyfunc(s.simplify)==s.zeros(c['rank'],1)
    assert s.simplify((c['decoder']*k2)[1]-s.Rational(1,16))==0
    # Mixed coefficient interpolation, including nonzero offsets.
    B=affine_lift(s.eye(2),s.Matrix([1,0]));D=s.zeros(3);D[0,2]=1;E=s.zeros(3);E[1,0]=1
    c2=compile_stages([('mix',B,(D,E))],[energy_form(s.eye(2),s.zeros(2,1))])
    u,v=s.symbols('u v');w=s.Matrix([2,-3]);direct=((B+u*D+v*E)*w.col_join(s.ones(1,1)))[:2,0]
    assert (stage_matrix(c2,'mix',[u,v])*encode(c2,w)-encode(c2,direct)).applyfunc(s.expand)==s.zeros(c2['rank'],1)
    return dict(branch_sign_cases=3,gelu_cap_rank=c['rank'],active_cap_exact=True,mixed_rank=c2['rank'],actual_word_witnesses=False,training_steps=0)


if __name__=='__main__':
    print(certificate())
    print(assembly_certificate())
