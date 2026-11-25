"""Source-legal cap-profile separators and exact affine sphere-fibre test.
Original Leech-LILA: Anatolii Kornienko; DASHI extension.
No fitted-model, reachability, native gfx803 or kernel claim.
"""
from fractions import Fraction as F


def fibre_affine_certificate(offset, blocks, radii):
    """Exact rational criterion on the ENTIRE product of independent spheres.

    y = offset + sum_i blocks[i] w_i, ||w_i|| = radii[i].
    Returns iff tests for y itself and squared norm of y.
    Zero radius blocks are removed; guarded subsets require a separate proof.
    """
    import sympy as s
    b=s.Matrix(offset)
    if b.cols!=1 or len(blocks)!=len(radii):raise ValueError('invalid affine fibre')
    active=[]
    for A,r in zip(blocks,radii):
        A=s.Matrix(A);r=s.sympify(r)
        if A.rows!=b.rows or A.cols<1 or r.is_Rational is not True or r<0:
            raise ValueError('invalid block or radius')
        if not all(x.is_Rational is True for x in A):raise ValueError('exact rational block required')
        if r:active.append((A,r))
    if not all(x.is_Rational is True for x in b):raise ValueError('exact rational offset required')
    linear=[];anisotropy=[];cross=[];lambdas=[]
    for i,(A,r) in enumerate(active):
        G=A.T*A; lam=s.trace(G)/A.cols;lambdas.append(lam)
        if A.T*b!=s.zeros(A.cols,1):linear.append(i)
        if G!=lam*s.eye(A.cols):anisotropy.append(i)
        for j,(B,_) in enumerate(active[:i]):
            if A.T*B!=s.zeros(A.cols,B.cols):cross.append((j,i))
    energy_constant=not(linear or anisotropy or cross)
    return dict(vector_constant=all(A==s.zeros(*A.shape) for A,_ in active),
        energy_constant=energy_constant,offset_cross=linear,anisotropy=anisotropy,
        inter_block_cross=cross,constant_energy=(b.dot(b)+sum(l*r*r for l,(_,r) in zip(lambdas,active))) if energy_constant else None)


def algebra_certificate():
    import sympy as s
    I=s.eye(2);Z=s.zeros(2)
    a=fibre_affine_certificate([0,0],[I],[1]);assert a['energy_constant'] and not a['vector_constant'] and a['constant_energy']==1
    assert fibre_affine_certificate([0,0],[s.diag(1,0)],[1])['anisotropy']==[0]
    assert fibre_affine_certificate([1,0],[I],[1])['offset_cross']==[0]
    assert fibre_affine_certificate([0,0],[I,I],[1,1])['inter_block_cross']==[(0,1)]
    assert fibre_affine_certificate([0,0],[I,Z],[0,2])['vector_constant']
    A=I.col_join(Z);B=Z.col_join(I)
    r=fibre_affine_certificate([0]*4,[A,B],[2,3])
    assert r['energy_constant'] and r['constant_energy']==13
    return dict(affine_controls=6,exact=True)


def source_certificate():
    """Run the real original sparse step with explicitly synthetic weights.
    CPU float64 fixture only; input states are legal capped states, not
    asserted reachable from the original signed-seed or fitted checkpoint.
    """
    import torch
    from models.history_orbit_moe import HistoryOrbitMoE
    from models.geometric_orbit_descent import OrbitDescentHistoryMoE
    from models.geometric_reflection_fibre import GROUP
    from scripts.x369_capped_semantic_geometry import cap_profile
    def profile(h):
        return cap_profile(tuple(tuple(tuple(F(float(v)) for v in b) for b in e) for e in h))
    def pair(indices, values):
        h=torch.zeros(4,2,12,dtype=torch.float64)
        h[1,:,list(indices)]=torch.tensor(values,dtype=h.dtype)
        return h
    def make(bias,router=False,project=False):
        m=HistoryOrbitMoE(d_model=12,num_experts=4,top_k=2,expert_width=12,route_geometry='c2_c3_residual').double()
        m.history_cap=1.
        with torch.no_grad():
            for p in m.parameters():p.zero_()
            m.expert_bias.copy_(torch.tensor(bias,dtype=torch.float64))
            if router:m.router.weight[0,12:]=1
            m.orientation.weight[0,0]=1
            if project:m.transitions[1].weight[:3,:3]=torch.eye(3,dtype=torch.float64)
        return OrbitDescentHistoryMoE(m,paired=True).eval()
    def run(m,a,b):
        assert profile(a)==profile(b) and not torch.equal(a,b)
        assert not a[0].any() and not b[0].any()
        h=torch.stack((a,b));assert (h.square().sum(-1)<=1).all()
        control=torch.full((2,),GROUP.index((0,0,0)),dtype=torch.long)
        with torch.no_grad():out=m.step_controlled(control,h,lambda c:torch.ones(len(c),12,dtype=h.dtype))
        assert not out.output.any() and not out.history[:,0].any()
        p,q=map(profile,out.history)
        assert p!=q
        return out.ids.tolist(),(p[1][1],q[1][1])
    route=run(make([3,0,1.5,-1],router=True),pair([9],[1]),pair([0,3,6,9],[.5]*4))
    assert route==([[0,2],[0,1]],((1,1),(0,0)))
    fixed=run(make([3,2,1,0],project=True),pair([0],[1]),pair([9],[1]))
    assert fixed==([[0,1],[0,1]],((1,1),(0,0)))
    return dict(route_separator=route,fixed_route_successor_separator=fixed,
        training_steps=0,reachable_from_seed=False,fitted=False,native_gpu=False)


if __name__=='__main__':
    print(algebra_certificate())
    print(source_certificate())
