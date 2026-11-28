"""Independent exact seed and projected normal-equation regression."""
import torch
import sympy as S
from scripts.x369_translation_repair import exact_seed_split,solve_projected,projection_groups
from scripts.x369_source_geometry import label_permutations
from models.geometric_reflection_fibre import GROUP,inv


def certificate():
    A=S.eye(3).row_join(S.eye(3)).col_join(S.eye(3).row_join(-S.eye(3))).row_join(S.zeros(6,6))
    L=A.T*(A*A.T).inv()
    B=S.Matrix(6,12,lambda i,j:S.Rational((i+2*j+3)%7,8))
    mat=lambda M:[[str(v) for v in row] for row in M.tolist()]
    for old in (A,B):
        r=exact_seed_split(dict(constructed_basis=mat(A),historical_basis=mat(old),constructed_coordinates=mat(L)),2,3)
        E,R=S.Matrix(r['equivariant_basis']),S.Matrix(r['retained_residual'])
        assert E+R==old and r['reverse_solved_from_defects'] and sum(e*z for e,z in zip(E,R))==0
        assert S.Rational(r['equivariant_energy'])+S.Rational(r['residual_energy'])==S.Rational(r['original_energy'])
        if old==A:assert R==S.zeros(6,12)
        else:assert R!=S.zeros(6,12)
    # Actual label-action tables invert the world-to-local pullback.
    tables=label_permutations()
    for j in range(3):
        table=torch.tensor([v[j] for v in tables])
        for i,g in enumerate(GROUP):
            assert torch.equal(table[i][table[GROUP.index(inv(g))]],torch.arange(table.shape[1]))
    torch.manual_seed(41);torch.set_num_threads(1);dt=torch.float64
    X=torch.randn(16,5,dtype=dt);Y=torch.randn(16,3,dtype=dt);B0=torch.randn(5,3,dtype=dt)
    table=torch.tensor([[0,1,2],[1,0,2]])
    masks=torch.tensor([[1,0],[1,1]],dtype=torch.bool).repeat(8,1)
    B,r=solve_projected(X,Y,masks,table,B0)
    groups=projection_groups(X,Y,masks,table)
    normal=sum((torch.kron(G.contiguous(),P.T.contiguous()) for G,_,P,_,_ in groups),torch.zeros(15,15,dtype=dt))+.001*torch.eye(15,dtype=dt)
    rhs=sum((H-G@B0@P for G,H,P,_,_ in groups),torch.zeros_like(B0))
    ref=B0+torch.linalg.solve(normal,rhs.flatten()).reshape_as(B0)
    assert r['converged'] and torch.allclose(B,ref,atol=1e-8,rtol=1e-8)
    assert r['regularized_objective']<=r['original_sse'] and r['solved_sse']>=r['stabilizer_irreducible_sse']-1e-10
    try:projection_groups(X,Y,torch.tensor([[0,1]],dtype=torch.bool).expand(len(X),2),table)
    except ValueError:pass
    else:raise AssertionError('non-subgroup accepted')
    print('PASS exact seed sector split and reverse seam, 36 label frame inverse checks, projected normal solve, stabilizer floor, non-subgroup refusal')


if __name__=='__main__':certificate()
