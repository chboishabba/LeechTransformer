"""Direct projected solver regression on actual label actions and hard fixtures."""
import itertools
import torch
from models.geometric_reflection_fibre import GROUP,mul
from scripts.x369_source_geometry import label_permutations
from scripts.x369_projected_direct import solve_direct,collision_bounds,label_orbits
from scripts.x369_translation_repair import projection_groups


def certificate():
    torch.manual_seed(41);torch.set_num_threads(2);dt=torch.float64
    X=torch.randn(16,5,dtype=dt);Y=torch.randn(16,3,dtype=dt);B0=torch.randn(5,3,dtype=dt)
    table=torch.tensor([[0,1,2],[1,0,2]]);masks=torch.tensor([[1,0],[1,1]],dtype=torch.bool).repeat(8,1)
    B,r=solve_direct(X,Y,masks,table,B0)
    groups=projection_groups(X,Y,masks,table)
    normal=sum(torch.kron(G.contiguous(),P.T.contiguous()) for G,_,P,_,_ in groups)+.001*torch.eye(15,dtype=dt)
    rhs=sum(H-G@B0@P for G,H,P,_,_ in groups)
    ref=B0+torch.linalg.solve(normal,rhs.flatten()).reshape_as(B0)
    assert torch.allclose(B,ref,atol=1e-8,rtol=1e-8) and r['converged']
    subgroups=[]
    for mask in range(1,1<<12,2):
        K=[g for i,g in enumerate(GROUP) if mask>>i&1]
        if all(mul(g,h) in K for g in K for h in K):subgroups.append([g in K for g in GROUP])
    assert len(subgroups)==16
    X=torch.randn(945,121,dtype=dt)
    X[:,40:80]=X[:,:40]+1e-8*X[:,40:80];X[:,80:120]=X[:,:40]*100;X[:,120]=1
    masks=torch.tensor(subgroups[:10],dtype=torch.bool)[torch.arange(945)%10]
    tables=label_permutations()
    for j in range(3):
        table=torch.tensor([v[j] for v in tables]);n=table.shape[1]
        Y=torch.nn.functional.one_hot(torch.randint(n,(945,)),n).double();B0=torch.randn(121,n,dtype=dt)
        B,r=solve_direct(X,Y,masks,table,B0)
        assert r['converged'] and r['regularized_objective']<r['original_sse'] and r['feature_directions_retained']==121
        assert sorted(map(len,label_orbits(table)))==([1,1,3,3,3,3] if j==1 else [1,2,6,6,6,6])
        print('head',j,'relative residual',max(r['normal_residual'],r['gram_normal_residual'])/r['initial_residual'])
    X=torch.tensor([[1.,0.],[1.,0.],[2.,0.],[2.,0.]],dtype=dt);Y=torch.eye(3,dtype=dt)[torch.tensor([0,1,0,2])]
    masks=torch.tensor([[1,0],[1,0],[1,1],[1,1]],dtype=torch.bool);table=torch.tensor([[0,1,2],[1,0,2]])
    r=collision_bounds(X,Y,masks,table,list(range(4)))
    assert r['exact_sse_lower_bound']=='9/4' and r['unprojected_sse_lower_bound']=='2'
    assert r['ideal_projected_unique_correct_upper_bound']==r['deterministic_unique_correct_upper_bound']==2
    assert r['conflicting_keys']==2 and r['first_conflict']['local_labels']==[0,1]
    X=torch.randn(3,7,dtype=dt);Y=torch.randn(3,3,dtype=dt);B0=torch.randn(7,3,dtype=dt)
    masks=torch.tensor([[1,0],[1,1],[1,0]],dtype=torch.bool);B,r=solve_direct(X,Y,masks,table,B0)
    assert r['converged'] and r['feature_directions_retained']==7
    print('PASS 15x15 reference, 3 hard 945x121 heads, exact collision floor, wide null directions; zero optimizer steps')


if __name__=='__main__':certificate()
