"""Independent algebra/geometry regression; not native trained-model acceptance."""
import itertools
from types import SimpleNamespace
import torch
from models.geometric_reflection_fibre import GROUP,mul
from models.geometric_orbit_descent import act_tensor,conjugate_control
from scripts.x369_frame_control_weld import (SEEDS,FIT_GUARDS,initial,transformed_seed,
    fit_control,bound_delta,leverage,control_collisions)
from scripts.x369_source_geometry import literal_controls
from scripts.train_x369_ordered_action import encode_inputs


def certificate():
    torch.set_num_threads(2);torch.manual_seed(41);dt=torch.float64
    s=torch.tensor([v+(0,0,0) for v in itertools.product((-1,0,1),repeat=6)],dtype=dt)
    model=SimpleNamespace(experts=2,width=3)
    E=torch.randn(6,12,dtype=dt);B=torch.randn(6,12,dtype=dt);count=0
    for u in (0,1,4,8,10):
        c=torch.full((len(s),1),u,dtype=torch.long)
        for mode in SEEDS:
            a=initial(model,s,c,mode,E,B)
            for g in GROUP:
                b=initial(model,transformed_seed(g,s),conjugate_control(g,c),mode,E,B)
                assert torch.allclose(b,act_tensor(g,a,True),atol=1e-12,rtol=1e-12)
                count+=len(s)
    assert count==131220
    for n,d in ((48,7),(945,97),(5,12)):
        X=torch.randn(n,d,dtype=dt);X[:,-1]=X[:,0];Y=torch.randn(n,3,dtype=dt);base=torch.randn(3,dtype=dt)
        D,r=fit_control(X,Y,base,.01)
        ref=torch.linalg.solve(X.T@X+.01*torch.eye(d,dtype=dt),X.T@(Y-base))
        assert r['converged'] and torch.allclose(D,ref,rtol=1e-7,atol=1e-8)
        assert r['bounded_sse']<=r['unbounded_sse']+1e-9 and r['regularized_objective']<=r['static_sse']
    assert torch.equal(bound_delta(torch.zeros(2,3,dtype=dt),0.),torch.zeros(2,3,dtype=dt))
    X=torch.tensor([[1.,0.,0.],[2.,0.,0.]],dtype=dt);H=torch.tensor([[0.,3.,4.]],dtype=dt)
    r=leverage(X,H,.01);assert r['rank']==1 and abs(r['outside_train_span_rms']-5)<1e-12 and abs(r['heldout_ridge_leverage_max']-2500)<1e-9
    r=control_collisions(torch.tensor([[1.,2.],[1.,2.],[2.,1.]],dtype=dt),torch.tensor([[1.,2.],[3.,4.],[7.,8.]],dtype=dt))
    assert r['conflicting_keys']==1 and r['exact_unrestricted_sse_floor']=='4'
    keys=set()
    for guards in FIT_GUARDS:
        _,o=encode_inputs(dict(word='abjn',guards=guards,mean=(0,0,0),residual=(0,0,0)),'signed')
        c,t=literal_controls(torch.tensor([o],dtype=dt))
        for u,j in zip(c.flatten(),t.flatten()):
            keys.add(int(j)*12+max(int(conjugate_control(g,u)) for g in GROUP))
    assert keys=={0,8,12,20,24,34,36,37}
    R,J,N=GROUP[8],GROUP[10],GROUP[1];I=GROUP[0]
    assert mul(R,mul(R,R))==mul(J,J)==mul(N,N)==I
    assert mul(J,mul(R,J))==mul(R,R) and mul(N,R)==mul(R,N) and mul(N,J)==mul(J,N)
    print('PASS',count,'seed covariance comparisons; 3 SVD ridge cases; radius, leverage, exact collision, 8 control keys and group relations')


if __name__=='__main__':certificate()
