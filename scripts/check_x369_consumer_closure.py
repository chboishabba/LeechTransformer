"""Exact geometry and factorial regression; no original model import or training."""
from itertools import product
import sympy as s
import torch
from audit_x369_consumer_closure import factorial_states,factorial_effect


def main():
    I=s.eye(3);O=s.zeros(3);P=I.row_join(O);T=O.row_join(I).col_join(I.row_join(O))
    # Source-legal ideal-real anchor: selected FFN reads first triad by
    # GELU(z)-GELU(-z)=z; transition swaps triads. Shared strict route/sign.
    assert P.rank()==3 and P.col_join(P*T).rank()==6 and T*T==s.eye(6)
    R=s.Matrix([[0,0,1],[1,0,0],[0,1,0]]);J=s.Matrix([[1,0,0],[0,0,1],[0,1,0]])
    for g in (R,-J,-I):
        G=s.diag(g,g);assert T*G==G*T and P*G==g*P
    states=list(product((-1,0,1),repeat=6))
    one={v[:3] for v in states};two={(v[:3],v[3:]) for v in states}
    assert len(one)==27 and len(two)==729
    # Ordered observer collision with pure phase residual hidden from DC chart.
    a=s.Matrix([1,1,1,0,0,0]);b=s.Matrix([1,1,1,1,-1,0]);assert P*a==P*b and P*T*a!=P*T*b
    # Identity control: same hidden directions never visible at any horizon.
    assert P.col_join(P*s.eye(6)).rank()==3
    # Four-way intervention exact decomposition; deliberate chart and residual errors.
    z=torch.tensor([[1.,2.,3.,4.]],dtype=torch.float64);target=torch.tensor([[2.,4.,5.,7.]],dtype=z.dtype)
    V=torch.eye(4,dtype=z.dtype)[:,:2];q=z@V;qg=target@V;A=torch.eye(2,dtype=z.dtype)
    arms=factorial_states(z,target,V,q,qg,A)
    assert all(torch.equal(a,torch.tensor([b],dtype=z.dtype)) for a,b in zip(arms,([1,2,3,4],[2,4,3,4],[1,2,5,7],[2,4,5,7])))
    assert torch.equal((arms[0]-target).square().sum(),(arms[1]-target).square().sum()+(arms[2]-target).square().sum())
    # Multiplicative consumer: effects on output are not additive.
    outs=[v[:,0]*v[:,2] for v in arms];c,r,m=factorial_effect(outs);assert (c.item(),r.item(),m.item())==(3.,2.,2.)
    # GELU pair identity numerical control for all 729 ternary six-vectors.
    H=torch.tensor(states,dtype=torch.float64);g=torch.nn.functional.gelu
    assert torch.allclose(g(H[:,:3])-g(-H[:,:3]),H[:,:3],atol=1e-15,rtol=0)
    print('PASS',s.__version__,torch.__version__,'729 states; 27->729 trace classes; ranks 3->6; 3 generator commutations; factorial and nonlinear mixed term; GELU readout')
if __name__=='__main__':main()
