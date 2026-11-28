import itertools
import torch
import sympy as s
from models.geometric_history_adapter import c3_orientation

T=torch.tensor(list(itertools.product((-1,0,1), repeat=6)),dtype=torch.float64)
R=lambda t:t.reshape(-1,2,3).roll(-1,-1).reshape_as(t)
q=c3_orientation(T)
assert torch.equal(c3_orientation(-T),-q)
assert torch.equal(c3_orientation(R(T)),q)
assert torch.equal(c3_orientation(R(R(T))),q)
assert torch.equal(q.squeeze(-1)==0,(T==0).all(-1))
x=torch.tensor([[1.,-1.,0.,0.,0.,0.]],dtype=torch.float64)
assert x[0,0]==1 and R(x)[0,0]==-1
assert torch.equal(c3_orientation(x),c3_orientation(R(x)))
v=torch.tensor([[1.,0.,-1.,0.,0.,0.]],dtype=torch.float64)
assert torch.equal(v.reshape(-1,2,3).flip(-1).reshape_as(v),-v)
r=s.Matrix([[0,1,0],[0,0,1],[1,0,0]])
j=s.Matrix([[0,0,1],[0,1,0],[1,0,0]])
a,b,c=s.symbols('a b c')
w=a*s.eye(3)+b*r+c*r**2
k=r.T*w
assert s.expand(j.T*k*j-k)==(a-c)*(r-r**2)
p=torch.tensor([[1.,-1.,0.],[2.,-2.,0.],[4.,-4.,0.]],dtype=torch.float64)
y=torch.nn.functional.gelu(p)
assert (y.sum(-1)>0).all()
u=torch.tensor([[1.,1.,1.],[-2.,-2.,-2.]],dtype=torch.float64)
assert torch.equal(torch.nn.functional.gelu(u),torch.nn.functional.gelu(u).roll(-1,-1))
print('PASS',torch.__version__,s.__version__,len(T),'two-triad signed charts, reflection obstruction, phase-shift head condition, GELU DC leak')
