import sympy as s
import torch
from models.geometric_cross_triads import CrossTriadBitLinear,basis
from itertools import product
I=s.eye(3); R=s.Matrix([[0,1,0],[0,0,1],[1,0,0]]);J=s.Matrix([[0,0,1],[0,1,0],[1,0,0]])
u=s.Matrix([1,1,1]);p=s.Matrix([1,-1,0]);q=s.Matrix([0,1,-1]);G=[I,R,R**2,J,R*J,u*p.T,u*q.T,p*u.T,q*u.T]
for k in (3,5,9):
 B=s.Matrix.hstack(*(g.reshape(9,1) for g in G[:k]));assert B.rank()==k
 assert torch.equal(basis({3:'c3',5:'s3',9:'full'}[k],dtype=torch.float64),torch.tensor([g.tolist() for g in G[:k]],dtype=torch.float64))
A=s.Matrix.vstack(s.Matrix.hstack(I,R),s.Matrix.hstack(J,I));D=s.Matrix.vstack(s.Matrix.hstack(R,I),s.Matrix.hstack(I,J))
x=s.Matrix(s.symbols('x:6'));h=s.Matrix(s.symbols('h:6'))
for k in range(3):
 M=s.diag(R**k,R**k);assert s.expand((A*x).dot(M*D*h)-x.dot(A.T*M*D*h))==0
x=s.Matrix([1,0,0,1,0,0]);z=s.Matrix([1,0,0,-1,0,0]);assert x[:3]==z[:3];assert x[3:]!=z[3:]
assert sum(x[i]*x[i+3] for i in range(3))!=sum(z[i]*z[i+3] for i in range(3))
torch.manual_seed(42);checks=0
for mode, dims, ternary in product(('c3','s3','full'),((6,9),(9,6),(6,6)),(False,True)):
 layer=CrossTriadBitLinear(*dims,mode=mode,ternary=ternary,dtype=torch.float64)
 with torch.no_grad():layer.coefficients.copy_(torch.randint(-1,2,layer.coefficients.shape).double());layer.bias.copy_(torch.arange(dims[1]).double())
 x=torch.randn(2,4,dims[0],dtype=torch.float64,requires_grad=True)
 y=layer(x);ref=torch.nn.functional.linear(x,layer.dense_weight(),layer.bias)
 assert torch.allclose(y,ref,atol=1e-12,rtol=1e-12)
 args=(x,layer.coefficients,layer.bias)
 g1=torch.autograd.grad(y.square().sum(),args);g2=torch.autograd.grad(ref.square().sum(),args)
 assert all(torch.allclose(a,b,atol=1e-10,rtol=1e-12) for a,b in zip(g1,g2))
 src=torch.nn.Linear(*dims,dtype=torch.float64)
 with torch.no_grad():src.weight.copy_(layer.dense_weight());src.bias.copy_(layer.bias)
 copy=CrossTriadBitLinear.from_linear(src,layer.coefficients.detach(),mode=mode,ternary=ternary)
 assert torch.allclose(copy(x),src(x),atol=1e-12,rtol=1e-12)
 with torch.no_grad():src.weight[0,0]+=0.25
 try:CrossTriadBitLinear.from_linear(src,layer.coefficients.detach(),mode=mode,ternary=ternary)
 except ValueError:pass
 else:raise AssertionError('nonmatching source accepted')
 checks+=1
print('PASS',s.__version__,torch.__version__,'basis ranks 3/5/9; directed contractions 3; CPU forward/backward/adapter cases',checks)
