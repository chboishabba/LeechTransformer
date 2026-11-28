import sympy as S
import torch
from models.x369_differential_backward import directed_vjp
def symbolic():
    x=S.Matrix(S.symbols('a b c')); h=S.Matrix(S.symbols('A B C')); w=S.Matrix(S.symbols('s f r')); s,f,r=w
    K=S.Matrix([[s,r,f],[f,s,r],[r,f,s]])
    a,b,c=x; A,B,C=h
    T=S.Matrix([a*A+b*B+c*C,a*B+b*C+c*A,a*C+b*A+c*B]); L=h.dot(K*x)
    checks=[L-w.dot(T),*(S.Matrix([L]).jacobian(x).T-K.T*h),*(S.Matrix([L]).jacobian(h).T-K*x),*(S.Matrix([L]).jacobian(w).T-T),sum(T)-(a+b+c)*(A+B+C)]
    g1,g2=S.symbols('g1 g2'); j=S.Matrix(S.symbols('J K L')); q=S.Matrix(S.symbols('U V W'))
    checks+=list(g1*(h-j)+g2*(q-j)-(-g1-g2)*j-g1*h-g2*q)
    assert all(S.expand(v)==0 for v in checks)
    print('SymPy',S.__version__,len(checks),'zero residuals')
def numeric(device):
    for dtype in (torch.float32,torch.float64):
        for relative in (False,True):
            torch.manual_seed(42)
            x=torch.randn(3,384,dtype=dtype,device=device,requires_grad=True); h=torch.randn(3,4,384,dtype=dtype,device=device,requires_grad=True); w=torch.randn(1,384,dtype=dtype,device=device,requires_grad=True)
            a,b,c=x.reshape(3,1,-1,3).unbind(-1); A,B,C=h.reshape(3,4,-1,3).unbind(-1)
            t=torch.stack((a*A+b*B+c*C,a*B+b*C+c*A,a*C+b*A+c*B),-1).reshape_as(h)
            y=torch.nn.functional.linear(t,w).squeeze(-1); y=y-y[:,:1] if relative else y
            z=directed_vjp(x,h,w,relative=relative); g=torch.randn_like(y)
            u=torch.autograd.grad(y,(x,h,w),g); v=torch.autograd.grad(z,(x,h,w),g)
            tol=1e-4 if dtype==torch.float32 else 1e-10
            for a,b in zip((y,*u),(z,*v)): torch.testing.assert_close(a,b,atol=tol,rtol=tol)
            print(dtype,relative,[float((a-b).detach().abs().max()) for a,b in zip((y,*u),(z,*v))])
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--gpu',action='store_true'); a=p.parse_args()
    if a.gpu and not torch.cuda.is_available(): raise RuntimeError('GPU required')
    symbolic(); numeric('cuda' if a.gpu else 'cpu')
