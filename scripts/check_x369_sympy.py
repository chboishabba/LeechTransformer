import sympy as S

def main():
    x=S.Matrix(S.symbols('a b c', real=True)); h=S.Matrix(S.symbols('A B C', real=True)); g=S.Matrix(S.symbols('D E F', real=True)); w=S.Matrix(S.symbols('s f r', real=True)); s,f,r=w
    K=S.Matrix([[s,r,f],[f,s,r],[r,f,s]])
    def original(h):
        a,b,c=x; A,B,C=h
        return w.dot(S.Matrix([a*A+b*B+c*C,a*B+b*C+c*A,a*C+b*A+c*B]))
    def fused(h): return h.dot(K*x)
    checks={ 'score':original(h)-fused(h), 'pair':original(h)-original(g)-fused(h-g)}
    alpha,beta,u=S.symbols('alpha beta u', real=True)
    loss=alpha*original(h)+beta*original(g); reduced=(alpha*h+beta*g).dot(K*x)
    checks['shared unary cancellation']=(u+original(h))-(u+original(g))-fused(h-g)
    for t in [*x,*h,*g,*w]: checks['gradient '+str(t)]=S.diff(loss-reduced,t)
    P=S.Matrix([[0,1,0],[0,0,1],[1,0,0]])
    for i,v in enumerate(P*K-K*P): checks['C3 commute '+str(i)]=v
    dx=S.Matrix(S.symbols('dx:3', real=True)); dh=S.Matrix(S.symbols('dh:3', real=True)); z=K*x
    checks['perturbation']=(h+dh).dot(z+dx)-h.dot(z)-h.dot(dx)-dh.dot(z)-dh.dot(dx)
    for name,v in checks.items():
        assert S.expand(v)==0,name
    print(f'{len(checks)} exact symbolic residuals zero; SymPy {S.__version__}')
    d=S.symbols('d',real=True); p=1/(1+S.exp(d)); assert S.simplify(S.diff(p,d)+S.exp(d)/(1+S.exp(d))**2)==0
    print('softmax monotonic derivative identity PASS')
if __name__=='__main__': main()
