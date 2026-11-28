import sympy as S

def main():
    a,b,c,A,B,C,ws,wf,wr,es,ef,ed=S.symbols('a b c A B C ws wf wr es ef ed')
    s=a*A+b*B+c*C; f=a*B+b*C+c*A; r=a*C+b*A+c*B
    D=(a+b+c)*(A+B+C)
    full=ws*s+wf*f+wr*r
    two=(ws-wr)*s+(wf-wr)*f+wr*D
    M=S.Matrix([[-1,-1],[1,0]]); G=S.Matrix([[2,1],[1,2]])
    U=S.Matrix([[1,0],[0,1],[-1,-1]])
    P=S.Matrix([[0,0,1],[1,0,0],[0,1,0]])
    d1,d2,k=S.symbols('d1 d2 k')
    checks=[s+f+r-D, D-s-f-r,full-two, (ws-wr)*(s+es)+(wf-wr)*(f+ef)+wr*(D+ed)-two-((ws-wr)*es+(wf-wr)*ef+wr*ed), *((M**3-S.eye(2))), *((M.T*G*M-G)), *((P*U-U*M)), *((U.T*U-G)), *((S.Matrix([s,f]).T*G*S.Matrix([s,f]))-S.Matrix([[s*s+f*f+(-s-f)**2]])), *((S.Matrix([two-full]).jacobian([a,b,c,A,B,C,ws,wf,wr]))), (d1+k)-(d2+k)-(d1-d2)]
    assert all(S.expand(v)==0 for v in checks)
    # Exact aggregate cancellation with unequal per-triad DC.
    assert 2*1+3*4==2*4+3*2
    assert 1!=4 and 4!=2
    print(f'{len(checks)} symbolic zero residuals; aggregate DC witness PASS; SymPy {S.__version__}')
if __name__=='__main__': main()
