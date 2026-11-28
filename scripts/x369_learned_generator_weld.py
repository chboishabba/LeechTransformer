"""Training-only exact generator identification, independent sparse residual.
Original Leech-LILA: Anatolii Kornienko; DASHI extension.
Linear composition is a declared inductive bias, not learned nonlinear action.
"""
from itertools import product
from collections import defaultdict
import json
from fractions import Fraction

def fit_generators(rows):
    """Fit only elementary training examples; reject inconsistent/full-rank failures."""
    import sympy as S
    data=defaultdict(list)
    for r in rows:
        if r['split']!='train' or len(r['word'])!=1:continue
        token=r['word'];key=(token,r['guards']['abjn'.index(token)])
        data[key].append((r['mean']+r['residual'],r['mean_label']+r['signed_label']))
    expected={(c,i) for c in 'abjn' for i in range(3 if c in 'ab' else 2)}
    if set(data)!=expected:raise ValueError('incomplete elementary training domain')
    result={}
    for k,records in sorted(data.items()):
        X=S.Matrix([x for x,y in records]);Y=S.Matrix([y for x,y in records])
        C=X.T*X
        if C.rank()!=6:raise ValueError('rank deficient elementary carrier')
        W=(C.inv()*X.T*Y).T
        if X*W.T!=Y:raise ValueError('nonlinear/inconsistent elementary labels')
        if any(v.q!=1 for v in W):raise ValueError('nonintegral learned generator')
        result[k]=tuple(tuple(int(v) for v in row) for row in W.tolist())
    return result

def certify(matrices):
    """Exact learned representation certificate; no target matrices in fitting."""
    import sympy as S
    from models.geometric_reflection_fibre import GROUP,mul
    from scripts.x369_simultaneous_weld import literal_matrix
    I=S.eye(6);M={k:S.Matrix(v) for k,v in matrices.items()}
    R,J,N=M['a',1],M['j',1],M['n',1]
    rel=(R**3==I,J**2==I,N**2==I,J*R*J==R**2,N*R==R*N,N*J==J*N)
    if not all(rel):raise ValueError('learned relation failed')
    # Literal owner: positive translation is inverse donor left roll,
    # reflection is R_left^2 J, negation is central.
    table={g:R**((-g[0]+2*g[1])%3)*J**g[1]*N**g[2] for g in GROUP}
    for g,h in product(GROUP,repeat=2):
        if table[g]*table[h]!=table[mul(g,h)]:raise ValueError('group convention mismatch')
    for k,v in M.items():
        tag='abjn'.index(k[0]);guard=k[1]
        idx=4*((-guard)%3) if tag<2 else 10*guard if tag==2 else guard
        if v!=table[GROUP[idx]]:raise ValueError('literal identification mismatch')
        if v!=S.Matrix(literal_matrix((idx,tag))):raise ValueError('independent action mismatch')
    if any(v.T*v!=I for v in table.values()):raise ValueError('nonorthogonal learned action')
    return dict(relations=6,group_products=144,literals=10,orthogonal=True,
                all_finite_words_ideal=True,training_rule='exact full-rank elementary linear regression')

def predict(matrices,r):
    z=r['mean']+r['residual']
    for c in r['word']:
        W=matrices[c,r['guards']['abjn'.index(c)]]
        z=tuple(sum(a*b for a,b in zip(row,z)) for row in W)
    return z

def audit(rows):
    rows=list(rows);m=fit_generators(rows);cert=certify(m)
    counts=defaultdict(int)
    for r in rows:
        if predict(m,r)!=r['mean_label']+r['signed_label']:raise AssertionError('independent endpoint')
        counts[r['split']]+=1
    return dict(certificate=cert,rows=dict(counts),matrices={f'{c}:{i}':v for (c,i),v in m.items()},
                optimizer_steps=0,nonlinear_learning_claim=False,release_admitted=False)

def main():
    from scripts.audit_x369_joint_action import joint_records
    print(json.dumps(audit(joint_records()),sort_keys=True))
if __name__=='__main__':main()
