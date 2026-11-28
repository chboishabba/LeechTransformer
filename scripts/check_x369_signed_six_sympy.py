"""Independent local SymPy/SignedSix exact acceptance; never a native GPU claim.

Original Leech-LILA by Anatolii Kornienko; DASHI extension.
Derives every literal action from the original two-register endpoint, not
from fitted generator matrices, a task label shortcut, or a neural model.
"""
import json
from itertools import product
from sympy import Matrix, eye, __version__ as sympy_version
from models.geometric_reflection_fibre import GROUP,mul
from scripts.audit_x369_ordered_signed_task import endpoint
from scripts.x369_costed_constructive import action,apply,compose,inverse,Block,literal_blocks,power,reopen


def basis_matrix(letter,guard):
    tag='abjn'.index(letter)
    guards=tuple(guard if j==tag else 0 for j in range(4))
    columns=[]
    for j in range(6):
        v=tuple(int(i==j) for i in range(6))
        m,d,_=endpoint(letter,guards,v[:3],v[3:])
        columns.append(m+d)
    return Matrix.hstack(*(Matrix(col) for col in columns))


def certificate():
    M={(letter,guard):basis_matrix(letter,guard)
       for letter in 'abjn'
       for guard in range(3 if letter in 'ab' else 2)}
    I=eye(6)
    R,J,N=(M['a',1],M['j',1],M['n',1])
    relations=(R**3==I,J**2==I,N**2==I,J*R*J==R**2,
               N*R==R*N,N*J==J*N)
    assert all(relations)
    group={g:R**((-g[0]+2*g[1])%3)*J**g[1]*N**g[2] for g in GROUP}
    for g,h in product(GROUP,repeat=2):
        assert group[g]*group[h]==group[mul(g,h)]
    assert all(W.T*W==I for W in group.values())
    for (letter,guard),W in M.items():
        tag='abjn'.index(letter)
        idx=4*((-guard)%3) if tag<2 else 10*guard if tag==2 else guard
        assert group[GROUP[idx]]==W

    matrices={k:tuple(tuple(int(x) for x in row) for row in W.tolist())
              for k,W in M.items()}
    blocks=literal_blocks(matrices)
    keys=tuple(sorted(blocks))
    seed=(1,0,-1,2,0,-2)
    pair_checks=0
    for a,b in product(blocks.values(),repeat=2):
        joined=a.then(b)
        assert apply(joined.action,seed)==apply(b.action,apply(a.action,seed))
        assert apply(inverse(joined.action),apply(joined.action,seed))==seed
        assert reopen(joined,keys)==(a.first,b.first)
        pair_checks+=1

    binary=blocks['a',1].then(blocks['j',1])
    for n in range(100):
        x,ledger=power(binary,n,True)
        assert reopen(x,keys)==(('a',1),('j',1))*n
        assert apply(x.action,seed)==tuple((J*R)**n*Matrix(seed))
        assert ledger['action_compositions']<=2*n.bit_length()
        if n:
            assert ledger['history_result_bits']==8*n
    huge=10**100
    x,ledger=power(binary,huge,False)
    assert x.extent==2*huge
    assert ledger['action_compositions']<=2*huge.bit_length()
    assert not x.history_retained
    try:reopen(x,keys)
    except ValueError:pass
    else:raise AssertionError('discarded history reopened')
    return dict(status='local_sympy_exact',sympy=sympy_version,
                relations=sum(relations),group_products=len(GROUP)**2,
                literal_actions=len(M),orthogonal=len(GROUP),
                signed_block_pairs=pair_checks,binary_power_cases=100,
                huge_repetition=str(huge),huge_extent_bits=x.extent.bit_length(),
                huge_action_compositions=ledger['action_compositions'],
                history_discard_refusal=True,optimizer_steps=0,
                agda_kernel_checked=False,native_gpu_checked=False)


if __name__=='__main__':
    print(json.dumps(certificate(),sort_keys=True))
