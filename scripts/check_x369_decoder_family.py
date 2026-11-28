"""Exact source-shaped decoder family tests; no fitted or GPU claim."""
from fractions import Fraction as Q
from itertools import product
from scripts.x369_exact_semantic_decoder import ExactSystem,fit,decode,verify_decoder_family


def certificate():
    zero=(0,)*3;e=(1,0,0);y=e+zero
    means=(zero,)+tuple(tuple(s*int(i==j) for i in range(3)) for j in range(3) for s in (-1,1))
    triads=tuple(product((-1,0,1),repeat=3))
    def history(m,d,offset=0):
        h=[[[Q(0)]*12 for _ in range(2)] for _ in range(4)]
        for i in range(3):
            h[0][0][offset+i]=Q(m[i]+d[i]);h[0][1][offset+i]=Q(m[i]-d[i])
        return tuple(tuple(tuple(b) for b in p) for p in h)
    seeds=tuple((i,history(m,d),m+d) for i,(m,d) in enumerate(product(means,triads)))
    f=fit(seeds,include_family=True);c=tuple(map(Q,f['coefficients']));N=tuple(tuple(map(Q,v)) for v in f['nullspace'])
    assert f['consistent'] and f['rank']==6 and f['parameters']==96 and f['family_dimension']==90
    assert c==(1,1,1,0,1,0)+(0,)*90 and f['free_columns']==tuple(range(6,96))
    for _,h,v in seeds:
        assert decode(h,c)==v
        assert all(decode(h,n)==(0,)*6 for n in N)
    def audit(rows):return verify_decoder_family(rows,c,N)
    h=history(e,zero,3);k=history((2,0,0),zero,3)
    a=audit((('adjust',h,y),));assert not a['canonical']['exact'] and a['joint_compatible']
    b=audit((('forced',history(e,zero),(-1,0,0)+zero),));assert not b['joint_compatible'] and b['scalar_rows']['forced_mismatch']==1
    r=audit((('a',h,y),('b',k,y)));assert not r['joint_compatible'] and r['scalar_rows']['forced_mismatch']==0
    assert audit((('a',h,y),))['joint_compatible'] and audit((('b',k,y),))['joint_compatible']
    assert Q(r['joint_conflict']['contradiction'])!=0
    # Nontrivial pivot/free dependency and affine offset.
    s=ExactSystem(3);s.insert((1,2,3),4,'first');s.insert((0,1,2),5,'second')
    assert s.solution()==(-6,5,0) and s.affine_family()['nullspace']==((1,-2,1),)
    assert s.solution({2:7})==(1,-9,7)
    print(dict(seeds=len(seeds),parameters=96,rank=6,free=90,adjustable=True,forced_mismatch=True,joint_conflict=True,training_steps=0))


def parser_certificate():
    from scripts.x369_source_acceptance import parse_args
    from contextlib import redirect_stderr
    from io import StringIO
    base=['--checkpoint','source.pt','--sha256','test','--source-head','test','--output','receipt.json']
    a=parse_args(base);assert a.future_budget==0 and not a.future_export
    for scope in ('fixed','all-literals'):
        a=parse_args(base+['--future-budget','729','--future-scope',scope,'--future-export'])
        assert a.future_budget==729 and a.future_export and a.future_scope==scope
    for args in (['--future-export'],['--future-budget','-1'],['--future-budget','1','--width','6']):
        with redirect_stderr(StringIO()):
            try:parse_args(base+args)
            except SystemExit as e:assert e.code==2
            else:raise AssertionError(args)
    print('PASS: legacy defaults, 2 future scopes, 3 fail-closed parser cases')


if __name__=='__main__':
    certificate()
    parser_certificate()
