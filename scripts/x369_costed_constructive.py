"""Costed exact constructive execution; no GPU performance inference.
Original Leech-LILA: Anatolii Kornienko; DASHI extension.
"""
from dataclasses import dataclass
from itertools import product

IDENTITY=tuple((i,1) for i in range(6))

def action(W):
    p=[]
    if len(W)!=6 or any(len(r)!=6 for r in W):raise ValueError('dimension')
    for r in W:
        j=[i for i,x in enumerate(r) if x]
        if len(j)!=1 or r[j[0]] not in (-1,1):raise ValueError('signed permutation')
        p.append((j[0],r[j[0]]))
    if len({i for i,s in p})!=6:raise ValueError('not bijective')
    return tuple(p)

def compose(a,b):
    """a after b, with six indexed reads and sign products."""
    return tuple((b[i][0],s*b[i][1]) for i,s in a)

def apply(a,z):return tuple(s*z[i] for i,s in a)

def inverse(a):
    out=[None]*6
    for j,(i,s) in enumerate(a):out[i]=(j,s)
    return tuple(out)

@dataclass(frozen=True)
class Block:
    action:tuple=IDENTITY
    extent:int=0
    first:object=None
    value:int=0
    history_retained:bool=True
    def __post_init__(self):
        if len(self.action)!=6 or {i for i,s in self.action}!=set(range(6)) or any(s not in (-1,1) for i,s in self.action):raise ValueError('invalid signed action')
        if type(self.history_retained) is not bool:raise ValueError('invalid retention flag')
        if not self.history_retained and self.value:raise ValueError('discarded history payload')
        if (self.extent==0)!=(self.first is None):raise ValueError('first/extent mismatch')
        if type(self.extent) is not int or self.extent<0 or type(self.value) is not int or self.value<0 or self.value.bit_length()>4*self.extent:raise ValueError('invalid extent')
    def then(self,b,history=True):
        if history and not (self.history_retained and b.history_retained):raise ValueError('history discarded')
        return Block(compose(b.action,self.action),self.extent+b.extent,
                     self.first if self.extent else b.first,
                     (self.value<<(4*b.extent))|b.value if history else 0,history)

def literal_blocks(matrices):
    keys=tuple(sorted(matrices))
    if len(keys)!=10:raise ValueError('literal domain')
    return {u:Block(action(matrices[u]),1,u,i) for i,u in enumerate(keys)}

def power(block,n,history=False):
    """Binary powering; charge group compositions separately from big integers.
    history=False intentionally discards literal digits, not extent.
    """
    if type(n) is not int or n<0:raise ValueError('nonnegative repetition')
    result=Block();compositions=0;digit_copy_bound=0
    def join(a,b):
        nonlocal compositions,digit_copy_bound
        compositions+=1
        if history:digit_copy_bound+=4*(a.extent+b.extent)
        return a.then(b,history)
    while n:
        if n&1:result=join(result,block)
        n>>=1
        if n:block=join(block,block)
    return result,dict(action_compositions=compositions,coordinate_reads=6*compositions,
        sign_products=6*compositions,history_result_bits=4*result.extent if history else 0,
        history_copy_bit_volume_upper_model=digit_copy_bound,
        extent_bits=result.extent.bit_length(),physical_cost_certified=False)

def reopen(b,keys):
    if len(keys)!=10 or len(set(keys))!=10:raise ValueError('literal domain')
    if not b.history_retained:raise ValueError('history discarded')
    out=[];v=b.value
    for _ in range(b.extent):
        v,d=divmod(v,16)
        if d>=len(keys):raise ValueError('invalid digit')
        out.append(keys[d])
    result=tuple(reversed(out))
    if result and result[0]!=b.first:raise ValueError('first literal mismatch')
    return result

def certificate():
    from scripts.x369_learned_generator_weld import fit_generators,certify
    from scripts.audit_x369_joint_action import joint_records
    matrices=fit_generators(joint_records());cert=certify(matrices)
    blocks=literal_blocks(matrices);keys=tuple(sorted(blocks));z=(1,0,-1,-1,1,0)
    pairs=0
    for a,b in product(blocks.values(),repeat=2):
        c=a.then(b)
        assert apply(c.action,z)==apply(b.action,apply(a.action,z))
        assert apply(inverse(c.action),apply(c.action,z))==z
        assert reopen(c,keys)==(a.first,b.first);pairs+=1
    block=blocks['a',1].then(blocks['j',1])
    for n in range(33):
        p,ledger=power(block,n,True)
        assert reopen(p,keys)==(('a',1),('j',1))*n
        v=z
        for _ in range(n):v=apply(block.action,v)
        assert apply(p.action,z)==v
    n=10**100
    p,ledger=power(block,n,False)
    assert p.extent==2*n and p.action==IDENTITY
    return dict(generator_certificate=cert,pairs=pairs,powers=33,huge_repetition=str(n),ledger=ledger,optimizer_steps=0)
if __name__=='__main__':
    import json
    print(json.dumps(certificate(),sort_keys=True))
