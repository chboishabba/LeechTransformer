"""Exact consumer-indexed ordered residuals; no neural training.
Original Leech-LILA: Anatolii Kornienko; DASHI extension.
Reuse the extent-retaining radix stack principle of DASHI PR #1548.
"""
from dataclasses import dataclass
from itertools import product
import json

ALPHABET=('a','b','j','n')
BASE=len(ALPHABET)

@dataclass(frozen=True)
class OrderedHistory:
    value:int=0
    extent:int=0
    def __post_init__(self):
        if type(self.value) is not int or type(self.extent) is not int or self.extent<0 or self.value<0 or self.value.bit_length()>2*self.extent:
            raise ValueError('invalid radix carrier')
    def push(self,u):
        return OrderedHistory(BASE*self.value+ALPHABET.index(u),self.extent+1)
    def pop(self):
        if not self.extent:raise ValueError('empty history')
        q,r=divmod(self.value,BASE)
        return OrderedHistory(q,self.extent-1),ALPHABET[r]
    def decode(self):
        h=self;out=[]
        while h.extent:
            h,u=h.pop();out.append(u)
        return tuple(reversed(out))
    @property
    def payload_bits(self):
        # Fixed-length payload width; extent/header/allocator charged separately.
        return 2*self.extent

def encode(word):
    h=OrderedHistory()
    for u in word:h=h.push(u)
    return h

def latch_step(r,u):
    if r not in (None,'a','b'):raise ValueError('invalid latch')
    if u not in ALPHABET:raise ValueError('invalid input')
    if r is None:
        if u not in ('a','b'):raise ValueError('outside declared initial domain')
        return u
    return r

def first_literal(word):
    r=None
    for u in word:r=latch_step(r,u)
    return r

def certificate(depth=7):
    if not 0<=depth<=9:raise ValueError('bounded audit scope')
    from scripts.audit_x369_ordered_signed_task import endpoint
    checks=0;pairs=0
    for n in range(depth+1):
        keys=set()
        for w in product(ALPHABET,repeat=n):
            h=encode(w)
            assert h.decode()==w and h not in keys
            keys.add(h);checks+=1
            if w:
                p,u=h.pop();assert p.push(u)==h
            if w and w[0] in ('a','b'):assert first_literal(w)==w[0]
        assert len(keys)==BASE**n
    # Independent semantic producer, all guards and seed coordinates.
    from scripts.audit_x369_ordered_signed_task import GUARDS,TRIADS
    from scripts.audit_x369_joint_action import MEANS
    for g,m,d,suffix in product(GUARDS,MEANS,TRIADS,('','a','j','n','jn','abjn')):
        a='ab'+suffix;b='ba'+suffix
        assert endpoint(a,g,m,d)==endpoint(b,g,m,d)
        assert first_literal(a)=='a' and first_literal(b)=='b'
        pairs+=1
    return dict(codec_roundtrips=checks,semantic_collision_pairs=pairs,
                endpoint_ceiling_numerator=1,endpoint_ceiling_denominator=2,
                exact_latch_accuracy=1,initialized_latch_bits=1,
                empty_state_retained=True,full_history_payload_bits_per_literal=2,
                optimizer_steps=0,physical_runtime_certified=False)

if __name__=='__main__':print(json.dumps(certificate(),sort_keys=True))
