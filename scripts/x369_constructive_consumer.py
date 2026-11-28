"""Constructive semantic and ordered residual execution, no Torch dependency.
Original Leech-LILA: Anatolii Kornienko; DASHI extension.
The source-owned matrices must be admitted before construction.
"""
from dataclasses import dataclass
from scripts.x369_ordered_residual_codec import latch_step,ALPHABET

@dataclass(frozen=True)
class State:
    semantic:tuple
    residual:object

class Consumer:
    def __init__(self,matrices,mode):
        if mode not in ('semantic','first','history'):raise ValueError('consumer')
        expected={(c,g) for c in ALPHABET for g in range(3 if c in 'ab' else 2)}
        if set(matrices)!=expected:raise ValueError('literal domain')
        self.mode=mode;self.actions={}
        for k,W in matrices.items():
            if len(W)!=6 or any(len(row)!=6 for row in W):raise ValueError('dimension')
            p=[]
            for row in W:
                nz=[i for i,v in enumerate(row) if v]
                if len(nz)!=1 or row[nz[0]] not in (-1,1):raise ValueError('signed permutation')
                p.append((nz[0],row[nz[0]]))
            if len({i for i,s in p})!=6:raise ValueError('nonbijective action')
            self.actions[k]=tuple(p)
    def initial(self,z):
        if len(z)!=6:raise ValueError('dimension')
        return State(tuple(z),(0,0) if self.mode=='history' else None)
    def step(self,state,u):
        p=self.actions[u];z=tuple(s*state.semantic[i] for i,s in p)
        r=state.residual
        if self.mode=='first':r=latch_step(r,u[0])
        elif self.mode=='history':
            # Preserve all TEN literal/guard choices, not merely four tags.
            # Base 16 permits fixed-width nibble push/pop with explicit extent.
            v,n=r
            digit=tuple(sorted(self.actions)).index(u)
            r=(16*v+digit,n+1)
        return State(z,r)
    def observe(self,state):
        if self.mode=='semantic':return state.semantic
        if self.mode=='first':return state.semantic,state.residual
        return state.semantic,self.reopen(state)
    def reopen(self,state):
        if self.mode!='history':raise ValueError('history not retained')
        v,n=state.residual;out=[];keys=tuple(sorted(self.actions))
        if type(v) is not int or type(n) is not int or min(v,n)<0 or v.bit_length()>4*n:raise ValueError('invalid history')
        for _ in range(n):
            v,d=divmod(v,16)
            if d>=len(keys):raise ValueError('invalid literal digit')
            out.append(keys[d])
        return tuple(reversed(out))
    def run(self,z,word):
        h=self.initial(z)
        for u in word:h=self.step(h,u)
        return h

def certificate():
    from scripts.x369_learned_generator_weld import fit_generators,certify
    from scripts.audit_x369_joint_action import joint_records
    rows=list(joint_records());M=fit_generators(rows);cert=certify(M)
    semantic=Consumer(M,'semantic');history=Consumer(M,'history');first=Consumer(M,'first')
    count=0
    for r in rows:
        z=r['mean']+r['residual'];w=tuple((c,r['guards']['abjn'.index(c)]) for c in r['word'])
        h=history.run(z,w);s=semantic.run(z,w)
        assert h.semantic==s.semantic==r['mean_label']+r['signed_label']
        assert history.reopen(h)==w
        if w[0][0] in 'ab':assert first.run(z,w).residual==w[0][0]
        count+=1
    return dict(rows=count,certificate=cert,history_payload_bits_per_indexed_literal=4,optimizer_steps=0)
if __name__=='__main__':
    import json
    print(json.dumps(certificate(),sort_keys=True))
