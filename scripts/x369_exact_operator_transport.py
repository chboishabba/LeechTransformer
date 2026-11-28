"""Historical operator transport, without fitting or source mutation.
Original Leech-LILA: Anatolii Kornienko; DASHI extension.
Native execution retains bytes; rational algebra describes stored values.
"""
import argparse
import json
from fractions import Fraction as Q
from pathlib import Path
from hashlib import sha256
from itertools import product
import torch
from models.geometric_orbit_descent import canonical_controlled, decode_frame
from models.geometric_stabilizer_codec import encode_history, decode_history, same_bits
from scripts.x369_translate_paired_checkpoint import restore, exclusive
from scripts.x369_capped_semantic_geometry import cap_profile
from scripts.x369_source_geometry import signed_seed
from scripts.x369_source_acceptance import source_identity, model_digest
from scripts.x369_indexed_residual_closure import add, sub, split, join, transport
from scripts.x369_simultaneous_weld import literal_matrix, apply_matrix, simultaneous, verify_joint_realization, repair_check
from scripts.x369_future_realization import pack
from scripts.audit_x369_joint_action import MEANS
from scripts.audit_x369_ordered_signed_task import TRIADS, gram
from scripts.train_x369_ordered_action import LABELS

CONTROLS=tuple((g,t) for t in range(4) for g in ((0,8,4) if t<2 else (0,10) if t==2 else (0,1)))

def key(x):
    if x.is_floating_point() and not torch.isfinite(x).all():
        raise ValueError("nonfinite source")
    a=x.detach().cpu().contiguous()
    return str(a.dtype),tuple(a.shape),a.reshape(-1).view(torch.uint8).numpy().tobytes()

class Historical:
    def __init__(self,old):
        self.old=old
        p=next(old.parameters())
        self.device,self.dtype=p.device,p.dtype
        self.shape=(1,old.experts,2,old.width)
        if old.training or old.width<9 or self.device.type!='cuda' or not torch.version.hip or not torch.cuda.get_device_properties(self.device).gcnArchName.startswith('gfx803'):
            raise ValueError("native frozen paired gfx803 source required")
        self.matrices={u:literal_matrix(u) for u in CONTROLS}

    def reopen(self,h):
        return torch.frombuffer(bytearray(h),dtype=self.dtype).reshape(self.shape).to(self.device)

    def history(self,h):
        return tuple(tuple(tuple(Q(float(x)) for x in b) for b in e) for e in self.reopen(h)[0].tolist())

    def values(self,h):
        return tuple(x for e in self.history(h) for b in e for x in b)

    def decode(self,h):
        # Exact rational version of the existing constructed decoder.
        w=self.old.width
        a,b=h[:w],h[w:2*w]
        return tuple((a[j]+b[j]+a[3+j]+b[3+j])/2 for j in range(3))+tuple((a[j]-b[j]+a[6+j]-b[6+j])/2 for j in range(3))

    def section(self,z):
        m,d=z[:3],z[3:];w=self.old.width
        h=[Q(0)]*(self.old.experts*2*w)
        for j in range(3):
            h[j]=(m[j]+d[j])/2;h[w+j]=(m[j]-d[j])/2
            h[3+j]=h[w+3+j]=m[j]/2
            h[6+j]=d[j]/2;h[w+6+j]=-d[j]/2
        return tuple(h)

    def action(self,u,z):
        return apply_matrix(self.matrices[u],z)

    @torch.no_grad()
    def seed(self,z):
        s=torch.tensor([tuple(z)+(0,0,0)],device=self.device,dtype=self.dtype)
        return key(self.old.initial(s))[2]

    @torch.no_grad()
    def once(self,h,u):
        g,t=u
        v={0:0,8:1,4:2}[g] if t<2 else int(g!=0)
        op=torch.tensor([[int(t==j) for j in range(4)]+[int(v==j) for j in range(3)]],device=self.device,dtype=self.dtype)
        world=self.reopen(h)
        cc,ch,mark,K=canonical_controlled(torch.tensor([g],device=self.device),world,True)
        reopened=decode_frame(decode_history(encode_history(ch)),mark,True)
        if not same_bits(world,reopened):
            raise RuntimeError("marked history did not reopen")
        x=self.old.embed(op)
        a=self.old.history.step(x,world,train_sign=False)
        b=self.old.history.step(x,reopened,train_sign=False)
        fields=('branches','history','ids','signs','weights')
        if any(key(getattr(a,k))!=key(getattr(b,k)) for k in fields):
            raise RuntimeError("historical operator parity failure")
        la=self.old.consume(a.branches,a.history)
        lb=self.old.consume(b.branches,b.history)
        if tuple(map(key,la))!=tuple(map(key,lb)):
            raise RuntimeError("historical consumer parity failure")
        obs=tuple((k,key(getattr(b,k))) for k in fields if k!='history')+tuple((f'logits_{j}',key(x)) for j,x in enumerate(lb))
        return obs,key(b.history)[2],(int(cc.item()),int(mark.item()))

    def step(self,h,u):
        a=self.once(h,u)
        if a!=self.once(h,u):
            raise RuntimeError("nonrepeatable historical edge")
        return a[:2]

    def packet(self,h,z):
        values=self.values(h)
        decoded,r=split(values,self.decode,self.section)
        d=sub(decoded,z)
        if join((add(z,d),r),self.section)!=values:
            raise AssertionError("signed reconstruction")
        return h,tuple(z),d,r

    def advance(self,p,u):
        h,z,d,r=p
        if p!=self.packet(h,z):
            raise ValueError("invalid incoming carrier")
        obs,n=self.step(h,u)
        target=self.action(u,z)
        result=self.packet(n,target)
        rec=transport(self.values(h),self.decode,self.decode,self.section,self.section,lambda _:self.values(n),lambda y:self.action(u,y))
        if result[2]!=add(self.action(u,d),rec['defect']):
            raise AssertionError("ordered signed cocycle")
        return obs,result

    def chart(self,h):
        return self.decode(self.values(h)),cap_profile(self.history(h))

    def score(self,obs,z):
        values=dict(obs)
        for j,(labels,y) in enumerate(zip(LABELS,(z[:3],gram(z[3:]),z[3:]))):
            _,_,raw=values[f'logits_{j}']
            v=torch.frombuffer(bytearray(raw),dtype=self.dtype).tolist()
            winners=[i for i,x in enumerate(v) if x==max(v)]
            target=labels.index(y)
            yield f'logits_{j}', 'correct_unique' if winners==[target] else 'wrong_unique' if len(winners)==1 else 'tie_target' if target in winners else 'tie_without_target'

    def relation_audit(self,z):
        # Equivalent target words are compared on actual recurrent paths.
        R,J,N=(8,0),(10,2),(1,3)
        pairs=(((R,R,R),()),((J,J),()),((N,N),()),((J,R,J),(R,R)),((N,R),(R,N)),((N,J),(J,N)))
        start=self.packet(self.seed(z),z)
        def run(word):
            p=start
            trace=[]
            for u in word:
                obs,p=self.advance(p,u)
                trace.append(obs)
            return p,tuple(trace)
        reports=[]
        for left,right in pairs:
            a,ao=run(left);b,bo=run(right)
            if a[1]!=b[1]:raise AssertionError("independent target relation")
            delta=sub(a[2],b[2]);residual=sub(a[3],b[3])
            if join((delta,residual),self.section)!=sub(self.values(a[0]),self.values(b[0])):
                raise AssertionError("relation defect reconstruction")
            reports.append(dict(left=left,right=right,target_equal=True,historical_bytes_equal=a[0]==b[0],signed_difference=pack(delta),residual_difference=pack(residual),left_final_observation=pack(ao[-1] if ao else None),right_final_observation=pack(bo[-1] if bo else None)))
        return reports

    def seed_dictionary(self):
        import sympy as S
        basis=tuple(tuple(int(i==j) for i in range(6)) for j in range(6))
        B=S.Matrix([self.values(self.seed(z)) for z in basis])
        s=torch.eye(6,device=self.device,dtype=self.dtype)
        A=S.Matrix(signed_seed(torch.cat((s,torch.zeros_like(s[:,:3])),1),self.old.experts,self.old.width).flatten(1).tolist()).applyfunc(S.Rational)
        rank=B.rank()
        if rank!=6:return dict(rank=rank,reverse_available=False)
        L=B.T*(B*B.T).inv()
        assert B*L==S.eye(6)
        def mat(x):return [[str(v) for v in row] for row in x.tolist()]
        return dict(rank=rank,reverse_available=True,historical_basis=mat(B),target_basis=mat(A),historical_section=mat(L),forward=mat(L*A),scope="linear seed image only")

    def audit(self,means,budget,seconds):
        seeds=tuple((self.seed(m+d),m+d,i) for i,(m,d) in enumerate(product(means,TRIADS)))
        cap=self.old.history.source.history_cap
        cap=None if cap is None else Q(float(cap))
        def repair(obs,n,target,delta):
            ids=tuple(torch.frombuffer(bytearray(dict(obs)['ids'][2]),dtype=torch.long).tolist())
            return repair_check(self.history(n),target,ids,cap,delta)
        # Exercise the carrier on each graph edge without inserting target labels
        # into historical state identity. The graph owner propagates labels separately.
        def checked_step(h,u):
            obs,p=self.advance(self.packet(h,self.decode(self.values(h))),u)
            return obs,p[0]
        result=simultaneous(seeds,CONTROLS,checked_step,self.chart,self.history,self.action,budget=budget,seconds=seconds,export=True,constructed=lambda h:self.decode(self.values(h)),score=self.score,repair=repair)
        if 'joint_realization' in result:
            result['independent_replay']=verify_joint_realization(result['joint_realization'],seeds,CONTROLS,self.chart,self.step,self.action)
        return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--sha256',required=True)
    p.add_argument('--source-head',required=True)
    p.add_argument('--phase-manifest',type=Path)
    p.add_argument('--phase-manifest-sha256')
    p.add_argument('--phase-receipt',type=Path)
    p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--budget',type=int,default=1000)
    p.add_argument('--seconds',type=float,default=90)
    a=p.parse_args()
    if a.receipt.exists() or a.budget<1 or not 0<a.seconds<=90 or len({bool(a.phase_manifest),bool(a.phase_manifest_sha256),bool(a.phase_receipt)})!=1:
        p.error("invalid scope or phase inputs")
    root=Path(__file__).resolve().parents[1]
    source=source_identity(root,a.source_head)
    if not torch.cuda.is_available() or not torch.version.hip or not torch.cuda.get_device_properties(0).gcnArchName.startswith('gfx803'):
        raise RuntimeError("native gfx803 required")
    old,m,rows,probe=restore(a,root,torch.device('cuda',0))
    before=model_digest(old)
    owner=Historical(old)
    means=MEANS if m['joint_mean'] else ((1,0,0),)
    seed_dictionary=owner.seed_dictionary()
    relations=owner.relation_audit(tuple(means[0])+tuple(TRIADS[0]))
    result=owner.audit(means,a.budget,a.seconds)
    if before!=model_digest(old) or source!=source_identity(root,a.source_head):
        raise RuntimeError("source mutation")
    receipt=dict(kind='x369_exact_operator_transport',source=source,historical_sha256=a.sha256,metadata=m,trace=probe,phase_manifest_sha256=a.phase_manifest_sha256,phase_receipt_sha256=None if not a.phase_receipt else sha256(a.phase_receipt.read_bytes()).hexdigest(),audit=result,seed_dictionary=seed_dictionary,relations=relations,training_steps=0,optimizer_migrated=False,universal_full_function_certified=False,release_admitted=False)
    data=(json.dumps(receipt,indent=2,allow_nan=False,default=lambda x: str(x) if isinstance(x,Q) else (_ for _ in ()).throw(TypeError(type(x).__name__)))+'\n').encode()
    exclusive(a.receipt,data)
    print(sha256(data).hexdigest())

if __name__=='__main__':
    main()
