"""Exact historical seed chart and marked action, no regression or training.
Original Leech-LILA: Anatolii Kornienko; DASHI extension.
"""
import argparse,json
from fractions import Fraction as Q
from hashlib import sha256
from pathlib import Path
from itertools import product
import sympy as S
from scripts.x369_exact_semantic_decoder import equations
from scripts.x369_future_realization import unpack,pack
from scripts.x369_simultaneous_weld import literal_matrix
from scripts.x369_source_future_acceptance import future_acceptance
from scripts.x369_future_realization import verify_realization
from scripts.x369_capped_semantic_geometry import cap_profile
from scripts.x369_exact_operator_transport import CONTROLS,Historical
from scripts.x369_translate_paired_checkpoint import restore,exclusive
from scripts.x369_source_acceptance import source_identity,model_digest
from scripts.audit_x369_joint_action import MEANS
from scripts.audit_x369_ordered_signed_task import TRIADS

def row(x):return S.Matrix(1,len(x),x)
def vec(x):return tuple(Q(v) for v in x)
def matrix(x):return S.Matrix([[S.Rational(v) for v in r] for r in x])

class SeedAction:
    def __init__(self,B,A):
        self.B,self.A=B,A
        if B.rows!=6 or A.rows!=6 or B.rank()!=6 or A.rank()!=6:
            raise ValueError("rank-six seed images required")
        self.L=B.T*(B*B.T).inv()
        self.K=A.T*(A*A.T).inv()
        self.P=self.L*B
        self.C=S.eye(B.cols)-self.P
        assert B*self.L==S.eye(6) and self.C*self.L==S.zeros(B.cols,6)
        self.actions={u:matrix(literal_matrix(u)).T for u in CONTROLS}
    def split(self,h):
        x=h*self.L;r=h-x*self.B
        assert r*self.L==S.zeros(1,6) and x*self.B+r==h
        return x,r
    def act(self,h,u):
        x,r=self.split(h)
        return x*self.actions[u]*self.B+r
    def target(self,h):
        x,r=self.split(h)
        return x*self.A,r
    def reverse(self,y,r):
        x=y*self.K
        if x*self.A!=y or r*self.L!=S.zeros(1,6):
            raise ValueError("invalid target image or complement")
        return x*self.B+r
    def certificate(self):
        # Work in the exact six-dimensional representation and prove the
        # complementary action is identity; avoid dense 96x96 products.
        R,J,N=(8,0),(10,2),(1,3)
        a,b,c=(self.actions[u] for u in (R,J,N));I=S.eye(6)
        assert a**3==I and b**2==I and c**2==I and b*a*b==a**2 and c*a==a*c and c*b==b*c
        for M in self.actions.values():
            assert M*M.T==I
        assert self.P*self.P==self.P and self.C*self.C==self.C and self.P*self.C==S.zeros(self.B.cols)
        return dict(rank=6,relations=6,complement_dimension=self.B.cols-6,identity_complement=True)

def replay_conflict(receipt):
    f=receipt['audit']['decoder_fit']
    if f['kind']!='linear_conflict' or f['consistent']:
        raise ValueError("expected exact linear obstruction")
    records={w['id']:w for w in f['witness_records']}
    total=[Q(0)]*f['parameters'];rhs=Q(0)
    for i,j,m,digest in f['conflict']:
        w=records[i];h=unpack(w['history']);z=unpack(w['label'])
        if sha256(str(h).encode()).hexdigest()!=digest:raise ValueError("history digest")
        eq=equations(h)[j];m=Q(m)
        if len(eq)!=len(total):raise ValueError("row dimension")
        total=[x+m*y for x,y in zip(total,eq)];rhs+=m*z[j]
    if any(total) or rhs!=Q(f['contradiction']) or not rhs:
        raise ValueError("invalid contradiction")
    return dict(witnesses=len(records),seed_only=all(not w['path']['word'] for w in records.values()),contradiction=str(rhs))

def seed_audit(owner,chart,seeds):
    count=0;bad=0;max_error=Q(0);first=None
    for z in seeds:
        h=row(owner.values(owner.seed(z)));x,r=chart.split(h)
        target,residual=chart.target(h)
        assert chart.reverse(target,residual)==h
        err=x-row(z);norm=max(abs(Q(v)) for v in err)
        if norm:
            bad+=1
            if first is None:first=dict(seed=z,error=pack(vec(err)))
        max_error=max(max_error,norm);count+=1
    return dict(rows=count,nonzero_native_coordinate_errors=bad,max_exact_coordinate_error=str(max_error),first=first)

def path_audit(owner,chart,z,word):
    h=owner.seed(z);x,r=chart.split(row(owner.values(h)));target=row(z)
    seed_defect=x-target;acc=S.zeros(1,6);trace=[]
    for u in word:
        obs,n=owner.step(h,u)
        M=chart.actions[u];y,s=chart.split(row(owner.values(n)))
        delta=y-x*M
        acc=acc*M+delta;seed_defect=seed_defect*M;target=target*M
        assert y-target==seed_defect+acc
        assert chart.reverse(y*chart.A,s)==row(owner.values(n))
        # Full operator commutator relative to identity-complement extension.
        # Not a native execution on a rounded transformed input.
        expected=chart.act(row(owner.values(h)),u)
        defect=row(owner.values(n))-expected
        assert defect==delta*chart.B+(s-r)
        trace.append(dict(control=u,seed_defect=pack(vec(seed_defect)),recurrent_defect=pack(vec(acc)),edge_defect=pack(vec(delta)),kernel_difference=pack(vec(s-r))))
        h,x,r=n,y,s
    return dict(seed=z,word=word,trace=trace)

def marked_future(owner,chart,seeds,budget,seconds):
    # Target labels are part of the extended state, not a historical input.
    # This permits a constructive target register even if raw-state semantic
    # descent fails. Retain raw observations and cap profile in the quotient.
    fine=tuple((owner.seed(z),tuple(z)) for z in seeds)
    def code(p):
        h,z=p
        x,r=chart.split(row(owner.values(h)))
        return tuple(z),vec(x),cap_profile(owner.history(h))
    def step(p,u):
        h,z=p
        obs,n=owner.step(h,u)
        target=owner.action(u,z)
        return obs,(n,target)
    result=future_acceptance(fine,CONTROLS,step,code,budget=budget,seconds=seconds,export_realization=True)
    if 'realization' in result:
        result['independent_replay']=verify_realization(result['realization'],fine,CONTROLS,code,step)
    result['scope']='extended historical bytes plus independent target register; no learned semantic descent asserted'
    return result

def main():
    import torch
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','phase-manifest','phase-receipt','prior-receipt','receipt'):
        p.add_argument('--'+name,type=Path,required=name in ('checkpoint','prior-receipt','receipt'))
    for name in ('sha256','source-head','phase-manifest-sha256','prior-receipt-sha256'):
        p.add_argument('--'+name,required=name!='phase-manifest-sha256')
    p.add_argument('--budget',type=int,default=1000)
    p.add_argument('--seconds',type=float,default=90)
    a=p.parse_args()
    if a.receipt.exists() or len({bool(a.phase_manifest),bool(a.phase_manifest_sha256),bool(a.phase_receipt)})!=1:p.error("invalid outputs or phase inputs")
    raw=a.prior_receipt.read_bytes()
    if sha256(raw).hexdigest()!=a.prior_receipt_sha256:raise ValueError("prior receipt digest")
    prior=json.loads(raw)
    if prior['historical_sha256']!=a.sha256:raise ValueError("different donor")
    witness=replay_conflict(prior)
    root=Path(__file__).resolve().parents[1];source=source_identity(root,a.source_head)
    if not torch.cuda.is_available() or not torch.version.hip or not torch.cuda.get_device_properties(0).gcnArchName.startswith('gfx803'):raise RuntimeError("native gfx803 required")
    old,m,_,probe=restore(a,root,torch.device('cuda',0));before=model_digest(old)
    owner=Historical(old);d=owner.seed_dictionary()
    for name in ('historical_basis','target_basis','historical_section'):
        if d[name]!=prior['seed_dictionary'][name]:raise ValueError("seed dictionary mismatch")
    chart=SeedAction(matrix(d['historical_basis']),matrix(d['target_basis']))
    seeds=tuple(m0+d0 for m0,d0 in product(MEANS if m['joint_mean'] else ((1,0,0),),TRIADS))
    result=dict(kind='x369_transported_seed_action',source=source,prior_sha256=a.prior_receipt_sha256,historical_sha256=a.sha256,conflict=witness,algebra=chart.certificate(),seeds=seed_audit(owner,chart,seeds),paths=[path_audit(owner,chart,z,w) for z in (seeds[0],seeds[-1]) for w in (((8,0),(10,2),(1,3)),((10,2),(8,0),(10,2)),((8,0),)*3)],future=marked_future(owner,chart,seeds,a.budget,a.seconds),training_steps=0,universal_recurrence_equivariance=False)
    if before!=model_digest(old) or source!=source_identity(root,a.source_head):raise RuntimeError("source mutation")
    exclusive(a.receipt,(json.dumps(result,indent=2)+'\n').encode())

if __name__=='__main__':main()
