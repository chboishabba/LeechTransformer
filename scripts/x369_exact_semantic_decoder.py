"""Exact reverse solve of an equivariant linear semantic decoder.
Original Leech-LILA: Anatolii Kornienko; DASHI extension.
Actual recurrent histories are the inputs; independent task endpoints are
constraints only. No supplied action register at decoding time.
"""
from fractions import Fraction as Q
from hashlib import sha256
from scripts.x369_exact_origin_ray import vec
from models.geometric_reflection_fibre import GROUP


def triads(h):
    for p in h:
        if len(p)!=2 or len(p[0])!=len(p[1]) or len(p[0])%3:raise ValueError("paired triads required")
        for i in range(0,len(p[0]),3):
            a,b=p[0][i:i+3],p[1][i:i+3]
            yield tuple((x+y)/2 for x,y in zip(a,b)),tuple((x-y)/2 for x,y in zip(a,b))


def dc(v):return (sum(v,Q(0))/3,)*3
def phase(v):return tuple(x-y for x,y in zip(v,dc(v)))
def twist(v):return tuple(v[(i+1)%3]-v[(i+2)%3] for i in range(3))


def equations(h):
    """Rows in output (M0,M1,M2,D0,D1,D2) order.
    Six coefficients per paired triad: Mdc,Ddc,Mphase,L Dphase,
    Dphase,L Mphase. This is the full real equivariant linear Hom space.
    """
    rows=[[] for _ in range(6)]
    for M,D in triads(h):
        m,d=phase(M),phase(D);a,b=dc(M),dc(D);lM,lD=twist(m),twist(d)
        for i in range(3):
            rows[i].extend((a[i],Q(0),m[i],lD[i],Q(0),Q(0)))
            rows[i+3].extend((Q(0),b[i],Q(0),Q(0),d[i],lM[i]))
    return tuple(map(tuple,rows))


class ExactSystem:
    """Incremental rational RREF, with a replayable sparse row combination."""
    def __init__(self,n):self.n=n;self.basis={};self.conflict=None
    def insert(self,coeff,value,row_id):
        if self.conflict is not None:raise ValueError("system already inconsistent")
        if len(coeff)!=self.n:raise ValueError("wrong dimension")
        a=list(map(Q,coeff))+[Q(value)];w={row_id:Q(1)}
        def sub(k,other):
            for i,v in other.items():
                w[i]=w.get(i,Q(0))-k*v
                if not w[i]:del w[i]
        for i,(r,proof) in sorted(self.basis.items()):
            k=a[i]
            if k:
                a=[v-k*z for v,z in zip(a,r)];sub(k,proof)
        i=next((i for i,v in enumerate(a[:-1]) if v),None)
        if i is None:
            if a[-1]:self.conflict=dict(rows=w,rhs=a[-1])
            return
        k=a[i];a=[v/k for v in a];w={j:v/k for j,v in w.items()}
        # Forward echelon suffices; solve by back substitution.
        self.basis[i]=(a,w)
    def solution(self):
        if self.conflict is not None:raise ValueError("inconsistent")
        x=[Q(0)]*self.n
        for i,(r,_) in sorted(self.basis.items(),reverse=True):
            x[i]=r[-1]-sum((a*b for a,b in zip(r[:-1],x)),Q(0))
        return tuple(x)


def decode(h,coeff):
    rows=equations(h)
    if len(coeff)!=len(rows[0]):raise ValueError("wrong decoder")
    return tuple(sum((a*b for a,b in zip(row,coeff)),Q(0)) for row in rows)


def read_history(t):
    if t.ndim!=3 or t.shape[1]!=2 or t.shape[-1]%3:raise ValueError("paired history required")
    return tuple(tuple(vec(b) for b in p) for p in t)


def fit(records):
    """records: (original witness ID, exact paired history, independent 6-label).
    Every accepted equation is exact. A conflict is an exact linear-decoder
    obstruction on these observations, not a nonlinear or ideal-real no-go.
    """
    sys=None;seen=set();n=0;hashes={};history_labels={}
    for witness,h,y in records:
        if witness in seen or len(y)!=6:raise ValueError("duplicate or malformed witness")
        seen.add(witness);rows=equations(h)
        if sys is None:sys=ExactSystem(len(rows[0]))
        hashes[witness]=sha256(str(h).encode()).hexdigest()
        if h in history_labels:
            prior,old=history_labels[h]
            if tuple(y)!=old:
                return dict(consistent=False,kind="history_collision",states=n+1,
                            witnesses=[prior,witness],labels=[old,tuple(y)],history_sha256=hashes[witness],universal_weld=False)
        else:history_labels[h]=(witness,tuple(y))
        for i,(row,v) in enumerate(zip(rows,y)):
            sys.insert(row,v,(witness,i))
            if sys.conflict:
                c=sys.conflict
                return dict(consistent=False,kind="linear_conflict",rank=len(sys.basis),parameters=sys.n,states=n+1,
                            conflict=[(w,i,str(a),hashes[w]) for (w,i),a in c["rows"].items()],
                            contradiction=str(c["rhs"]),universal_weld=False)
        n+=1
    if sys is None:raise ValueError("empty records")
    return dict(consistent=True,rank=len(sys.basis),parameters=sys.n,states=n,
                coefficients=tuple(map(str,sys.solution())),universal_weld=False)


def source_records(model,rows):
    """Actual NEW source execution, native gfx803 only; no historical migration.
    All chronological prefixes including seed. Original task endpoints are
    used solely as equation targets, not fed to transitions or decoder.
    """
    import torch
    from scripts.x369_source_geometry import SourceGeometryOrderedLearner,signed_seed,literal_controls
    from scripts.train_x369_ordered_action import encode_inputs
    from scripts.audit_x369_ordered_signed_task import endpoint
    if type(model) is not SourceGeometryOrderedLearner or model.training:raise ValueError("new eval architecture required")
    p=next(model.parameters());dev=p.device
    if dev.type!="cuda" or not torch.version.hip or not getattr(torch.cuda.get_device_properties(dev),"gcnArchName","").startswith("gfx803"):
        raise RuntimeError("native gfx803 required")
    # Stream individual rows to avoid silently mixing batch-dependent traces.
    # Explicitly a single-row execution contract, not the old batch receipt.
    with torch.no_grad():
        for i,r in enumerate(rows):
            seed,ops=encode_inputs(r,"signed")
            s=torch.tensor([seed],device=dev,dtype=p.dtype)
            h=signed_seed(s,model.experts,model.width)
            if not torch.isfinite(h).all():raise ValueError("nonfinite source state")
            yield ((i,0),read_history(h[0]),tuple(r["mean"])+tuple(r["residual"]))
            c,t=literal_controls(torch.tensor([ops],device=dev,dtype=p.dtype))
            for j in range(c.shape[1]):
                tag=t[:,j]
                out=model.history.step_controlled(c[:,j],h,lambda z:model.control_embed(tag*len(GROUP)+z),train_sign=False)
                h=out.history
                if not torch.isfinite(h).all():raise ValueError("nonfinite source state")
                m,d,_=endpoint(r["word"][:j+1],r["guards"],r["mean"],r["residual"])
                yield ((i,j+1),read_history(h[0]),tuple(m)+tuple(d))


def verify_decoder(records,coeff):
    """Independent replay; no refit. A match is scoped to supplied histories."""
    coeff=tuple(map(Q,coeff));n=0
    for witness,h,y in records:
        got=decode(h,coeff);n+=1
        if got!=tuple(y):return dict(exact=False,states=n,witness=witness,
            expected=tuple(map(str,y)),decoded=tuple(map(str,got)),universal_weld=False)
    return dict(exact=True,states=n,universal_weld=False)
