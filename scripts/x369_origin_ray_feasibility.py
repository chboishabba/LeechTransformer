"""Exact rational QF_LRA necessary kernel feasibility for the new source.
Original Leech-LILA: Anatolii Kornienko; DASHI extension.
No fitted checkpoint migration. UNSAT of all three slices excludes a real
full-domain stable linear kernel under the declared equivariant observer.
SAT is only a survivor of an outer approximation.
"""
from fractions import Fraction as Q
from itertools import permutations
from models.geometric_reflection_fibre import GROUP,act
from scripts.x369_exact_origin_ray import vec,mat,pair_g,flatten,conjugate
from scripts.x369_r4_interval_certificate import fused_activation,point,exp_point


def num(x):
    x=Q(x)
    s=str(abs(x.numerator)) if x.denominator==1 else f"(/ {abs(x.numerator)} {x.denominator})"
    return f"(- {s})" if x<0 else s


def op(name,*args):
    if not args:return {"and":"true","or":"false","+":"0"}[name]
    if len(args)==1 and name in ("and","or","+"):return args[0]
    return f"({name} {' '.join(args)})"


def add(*x):return op("+",*x)
def neg(x):return op("-",x)
def scale(a,x):return op("*",num(a),x)
def eq(a,b):return op("=",a,b)
def ite(c,a,b):return op("ite",c,a,b)
def both(*x):return op("and",*x)
def either(*x):return op("or",*x)
def lex_ge(a,b):
    # Recursively preserve earlier ties; no tolerance.
    result="true"
    for x,y in reversed(tuple(zip(a,b))):
        result=either(op(">",x,y),both(eq(x,y),result))
    return result


def symbolic_action(g,v):
    # act() multiplies by integer signs; use coordinate indices instead.
    k,b,a=g;out=[]
    for i in range(0,len(v),3):
        z=v[i:i+3];z=z[::-1] if b else z;z=z[k:]+z[:k]
        out.extend(neg(x) if a else x for x in z)
    return tuple(out)


def pa(g,p):
    if g[1]:p=p[::-1]
    return tuple(symbolic_action(g,b) for b in p)


class SMT:
    def __init__(self):self.lines=["(set-logic QF_LRA)"];self.i=0
    def var(self):
        self.i+=1;n=f"v{self.i}";self.lines.append(f"(declare-fun {n} () Real)");return n
    def bind(self,s,typ="Real"):
        self.i+=1;n=f"v{self.i}";self.lines.append(f"(define-fun {n} () {typ} {s})");return n
    def assert_(self,s):self.lines.append(f"(assert {s})")
    def dot(self,a,b):return self.bind(add(*(scale(x,y) for x,y in zip(a,b) if x)))
    def mv(self,A,v):return tuple(self.dot(r,v) for r in A)
    def finish(self):return "\n".join(self.lines+["(check-sat)",""])


def slice_history(s,E,d,block):
    params=[];h=[]
    for _ in range(E):
        p=[[],[]]
        for _ in range(d//3):
            a=s.var();params.append(a)
            if block=="mean_dc":M=(a,a,a);D=("0",)*3
            elif block=="residual_dc":M=("0",)*3;D=(a,a,a)
            elif block=="standard":
                b=s.var();params.append(b);M=(a,scale(-2,a),a);D=(b,"0",neg(b))
            else:raise ValueError("unknown slice")
            p[0].extend(add(m,v) for m,v in zip(M,D))
            p[1].extend(add(m,neg(v)) for m,v in zip(M,D))
        h.append(tuple(map(tuple,p)))
    # Positive homogeneous constraints: normalize largest absolute parameter to 1.
    for a in params:s.assert_(both(op(">=",a,"(- 1)"),op("<=",a,"1")))
    s.assert_(either(*(either(eq(a,"1"),eq(a,"(- 1)")) for a in params)))
    return tuple(h),len(params)


def project(s,p,K):
    # UNNORMALIZED sum; common positive |K| omitted everywhere.
    rows=[pa(g,p) for g in GROUP]
    return tuple(tuple(s.bind(add(*(ite(k,r[b][i],"0") for k,r in zip(K,rows))))
                       for i in range(len(p[b]))) for b in range(2))


def branch(s,x,h,w,b,A,B,T,capped):
    d=len(x);a=s.dot(w[d:],h);c=sum((i*j for i,j in zip(w[:d],x)),Q(0))
    # Fallback on identically-zero score is overapproximated by any sign,
    # including zero. Safe for exclusion, not complete for ray feasibility.
    r=s.var();guard=both(op("=>",op(">",a,"0"),eq(r,"1")),
        op("=>",op("<",a,"0"),eq(r,"(- 1)")),
        op("=>",eq(a,"0"),eq(r,num(1 if c>0 else -1))) if c else "true",
        either(eq(r,"1"),eq(r,"(- 1)"),eq(r,"0")))
    # Sign times ReLU(sign * Hh) equals positive/negative part of Hh.
    z=s.mv(tuple(row[d:] for row in A),h)
    z=tuple(s.bind(ite(eq(r,"1"),ite(op(">",v,"0"),v,"0"),
        ite(eq(r,"(- 1)"),ite(op("<",v,"0"),v,"0"),"0"))) for v in z)
    y=s.mv(B,z)
    t=("0",)*d if capped else tuple(s.bind(ite(eq(r,"0"),"0",v))
                for v in s.mv(T,h+("0",)*d+y))
    return y,t,guard


def enclose_zero(s,groups):
    # Each group is (certified rational exp interval, summed slope).
    # Linear outer enclosure valid for arbitrary REAL slope coordinates.
    lo=[];hi=[]
    for I,v in groups:
        positive=op(">=",v,"0")
        lo.append(ite(positive,scale(I.lo,v),scale(I.hi,v)))
        hi.append(ite(positive,scale(I.hi,v),scale(I.lo,v)))
    return both(op("<=",add(*lo),"0"),op(">=",add(*hi),"0"))


def build(model,block,controls=None):
    import torch
    from scripts.x369_source_geometry import SourceGeometryOrderedLearner
    from models.history_orbit_moe import HistoryOrbitMoE
    if type(model) is not SourceGeometryOrderedLearner or model.training:raise ValueError("new eval source required")
    src=model.history.source;d=model.width;E=model.experts
    if (type(src) is not HistoryOrbitMoE or src.route_geometry!="c2_c3_residual"
        or src.transition_rank is not None or getattr(src,"route_probe_action",None) is not None
        or any(not torch.isfinite(p).all() for p in model.parameters())
        or (src.history_cap is not None and not 0<float(src.history_cap)<float("inf"))):
        raise ValueError("unsupported source")
    controls=tuple((u,t) for u in range(len(GROUP)) for t in range(4)) if controls is None else tuple(controls)
    if not controls or any(not 0<=u<len(GROUP) or not 0<=t<4 for u,t in controls):raise ValueError("invalid controls")
    s=SMT();h,n=slice_history(s,E,d,block)
    bias=vec(src.expert_bias);base=vec(src.router.weight)[d:];w=vec(src.c3_residual_weight)[d:];orient=vec(src.orientation.weight)
    W=tuple(r[2*d:] for r in mat(model.readout.weight))
    tables=[getattr(model,f"label_perm_{i}").detach().cpu().tolist() for i in range(3)]
    experts=[(mat(e[0].weight),mat(e[2].weight),mat(t.weight)) for e,t in zip(src.experts,src.transitions)]
    orders=[p for p in permutations(range(E),src.top_k)
        if all(bias[i]>=bias[j] for i,j in zip(p,p[1:]))
        and all(bias[i]>=bias[j] for i in p for j in set(range(E))-set(p))]
    # Conjugate controls in same class and tag are redundant on invariant kernel,
    # but retaining all supplied controls is sound.
    for u,tag in controls:
        cu=max(conjugate(g,u) for g in GROUP);marks=[g for g in GROUP if conjugate(g,u)==cu]
        x=vec(model.control_embed.weight[tag*len(GROUP)+cu])
        z=tuple(v.lo for v in fused_activation(list(map(point,act((1,0,0),x))),list(map(point,w))))
        choices=[]
        for g in marks:
            c=tuple(pa(g,p) for p in h);flat=flatten(c)
            can=both(*(lex_ge(flat,flatten(tuple(pa(a,p) for p in h))) for a in marks))
            K=[s.bind(both(*(eq(a,b) for a,b in zip(flat,flatten(tuple(pa(k,p) for p in c)))))
                 if conjugate(k,cu)==cu else "false","Bool") for k in GROUP]
            slopes=[];Y=[];T=[];guards=[]
            for p,(A,B,C) in zip(c,experts):
                m=tuple(s.bind(scale(Q(1,2),add(a,b))) for a,b in zip(*p))
                ab=tuple(s.bind(ite(op(">=",v,"0"),v,neg(v))) for v in m)
                slopes.append(s.bind(add(s.dot(base,ab),s.dot(z,m))))
                y,t,guard=zip(*(branch(s,x,bh,orient,b,A,B,C,src.history_cap is not None) for b,bh in enumerate(p)))
                Y.append(flatten((project(s,y,K),)));T.append(project(s,t,K));guards.extend(guard)
            inactive=[project(s,p,K) for p in c]
            local=[]
            for order in orders:
                lead=slopes[order[0]];outside=set(range(E))-set(order)
                route=both(*(eq(slopes[i],lead) for i in order),*(op("<=",slopes[j],lead) for j in outside))
                groups={}
                for e in order:groups.setdefault(bias[e],[]).append(e)
                intervals={b:point(1) if b==bias[order[0]] else exp_point(b-bias[order[0]]) for b in groups}
                output=both(*(enclose_zero(s,[(intervals[b],add(*(Y[e][i] for e in es))) for b,es in groups.items()]) for i in range(2*d)))
                nxt=tuple(T[e] if e in order else inactive[e] for e in range(E))
                v=s.mv(W,flatten(nxt));off=0;label=[]
                for table in tables:
                    width=len(table[0])
                    for i in range(width):
                        label.append(eq(add(*(ite(k,v[off+row[i]],"0") for k,row in zip(K,table))),"0"))
                    off+=width
                local.append(both(route,output,*label))
            choices.append(both(can,*guards,either(*local)))
        s.assert_(either(*choices))
    return s.finish(),n


def solve_smt(program,timeout_ms=90000):
    """Use system libz3 directly (no Python wheel required). UNKNOWN is open."""
    import ctypes as C
    from ctypes.util import find_library
    lib=C.CDLL(find_library("z3") or "libz3.so.4")
    ptr=C.c_void_p
    for name,rest,args in [
        ("Z3_mk_config",ptr,[]),("Z3_set_param_value",None,[ptr,C.c_char_p,C.c_char_p]),
        ("Z3_mk_context",ptr,[ptr]),("Z3_del_config",None,[ptr]),
        ("Z3_eval_smtlib2_string",C.c_char_p,[ptr,C.c_char_p]),("Z3_del_context",None,[ptr])]:
        f=getattr(lib,name);f.restype=rest;f.argtypes=args
    cfg=lib.Z3_mk_config();lib.Z3_set_param_value(cfg,b"timeout",str(timeout_ms).encode())
    ctx=lib.Z3_mk_context(cfg);lib.Z3_del_config(cfg)
    try:result=lib.Z3_eval_smtlib2_string(ctx,program.encode()).decode().strip()
    finally:lib.Z3_del_context(ctx)
    if result not in ("sat","unsat","unknown"):raise RuntimeError(result)
    return result


def audit(model,controls=None,timeout_ms=90000,receipt=None):
    """Monotone necessary test; record each block immediately, no training.
    A subset of controls is sufficient for an exclusion; not for admission.
    receipt is an optional callback accepting the cumulative receipt.
    """
    from hashlib import sha256
    if timeout_ms<1 or timeout_ms>90000:raise ValueError("per-block budget 1..90000 ms")
    # Start with just the identity control and one tag. More controls can be
    # supplied explicitly when this cheap cut survives. The invariant-kernel
    # proof itself still assumes the full conjugation-closed legal alphabet.
    controls=tuple(controls) if controls is not None else ((0,0),)
    result=dict(controls=controls,blocks={},real_full_domain_kernel_excluded=False,
        necessary_outer_approximation=True,native_float_certified=False,universal_weld=False)
    for block in ("mean_dc","residual_dc","standard"):
        program,n=build(model,block,controls)
        status=solve_smt(program,timeout_ms)
        result["blocks"][block]=dict(status=status,variables=n,smt_sha256=sha256(program.encode()).hexdigest())
        if receipt is not None:receipt(result)
    result["real_full_domain_kernel_excluded"]=all(v["status"]=="unsat" for v in result["blocks"].values())
    if receipt is not None:receipt(result)
    return result
