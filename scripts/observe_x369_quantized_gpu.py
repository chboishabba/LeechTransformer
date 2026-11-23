"""Observe native source TopK and quantized two-channel score on GPU.
Retrospective exact dyadic comparison, NOT a prospective HIP rounding bound.
No changes to original R4 source/fingerprint. No F5 or fitted assets.
"""
import argparse,hashlib,json
from fractions import Fraction as Q
import torch
from models.history_orbit_moe import HistoryOrbitMoE
from models.trialectic_wave_prior import rotate3
from scripts.observe_x369_r4_original_topk import source_selected_with_logits
from scripts.x369_quantized_consumer import encode,quantize

def fraction(v):
    return Q(float(v))

def hex_sha(t):
    a=t.detach().contiguous().cpu().numpy()
    return hashlib.sha256(a.tobytes()).hexdigest()

@torch.no_grad()
def receipt(module,x,h,quantum,omit=False):
    """Exact model-evaluation audit, not a claim of universally bounded kernel errors."""
    if module.route_geometry!="c2_c3_residual" or module.router_selection!="topk" or getattr(module,"route_probe_action",None) is not None:
        raise ValueError("requires original coherent topk R4 without interventions")
    if x.ndim!=2 or h.shape!=(x.shape[0],module.num_experts,module.d_model):
        raise ValueError("shape mismatch")
    if not (torch.isfinite(x).all() and torch.isfinite(h).all()):
        raise ValueError("nonfinite source operands")
    ids,weights,logits=source_selected_with_logits(module,x,h)
    phase=rotate3(x)
    tw=module.c3_residual_weight[:,module.d_model:]
    # Same original C2/unary contribution; the observer records the original topk operand.
    from models.trialectic_wave_prior import trialectic_router_invariants
    import torch.nn.functional as F
    features=torch.cat((x[:,None,:].expand(-1,module.num_experts,-1),h),dim=-1)
    unary=F.linear(trialectic_router_invariants(phase),module.c3_residual_weight[:,:module.d_model])
    baseline=module.router(features.abs()).squeeze(-1)+module.expert_bias+unary
    xs=phase.detach().cpu().tolist()
    hs=h.detach().cpu().tolist()
    ws=tw.detach().cpu().tolist()[0]
    raw=logits.detach().cpu().tolist()
    base=baseline.detach().cpu().tolist()
    records=[]
    for row,(xx,experts) in enumerate(zip(xs,hs)):
        scores=[]; pair_errors=[]
        for expert,hh in enumerate(experts):
            triads=[]; heads=[]
            for t in range(0,len(xx),3):
                a,b,c=map(fraction,xx[t:t+3]); A,B,C=map(fraction,hh[t:t+3])
                triads.append((a*A+b*B+c*C,a*B+b*C+c*A,a*C+b*A+c*B))
                heads.append(tuple(map(fraction,ws[t:t+3])))
            candidate,err=encode(triads,heads,Q(quantum),omit)
            full=candidate+err
            native=fraction(raw[row][expert])
            # baseline is itself a rounded CUDA evaluation, treated as a
            # stored dyadic value, not the exact ideal C2/unary expression.
            native_directed=native-fraction(base[row][expert])
            residual=native_directed-full
            scores.append(dict(expert=expert,candidate=str(candidate),quantization_error=str(err),
                native_directed_residual=str(residual),exact_dyadic_directed=str(full),
                observed_logit=str(native)))
        order=ids[row].tolist()
        candidate=[Q(v["candidate"])+fraction(base[row][e]) for e,v in enumerate(scores)]
        candidate_order=sorted(range(len(candidate)),key=lambda e:(-candidate[e],e))
        min_pair=None
        for i in order:
            for j in range(module.num_experts):
                if j not in order:
                    gap=candidate[i]-candidate[j]
                    min_pair=gap if min_pair is None else min(min_pair,gap)
        records.append(dict(sample=row,ids=order,quantized_selected_ids=candidate_order[:module.top_k],
            strict_candidate_membership_margin=None if min_pair is None else str(min_pair),
            native_selected_weight=[float(t) for t in weights[row].tolist()],
            expert_scores=scores))
    return dict(kind="x369_native_two_channel_quantization_observation",device=str(x.device),
        dtype=str(x.dtype),quantum=str(Q(quantum)),omit_dc=omit,source_x_sha256=hex_sha(x),
        source_h_sha256=hex_sha(h),model_state_owner="synthetic test state; checkpoint binding not claimed",
        records=records,original_fingerprint_unchanged=True,retrospective_exact_dyadic=True,
        prospective_native_rounding_certified=False,nat_cell_certified=False,
        future_history_sufficient=False,source_code_attribution="LeechTransformer / Anatolii Kornienko")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--require-gfx803",action="store_true")
    p.add_argument("--dtype",choices=("float32","float64"),default="float64")
    p.add_argument("--batch",type=int,default=3)
    p.add_argument("--width",type=int,default=384)
    p.add_argument("--experts",type=int,default=4)
    p.add_argument("--quantum",default="1/256")
    p.add_argument("--omit-dc",action="store_true")
    a=p.parse_args()
    if min(a.batch,a.width,a.experts)<1 or a.width%3 or a.experts<2:
        p.error("bad shape")
    if Q(a.quantum)<=0:p.error("quantum must be positive")
    if not torch.cuda.is_available():raise RuntimeError("GPU required, no CPU fallback")
    dev=torch.device("cuda",torch.cuda.current_device()); prop=torch.cuda.get_device_properties(dev)
    arch=getattr(prop,"gcnArchName","")
    if a.require_gfx803 and (not torch.version.hip or not arch.startswith("gfx803")):
        raise RuntimeError("gfx803 HIP required")
    torch.manual_seed(41)
    m=HistoryOrbitMoE(d_model=a.width,num_experts=a.experts,top_k=2,route_geometry="c2_c3_residual").to(dev,getattr(torch,a.dtype)).eval()
    with torch.no_grad():m.c3_residual_weight.normal_(std=.003)
    x=torch.randn(a.batch,a.width,device=dev,dtype=getattr(torch,a.dtype))
    h=torch.randn(a.batch,a.experts,a.width,device=dev,dtype=getattr(torch,a.dtype))
    out=receipt(m,x,h,a.quantum,a.omit_dc)
    out.update(hardware=dict(name=prop.name,arch=arch,hip=torch.version.hip,torch=torch.__version__))
    print(json.dumps(out,sort_keys=True,allow_nan=False),flush=True)
if __name__=="__main__":main()
