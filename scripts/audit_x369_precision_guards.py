"""Actual source guard measurements and FP32/FP64 path comparison.
Empirical same-device diagnostic, NOT a calibrated real-arithmetic certificate.
Original Leech-LILA: Anatolii Kornienko; DASHI extension.
"""
import copy
from collections import defaultdict
import torch


def ordered_gap(logits,ids):
    """Minimum strict comparison needed to retain ordered TopK."""
    v=logits.gather(1,ids);parts=[]
    if ids.shape[1]>1:parts.append((v[:,:-1]-v[:,1:]).amin(1))
    if ids.shape[1]<logits.shape[1]:
        outside=torch.ones_like(logits,dtype=torch.bool).scatter_(1,ids,False)
        parts.append(v[:,-1]-logits.masked_fill(~outside,-torch.inf).amax(1))
    return torch.stack(parts).amin(0) if parts else torch.full_like(v[:,0],torch.inf)


def measured_order_bound(a,b,ids_b):
    gap=ordered_gap(b,ids_b);eps=(a-b).abs().amax(1)
    return gap,eps,gap>2*eps


def observe_step(model,op,h):
    from scripts.observe_x369_r4_original_topk import source_selected_with_logits
    source=model.history.source if model.paired else model.history
    if source.route_geometry!='c2_c3_residual' or getattr(source,'route_probe_action',None) is not None:
        raise ValueError('unintervened coherent source required')
    x=model.embed(op);route_h=(h[:,:,0]+h[:,:,1])/2 if model.paired else h
    ids,weights,logits=source_selected_with_logits(source,x,route_h)
    scores=x.new_empty((*ids.shape,2) if model.paired else ids.shape)
    # Match original per-expert selected row shapes and branch-local dot products.
    for e in range(source.num_experts):
        pos=(ids==e).nonzero(as_tuple=False)
        if not len(pos):continue
        rows,slots=pos.unbind(1);raw=x.index_select(0,rows)
        for b in range(2 if model.paired else 1):
            old=h[rows,e,b] if model.paired else h[rows,e]
            z=torch.cat((raw,old),-1);s=(2*torch.nn.functional.linear(z,source.orientation.weight,None)).squeeze(-1)
            if model.paired:scores[rows,slots,b]=s
            else:scores[rows,slots]=s
    if 'cap_history' in source.__dict__:raise ValueError('already shadowed cap method')
    cap=source.history_cap;original=source.cap_history;norms=[]
    def capture_cap(v):
        if cap is not None:norms.append(torch.linalg.vector_norm(v.double(),dim=-1).detach())
        return original(v)
    source.cap_history=capture_cap
    try:y,h1,i,s,w=model.advance(op,h)
    finally:del source.cap_history
    if not torch.equal(ids,i) or not torch.equal(weights,w):raise RuntimeError('source route observation differs from actual step')
    if not torch.equal(s[scores!=0],scores[scores!=0].sign()):raise RuntimeError('orientation score replay mismatch')
    p=torch.cat([v.softmax(-1) for v in model.consume(y,h1)],-1)
    if not all(torch.isfinite(v).all() for v in (y,h1,logits,scores,w,p)):raise RuntimeError('nonfinite actual trace')
    cap_info=None
    if cap is not None:
        n=torch.cat(norms);c=float(cap);cap_info=dict(calls=len(n),outside=int((n>c).sum()),boundary=int((n==c).sum()),minimum_margin=float((n-c).abs().min()))
    return dict(y=y,h=h1,ids=i,signs=s,weights=w,p=p,logits=logits,scores=scores,cap=cap_info)


def audit_precision_guards(model,rows,device,max_rows=256):
    """Both paths advance their OWN histories; compare every visited step.
    No FP64-as-exact or universal perturbation guarantee."""
    from scripts.train_x369_ordered_action import batch
    if model.training or max_rows<1:raise ValueError('eval and positive sample budget required')
    eligible=[r for r in rows if r['split']=='composition']
    chosen=eligible if len(eligible)<=max_rows else [eligible[i*len(eligible)//max_rows] for i in range(max_rows)]
    groups=defaultdict(list)
    for r in chosen:groups[len(r['word'])].append(r)
    ref=copy.deepcopy(model).double().eval();stats=defaultdict(lambda:dict(rows=0,route_mismatch=0,sign_mismatch_same_route=0,observed_order_bound=0,observed_sign_bound=0,zero_score32=0,zero_score64=0,topk_ties32=0,topk_ties64=0,max_logit_delta=0.,max_history_delta=0.,max_output_delta=0.,max_probability_l1=0.,min_gap64=None,min_abs_score64=None,cap32=[],cap64=[]))
    with torch.no_grad():
        for length,rs in groups.items():
            s,o,_=batch(rs,'signed',device,torch.float32);h=model.initial(s);h64=ref.initial(s.double())
            for t in range(length):
                a=observe_step(model,o[:,t],h);b=observe_step(ref,o[:,t].double(),h64);h=a['h'];h64=b['h']
                v=stats[t+1];v['rows']+=len(rs);same=(a['ids']==b['ids']).all(1)
                v['route_mismatch']+=int((~same).sum());v['sign_mismatch_same_route']+=int(((a['signs']!=b['signs']).flatten(1).any(1)&same).sum())
                gap,eps,paid=measured_order_bound(a['logits'].double(),b['logits'],b['ids'])
                if (paid&~same).any():raise RuntimeError('observed strict TopK comparison violated')
                v['observed_order_bound']+=int(paid.sum());v['topk_ties32']+=int((ordered_gap(a['logits'],a['ids'])==0).sum());v['topk_ties64']+=int((gap==0).sum())
                sa=a['scores'].double().flatten(1);sb=b['scores'].flatten(1);spaid=same&(sb.abs()>(sa-sb).abs()).all(1)
                if (spaid&(a['signs']!=b['signs']).flatten(1).any(1)).any():raise RuntimeError('observed strict orientation comparison violated')
                v['observed_sign_bound']+=int(spaid.sum());v['zero_score32']+=int((sa==0).sum());v['zero_score64']+=int((sb==0).sum())
                for name,val in [('max_logit_delta',eps.max()),('max_history_delta',(h.double()-h64).abs().max()),('max_output_delta',(a['y'].double()-b['y']).abs().max()),('max_probability_l1',(a['p'].double()-b['p']).abs().sum(1).max())]:v[name]=max(v[name],float(val))
                for name,val in [('min_gap64',gap.min()),('min_abs_score64',sb.abs().min())]:
                    z=float(val);v[name]=z if v[name] is None else min(v[name],z)
                if a['cap'] is not None:v['cap32'].append(a['cap']);v['cap64'].append(b['cap'])
    return dict(sampled_rows=len(chosen),by_time=dict(stats),fp64_is_exact_reference=False,rounding_intervals_certified=False,universal_guard_transport_certified=False)
