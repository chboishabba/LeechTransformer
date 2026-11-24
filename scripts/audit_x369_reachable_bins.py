"""Bounded exact-code witness search; not an invariant-domain certificate."""
from itertools import product, combinations
import torch


def code_bins(q, quantum):
    if not 0 < quantum < float('inf') or not torch.isfinite(q).all():
        raise ValueError('finite positive quantum and finite coordinates required')
    v=(q.double()/quantum).round()
    if not torch.isfinite(v).all() or (v.abs()>=2**53).any():
        raise ValueError('unsafe bin range')
    return [tuple(map(int,row)) for row in v.tolist()]


def bounded_witnesses(codes, labels, observe, horizon):
    """observe(word) gives named tensors with one row per state. Identical
    actual integer codes are compared, not approximate proximity. Words
    visited breadth-first; all terminal observations at every depth.
    Equality uses torch.equal on same-dtype finite tensors, not real proof.
    """
    if not 1<=horizon<=3 or not codes or not labels:raise ValueError('invalid scope')
    groups={}
    for i,c in enumerate(codes):groups.setdefault(c,[]).append(i)
    pending={p for g in groups.values() for p in combinations(g,2)}
    total=len(pending);witnesses=[];depth_counts={};checked=0
    for depth in range(1,horizon+1):
        depth_counts[str(depth)]=0
        for word in product(labels,repeat=depth):
            obs=observe(word);checked+=1
            if not obs or any(not isinstance(v,torch.Tensor) or len(v)!=len(codes) or not torch.isfinite(v).all() for v in obs.values()):raise ValueError('finite aligned observations required')
            for i,j in sorted(pending):
                different=[k for k,v in obs.items() if not torch.equal(v[i],v[j])]
                if different:
                    witnesses.append(dict(i=i,j=j,word=list(word),fields=different));pending.remove((i,j));depth_counts[str(depth)]+=1
    return dict(states=len(codes),code_classes=len(groups),collision_pairs=total,separated_pairs=len(witnesses),unseparated_pairs=len(pending),first_separation_depth=depth_counts,words_checked=checked,witnesses=witnesses,universal_closure_certified=False)


def audit_reachable_bins(model, rows, device, dtype, quantum=.125, max_contexts=8, horizon=2):
    """Public composition prefixes, natural histories only; context is part
    of code (word, original guards, initial mean, fixed model revision).
    Train-only frozen chart; no heldout choice of bin width or contexts.
    """
    from scripts.audit_x369_hidden_generators import capture,fit_natural_generators,key
    from scripts.train_x369_ordered_action import encode_inputs
    if model.training or max_contexts<1:raise ValueError('eval and positive context count required')
    Z=capture(model,rows,device,dtype);f=fit_natural_generators(rows,Z)
    if not f['accepted']:return dict(chart_admitted=False,universal_closure_certified=False)
    groups={}
    for i,r in enumerate(rows):
        if r['split']=='composition':groups.setdefault(key(r),[]).append(i)
    keys=sorted(groups);chosen=keys if len(keys)<=max_contexts else [keys[j*len(keys)//max_contexts] for j in range(max_contexts)]
    shape=(-1,model.experts,2,model.width) if model.paired else (-1,model.experts,model.width)
    reports=[]
    with torch.no_grad():
        for context in chosen:
            ix=groups[context];codes=code_bins(f['q'][ix],quantum)
            def observe(word):
                ops=torch.tensor([encode_inputs(dict(rows[i],word=''.join(word)),'signed')[1] for i in ix],device=device,dtype=dtype)
                h=Z[ix].to(device,dtype).reshape(shape)
                for t in range(len(word)):y,h,ids,signs,weights=model.advance(ops[:,t],h)
                return dict(raw_output=y.cpu(),ids=ids.cpu(),signs=signs.cpu(),weights=weights.cpu(),probabilities=torch.cat([p.softmax(-1) for p in model.consume(y,h)],-1).cpu())
            # Replay identical observation batch before searching; no claim of global determinism.
            a,b=observe(('a',)),observe(('a',))
            if not all(torch.equal(a[k],b[k]) for k in a):raise RuntimeError('identical replay differs')
            receipt=bounded_witnesses(codes,tuple('abjn'),observe,horizon)
            for w in receipt['witnesses']:
                w['initial_residuals']=[rows[ix[w[k]]]['residual'] for k in ('i','j')]
            reports.append(dict(context=context,**receipt))
    return dict(chart_admitted=True,quantum=quantum,horizon=horizon,contexts_total=len(keys),contexts_sampled=len(chosen),reports=reports,universal_closure_certified=False)
