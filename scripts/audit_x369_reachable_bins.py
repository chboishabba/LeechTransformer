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
    """Exact code-pair comparison, vectorized over pending pairs.
    Repeat each separating observation batch before retaining a witness.
    """
    if not 1<=horizon<=3 or not codes or not labels:raise ValueError('invalid scope')
    groups={}
    for i,c in enumerate(codes):groups.setdefault(c,[]).append(i)
    pairs=torch.tensor([p for g in groups.values() for p in combinations(g,2)],dtype=torch.long).reshape(-1,2)
    pending=torch.ones(len(pairs),dtype=torch.bool);witnesses=[];depth_counts={};checked=0;replayed=0
    for depth in range(1,horizon+1):
        depth_counts[str(depth)]=0
        for word in product(labels,repeat=depth):
            obs=observe(word);checked+=1
            if not obs or any(not isinstance(v,torch.Tensor) or len(v)!=len(codes) or not torch.isfinite(v).all() for v in obs.values()):raise ValueError('finite aligned observations required')
            ix=pending.nonzero().flatten()
            if not len(ix):continue
            i,j=pairs[ix].unbind(1);obs={k:v.detach().cpu() for k,v in obs.items()}
            flags={k:(v[i]!=v[j]).reshape(len(ix),-1).any(1) for k,v in obs.items()}
            hits=torch.stack(list(flags.values())).any(0).nonzero().flatten()
            if not len(hits):continue
            again=observe(word);replayed+=1
            if set(again)!=set(obs) or any(not torch.equal(v,again[k].detach().cpu()) for k,v in obs.items()):raise RuntimeError('separating word not repeatable')
            for n in hits.tolist():
                a,b=int(i[n]),int(j[n]);fields=[k for k,v in flags.items() if bool(v[n])]
                witnesses.append(dict(i=a,j=b,word=list(word),fields=fields,max_abs_delta={k:float((obs[k][a].double()-obs[k][b].double()).abs().max()) for k in fields}));depth_counts[str(depth)]+=1
            pending[ix[hits]]=False
    return dict(states=len(codes),code_classes=len(groups),collision_pairs=len(pairs),separated_pairs=len(witnesses),unseparated_pairs=int(pending.sum()),first_separation_depth=depth_counts,words_checked=checked,separating_words_replayed=replayed,witnesses=witnesses,universal_closure_certified=False)


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
