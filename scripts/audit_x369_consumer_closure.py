"""Factorial chart/residual interventions. Original Leech-LILA: Anatolii Kornienko.
DASHI extension; finite diagnostic, not a universal behavioural certificate.
"""
import torch


def factorial_states(z, target, V, q, q_target, A):
    """Rows; V has orthonormal columns. Same-prefix natural baseline.
    00=fitted chart + old residual; 10=oracle chart + old residual;
    01=fitted chart + natural residual; 11=natural complete history.
    Oracle components are diagnostic, not deployable compression.
    """
    fitted=(q@A-q)@V.T
    chart=(q_target-q)@V.T
    rest=target-z-chart
    return z+fitted,z+chart,z+fitted+rest,target


def factorial_effect(outputs):
    a,b,c,d=outputs
    return b-a,c-a,d-b-c+a


def audit_consumer_closure(model,rows,device,dtype,max_pairs=128,horizon=2):
    """Frozen natural generator chart, matched public continuations.
    Every appended literal reads the SAME original guard assignment.
    Deterministically bounded sample per generator, not a lockbox or proof.
    """
    from itertools import product
    from scripts.audit_x369_hidden_generators import capture,fit_natural_generators
    from scripts.train_x369_ordered_action import encode_inputs
    if model.training:raise ValueError('evaluation mode required')
    if max_pairs<1 or not 1<=horizon<=3:raise ValueError('invalid audit bounds')
    Z=capture(model,rows,device,dtype);f=fit_natural_generators(rows,Z)
    if not f['accepted']:return dict(chart_admitted=False,behavioural_closure_established=False)
    V,q,A,targets=(f[k] for k in ('V','q','A','targets'));Z=Z.double()
    shape=(-1,model.experts,2,model.width) if model.paired else (-1,model.experts,model.width)
    report={}
    with torch.no_grad():
        for name,ix in targets.items():
            eligible=[i for i,r in enumerate(rows) if r['split']!='train' and int(ix[i])!=i]
            chosen=eligible if len(eligible)<=max_pairs else [eligible[j*len(eligible)//max_pairs] for j in range(max_pairs)]
            if not chosen:continue
            j=ix[chosen];states=factorial_states(Z[chosen],Z[j],V,q[chosen],q[j],A[name])
            entry={}
            for depth in range(1,horizon+1):
                errors=[0.]*4;hist=[0.]*4;routes=[0]*4;signs=[0]*4;weights=[0.]*4;mix=0.;count=0
                for letters in product('abjn',repeat=depth):
                    ops=torch.tensor([encode_inputs(dict(rows[i],word=''.join(letters)),'signed')[1] for i in chosen],device=device,dtype=dtype)
                    results=[]
                    for state in states:
                        h=state.to(device,dtype).reshape(shape)
                        for t in range(depth):y,h,ids,ss,ww=model.advance(ops[:,t],h)
                        p=torch.cat([p.softmax(-1) for p in model.consume(y,h)],-1)
                        results.append((p,h,ids,ss,ww))
                    p0,h0,i0,s0,w0=results[-1]
                    for k,(p,h,i,s,w) in enumerate(results):
                        errors[k]+=float((p-p0).abs().sum());hist[k]+=float((h-h0).flatten(1).norm(dim=-1).sum());routes[k]+=int((i!=i0).any(-1).sum());signs[k]+=int((s!=s0).flatten(1).any(-1).sum());weights[k]+=float((w-w0).abs().sum())
                    mix+=float(factorial_effect([a[0] for a in results])[2].abs().sum());count+=len(chosen)
                entry[str(depth)]=dict(comparisons=count,probability_l1=[v/count for v in errors],history_l2=[v/count for v in hist],route_mismatch=routes,sign_mismatch=signs,weight_l1=[v/count for v in weights],mixed_probability_effect_l1=mix/count)
            report[name]=dict(eligible=len(eligible),sampled=len(chosen),by_depth=entry)
    return dict(chart_admitted=True,arms=['fitted_old','oracle_chart_old','fitted_natural_residual','natural_full'],generators=report,behavioural_closure_established=False)
