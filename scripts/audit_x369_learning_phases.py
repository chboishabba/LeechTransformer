"""Checkpoint-indexed phase diagnostics; original Leech-LILA: Anatolii Kornienko.
User six-phase hypothesis; DASHI extension. No phase classification or release claim.
"""
from itertools import product, permutations
import math
import torch


from scripts.x369_phase_geometry import rms,triad_phase,mod3_spectrum,common_hom,cube_metrics,orbit_spectrum,anchored_phase,cross_time


def phase_audit(model,rows,device,dtype,contexts=4,pairs=32,horizon=2):
    """Fixed full-context sample, training-only chart, composition-only intervention.
    Actual natural histories, not artificial target labels fed into the model.
    """
    from scripts.audit_x369_joint_action import select_contexts,fit_joint,state,context,ZERO,acted
    from scripts.audit_x369_hidden_generators import capture
    from scripts.train_x369_ordered_action import encode_inputs
    if model.training or min(contexts,pairs)<1 or not 1<=horizon<=3:raise ValueError('invalid audit scope')
    rows=select_contexts(rows,contexts);Z=capture(model,rows,device,dtype).double()
    f=fit_joint(rows,Z)
    report=dict(rows=len(rows),rank=f['rank'],admitted=f['admitted'],six_phase_sequence_established=False,semantic_safety_established=False)
    tr=torch.tensor([r['split']=='train' for r in rows]);ev=~tr
    Y=torch.tensor([sum(state(r),()) for r in rows],dtype=torch.float64)
    lookup={(context(r),state(r)):i for i,r in enumerate(rows)}
    ids=ev.nonzero().flatten();remap=torch.full((len(rows),),-1,dtype=torch.long);remap[ids]=torch.arange(len(ids))
    rot=torch.tensor([lookup[context(rows[i]),acted('R',state(rows[i]))] for i in ids.tolist()])
    if (remap[rot]<0).any():raise ValueError('rotation orbit not closed')
    report['orbit']=orbit_spectrum(Z[ids],remap[rot],Y[ids])
    report['orbit_by_depth']={}
    for depth in sorted({len(rows[i]['word']) for i in ids.tolist()}):
        loc=torch.tensor([j for j,i in enumerate(ids.tolist()) if len(rows[i]['word'])==depth]);local=torch.full((len(ids),),-1,dtype=torch.long);local[loc]=torch.arange(len(loc))
        report['orbit_by_depth'][str(depth)]=orbit_spectrum(Z[ids[loc]],local[remap[rot][loc]],Y[ids[loc]])
    report['weight_spectra']={}
    with torch.no_grad():
        for name,layer in model.named_modules():
            if isinstance(layer,torch.nn.Linear):w=layer.weight
            elif hasattr(layer,'base') and hasattr(layer,'delta'):w=layer.base+layer.delta()
            else:continue
            sv=torch.linalg.svdvals(w.detach().double().cpu());top=float(sv[0]);rank=int(torch.linalg.matrix_rank(w.detach().double().cpu()))
            report['weight_spectra'][name]=dict(stable_rank=float(sv.square().sum()/sv[0].square()) if top else None,condition_nonzero_support=float(sv[0]/sv[rank-1]) if rank else None,rank=rank,singular_values=sv.tolist())
    if not f['admitted']:return report
    V,q,A,ix=(f[k] for k in ('V','q','A','targets'))
    U=torch.linalg.pinv(Y[tr])@q[tr];decoder=torch.linalg.pinv(q[tr])@Y[tr]
    report.update(common_hom=common_hom(A,U),relations=f['relations'],train=f['train'],composition=f['composition'],chart=dict(U=U.tolist(),A={g:a.tolist() for g,a in A.items()},V=V.tolist(),decoder=decoder.tolist(),composition_q=q[ids].tolist(),composition_keys=[[rows[i]['word'],list(rows[i]['guards']),list(rows[i]['mean']),list(rows[i]['residual'])] for i in ids.tolist()]))
    # Remap target indices to the composition-only closed set.
    ids=ev.nonzero().flatten();remap=torch.full((len(rows),),-1,dtype=torch.long);remap[ids]=torch.arange(len(ids))
    targets={g:remap[t[ids]] for g,t in ix.items()}
    if any((v<0).any() for v in targets.values()):raise ValueError('composition orbit not closed')
    report.update(cube=cube_metrics(A,q[ids],targets),anchored=anchored_phase(q[ids]@decoder,Y[ids]))
    lookup={(context(r),state(r)):i for i,r in enumerate(rows)}
    base=Z[[lookup[context(r),(ZERO,ZERO)] for r in rows]]
    residual=Z-base-q@V.T
    # A phase null rotates only residual triad in the decoded chart. It
    # preserves decoded DC and phase power, NOT ambient latent norm.
    if int(torch.linalg.matrix_rank(decoder))!=6:return dict(report,intervention='decoder_rank_deficient')
    eligible=[i for i in ids.tolist() if i!=int(ix['R'][i]) and triad_phase(Y[i,3:]).abs()>0]
    chosen=eligible if len(eligible)<=pairs else [eligible[j*len(eligible)//pairs] for j in range(pairs)]
    if not chosen:return dict(report,intervention='no_nonfixed_phase_pairs')
    j=ix['R'][chosen];natural=q[j];decoded=natural@decoder
    scrambled=decoded.clone();scrambled[:,3:]=scrambled[:,3:].roll(1,-1)
    null=scrambled@torch.linalg.inv(decoder)
    arms=[Z[j],base[j]+natural@V.T,base[j]+(q[chosen]@A['R'])@V.T+residual[j],base[j]+null@V.T+residual[j],Z[chosen]]
    names=['natural','remove_outside_chart','fitted_R_natural_residual','relative_phase_null','unedited']
    shape=(-1,model.experts,2,model.width) if model.paired else (-1,model.experts,model.width)
    stats={str(d):{k:dict(probability_l1=0.,output_l2=0.,history_l2=0.,route_mismatch=0,sign_mismatch=0,weight_l1=0.,comparisons=0) for k in names} for d in range(1,horizon+1)}
    with torch.no_grad():
        for depth in range(1,horizon+1):
            for word in product('abjn',repeat=depth):
                ops=torch.tensor([encode_inputs(dict(rows[i],word=''.join(word)),'signed')[1] for i in chosen],device=device,dtype=dtype)
                observed=[]
                for z in arms:
                    h=z.to(device,dtype).reshape(shape)
                    for t in range(depth):y,h,ii,ss,ww=model.advance(ops[:,t],h)
                    p=torch.cat([p.softmax(-1) for p in model.consume(y,h)],-1);observed.append((p,y,h,ii,ss,ww))
                p0,y0,h0,i0,s0,w0=observed[0]
                for name,(p,y,h,i,s,w) in zip(names,observed):
                    v=stats[str(depth)][name];v['comparisons']+=len(chosen);v['probability_l1']+=float((p-p0).abs().sum());v['output_l2']+=float((y-y0).flatten(1).norm(dim=1).sum());v['history_l2']+=float((h-h0).flatten(1).norm(dim=1).sum());v['route_mismatch']+=int((i!=i0).flatten(1).any(1).sum());v['sign_mismatch']+=int((s!=s0).flatten(1).any(1).sum());v['weight_l1']+=float((w-w0).abs().sum())
    for depth in stats.values():
        for v in depth.values():
            for k in ('probability_l1','output_l2','history_l2','weight_l1'):v[k]/=v['comparisons']
    report.update(intervention=dict(pairs=len(chosen),eligible=len(eligible),decoded_phase_power_delta=float((triad_phase(decoded.reshape(-1,2,3)).abs()-triad_phase(scrambled.reshape(-1,2,3)).abs()).abs().max()),by_depth=stats,cleanup_certified=False))
    return report

