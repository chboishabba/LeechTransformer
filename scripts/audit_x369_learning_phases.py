"""Checkpoint-indexed phase diagnostics; original Leech-LILA: Anatolii Kornienko.
User six-phase hypothesis; DASHI extension. No phase classification or release claim.
"""
from itertools import product, permutations
import math
import torch


def rms(x):
    return float(x.abs().square().mean().sqrt())


def triad_phase(x):
    if x.shape[-1]!=3:raise ValueError('triad required')
    x=x.double()
    return torch.complex(x[...,0]-(x[...,1]+x[...,2])/2,(x[...,1]-x[...,2])*(math.sqrt(3)/2))


def mod3_spectrum(logits):
    """Full public nine-point evaluation, never a training/selection objective."""
    if logits.shape!=(9,3) or not torch.isfinite(logits).all():raise ValueError('finite canonical logits required')
    z=triad_phase(logits).reshape(3,3)
    c=torch.fft.fft2(z)/9
    energy=c.abs().square();total=float(energy.sum());target=c[1,1]
    return dict(dc=logits.double().mean(-1).tolist(),coefficients_real=c.real.tolist(),coefficients_imag=c.imag.tolist(),energy=energy.tolist(),total_energy=total,target_fraction=float(energy[1,1])/total if total else None,target_phase=float(torch.angle(target)) if float(target.abs()) else None,zero_response=total==0,conjugate_energy=float(energy[2,2]),mixed_energy=float(energy[1,2]+energy[2,1]))


def action_matrices():
    r=torch.tensor([[0.,1,0],[0,0,1],[1,0,0]],dtype=torch.float64)
    j=torch.tensor([[1.,0,0],[0,0,1],[0,1,0]],dtype=torch.float64)
    return dict(R=torch.block_diag(r,r),J=torch.block_diag(j,-j),N=-torch.eye(6,dtype=r.dtype))


def common_hom(A,U):
    """Row convention rho_g U = U A_g. Independent fit U supplied, not imposed."""
    rho=action_matrices();I=torch.eye(6,dtype=U.dtype)
    # Row-major flatten: vec_row(B U C) = (B kron C^T) vec_row(U).
    C=torch.cat([torch.kron(rho[g],I)-torch.kron(I,A[g].T.contiguous()) for g in rho])
    _,s,Vh=torch.linalg.svd(C,full_matrices=False)
    sv=torch.linalg.svdvals(U);rank=int(torch.linalg.matrix_rank(U))
    # Diagnostic projection onto numerical near-nullspace; not existence proof.
    mask=s<=max(float(s.max()),1.)*1e-8
    P=Vh[mask];projected=(P.T@(P@U.flatten())).reshape(6,6)
    return dict(singular_values=s.tolist(),numerical_nullity=int(mask.sum()),candidate_rank=rank,candidate_condition=float(sv[0]/sv[-1]) if rank==6 else None,candidate_relative_residual=float((C@U.flatten()).norm()/U.norm()) if U.norm()>0 else None,projected_candidate_rank=int(torch.linalg.matrix_rank(projected)),exact_invertible_hom_certified=False)


def cube_metrics(A,q,targets):
    """Indexed six faces and six paths, row matrices, compared to natural endpoint."""
    I=torch.eye(6,dtype=q.dtype);R,J,N=(A[g] for g in 'RJN')
    def edge(v,a):
        return (R if v[1]==0 else R@R,J,N)[a]
    def vertex(v):
        ix=torch.arange(len(q))
        for bit,g in zip(v,'RJN'):
            if bit:ix=targets[g][ix]
        return ix
    faces={};paths={}
    for a,b in ((0,1),(0,2),(1,2)):
        c=3-a-b
        for side in (0,1):
            v=[0,0,0];v[c]=side;va=v.copy();va[a]=1;vb=v.copy();vb[b]=1;end=va.copy();end[b]=1
            x=q[vertex(v)];p=x@edge(v,a)@edge(va,b);r=x@edge(v,b)@edge(vb,a);t=q[vertex(end)]
            faces[f'{a}{b}:{side}']=dict(path_rmse=rms(p-r),left_natural_rmse=rms(p-t),right_natural_rmse=rms(r-t))
    for order in permutations(range(3)):
        v=[0,0,0];B=I
        for a in order:B=B@edge(v,a);v[a]=1
        paths[''.join(map(str,order))]=rms(q@B-q[vertex(v)])
    return dict(faces=faces,paths=paths,universal_closure_certified=False)


def orbit_spectrum(Z,ix):
    """Raw feature response across natural R orbit; no claim of physical oscillation."""
    x=torch.stack((Z,Z[ix],Z[ix[ix]]),dim=1).double()
    c=torch.fft.fft(x,dim=1)/3
    e=c.abs().square().mean(0);total=float(e.sum())
    return dict(feature_energy=e.tolist(),total_energy=total,dc_energy=float(e[0].sum()),phase_energy=float(e[1:].sum()),zero_response=total==0,complex_r1_real=c[:,1].real.mean(0).tolist(),complex_r1_imag=c[:,1].imag.mean(0).tolist())


def anchored_phase(pred,true):
    p=triad_phase(pred.reshape(-1,2,3));t=triad_phase(true.reshape(-1,2,3))
    out=[]
    for j in range(2):
        a,b=p[:,j],t[:,j];mask=b.abs()>0;den=b[mask].abs().square().sum()
        c=(a[mask]*b[mask].conj()).sum()/den if den>0 else torch.tensor(0j)
        # Anchored coefficient can be zero even with high phase power.
        out.append(dict(nonzero_target=int(mask.sum()),zero_pred=int((a.abs()==0).sum()),anchored_real=float(c.real),anchored_imag=float(c.imag),anchored_phase=float(torch.angle(c)) if den>0 and c.abs()>0 else None,phase_rmse=rms(a-b),dc_rmse=rms(pred.reshape(-1,2,3)[:,j].mean(1)-true.reshape(-1,2,3)[:,j].mean(1))))
    return out


def phase_audit(model,rows,device,dtype,contexts=4,pairs=32,horizon=2):
    """Fixed full-context sample, training-only chart, composition-only intervention.
    Actual natural histories, not artificial target labels fed into the model.
    """
    from scripts.audit_x369_joint_action import select_contexts,fit_joint,state,context,ZERO
    from scripts.audit_x369_hidden_generators import capture
    from scripts.train_x369_ordered_action import encode_inputs
    if model.training or min(contexts,pairs)<1 or not 1<=horizon<=3:raise ValueError('invalid audit scope')
    rows=select_contexts(rows,contexts);Z=capture(model,rows,device,dtype).double()
    f=fit_joint(rows,Z)
    report=dict(rows=len(rows),rank=f['rank'],admitted=f['admitted'],six_phase_sequence_established=False,semantic_safety_established=False)
    if not f['admitted']:return report
    V,q,A,ix=(f[k] for k in ('V','q','A','targets'));tr=torch.tensor([r['split']=='train' for r in rows]);ev=~tr
    Y=torch.tensor([sum(state(r),()) for r in rows],dtype=q.dtype)
    U=torch.linalg.pinv(Y[tr])@q[tr];decoder=torch.linalg.pinv(q[tr])@Y[tr]
    report.update(common_hom=common_hom(A,U),relations=f['relations'],train=f['train'],composition=f['composition'])
    # Remap target indices to the composition-only closed set.
    ids=ev.nonzero().flatten();remap=torch.full((len(rows),),-1,dtype=torch.long);remap[ids]=torch.arange(len(ids))
    targets={g:remap[t[ids]] for g,t in ix.items()}
    if any((v<0).any() for v in targets.values()):raise ValueError('composition orbit not closed')
    report.update(cube=cube_metrics(A,q[ids],targets),anchored=anchored_phase(q[ids]@decoder,Y[ids]),orbit=orbit_spectrum(Z[ids],targets['R']))
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
