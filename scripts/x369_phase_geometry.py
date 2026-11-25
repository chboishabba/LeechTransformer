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


def orbit_spectrum(Z,ix,Y):
    """Raw feature response across natural R orbit; no claim of physical oscillation."""
    x=torch.stack((Z,Z[ix],Z[ix[ix]]),dim=1).double()
    c=torch.fft.fft(x,dim=1)/3
    e=c.abs().square().mean(0);total=float(e.sum())
    anchors=triad_phase(Y.reshape(-1,2,3));coeff=[]
    for j in range(2):
        a=anchors[:,j];den=a.abs().square().sum()
        b=(a.conj()[:,None]*c[:,1]).sum(0)/den if den>0 else None
        coeff.append(None if b is None else dict(real=b.real.tolist(),imag=b.imag.tolist(),anchor_energy=float(den)))
    return dict(feature_energy=e.tolist(),total_energy=total,dc_energy=float(e[0].sum()),phase_energy=float(e[1:].sum()),zero_response=total==0,anchored_feature_coefficients=coeff)


def anchored_phase(pred,true):
    p=triad_phase(pred.reshape(-1,2,3));t=triad_phase(true.reshape(-1,2,3))
    out=[]
    for j in range(2):
        a,b=p[:,j],t[:,j];mask=b.abs()>0;den=b[mask].abs().square().sum()
        c=(a[mask]*b[mask].conj()).sum()/den if den>0 else torch.tensor(0j)
        # Anchored coefficient can be zero even with high phase power.
        out.append(dict(nonzero_target=int(mask.sum()),zero_pred=int((a.abs()==0).sum()),anchored_real=float(c.real),anchored_imag=float(c.imag),anchored_phase=float(torch.angle(c)) if den>0 and c.abs()>0 else None,phase_rmse=rms(a-b),dc_rmse=rms(pred.reshape(-1,2,3)[:,j].mean(1)-true.reshape(-1,2,3)[:,j].mean(1))))
    return out


def cross_time(previous,current):
    """Train-fitted change of chart; composition comparison is independent.
    No assumption that optimizer updates commute with the task.
    """
    if 'chart' not in previous or 'chart' not in current:return dict(admitted=False,reason='unidentified_chart')
    p,q=previous['chart'],current['chart']
    if p['composition_keys']!=q['composition_keys']:raise ValueError('checkpoint probe domain changed')
    mat=lambda x:torch.tensor(x,dtype=torch.float64)
    U,W=mat(p['U']),mat(q['U'])
    if min(int(torch.linalg.matrix_rank(U)),int(torch.linalg.matrix_rank(W)))!=6:return dict(admitted=False,reason='rank')
    B=torch.linalg.solve(U,W)
    old,new=mat(p['composition_q']),mat(q['composition_q'])
    return dict(admitted=True,transport=B.tolist(),condition=float(torch.linalg.cond(B)),composition_transport_rmse=rms(old@B-new),generator_square_defect={g:rms(mat(p['A'][g])@B-B@mat(q['A'][g])) for g in 'RJN'},subspace_projector_difference=float((mat(p['V'])@mat(p['V']).T-mat(q['V'])@mat(q['V']).T).norm()),learning_dynamics_equivariant=False)
