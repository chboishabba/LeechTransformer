"""Source-owned grokking trajectory measurements, not a mechanism claim.

Use the original nine-point task and original punctured-nine Fourier
basis. Preserve all route/sign histories at the actual source step.
Only the small Gram eigenproblem is on CPU (diagnostic, not inference).
"""
import torch
from models.punctured_nine_wave_prior import SITES,FREQUENCY_PAIRS,real_fourier9_basis
from scripts.audit_x369_typed_mod3_prediction_consumer import TASK_PAIRS
from models.e8_root_prior import e8_router_invariants
from models.punctured_nine_wave_prior import pair_wave_router_features


def spectral(x):
    if x.shape[0]!=9:raise ValueError('expected nine actual task rows')
    a=x.detach().reshape(9,-1).double()
    if not bool(torch.isfinite(a).all()):raise ValueError('nonfinite activation')
    order=torch.tensor([TASK_PAIRS.index((i%3,j%3)) for i,j in SITES],device=a.device)
    a=a.index_select(0,order)
    centered=a-a.mean(0,keepdim=True)
    basis=real_fourier9_basis(device=a.device,dtype=a.dtype)
    modes=(basis.T@centered).reshape(4,2,-1).square().sum((1,2))
    energy=centered.square().sum()
    gram=(centered@centered.T).cpu()
    eig=torch.linalg.eigvalsh(gram).clamp_min(0)
    total=float(energy)
    return dict(centered_energy=total,dc_energy=float((a.mean(0).square().sum())*9),
        frequency_pairs=[list(v) for v in FREQUENCY_PAIRS],
        character_pair_energy=modes.tolist(),
        character_pair_fraction=(modes/energy).tolist() if total else None,
        gram_eigenvalues=eig.tolist(),
        effective_rank=float(eig.sum().square()/eig.square().sum()) if float(eig.square().sum()) else None,
        parseval_absolute_residual=abs(float(modes.sum())-total),
        marchenko_pastur_null_calibrated=False)


def measure(m,pairs,selector,is_learner):
    rows=[]
    def observer(t,x,y,previous_h,h,ids,signs,w):
        # Original actual selected route and post-step history; no recomputed
        # route or guessed semantic labelling of sign.
        p=w.detach().double()
        row=dict(token_step=t,activation=spectral(x),expert_output=spectral(y),
                 history=spectral(h),selected_ids=ids.detach().tolist(),
                 local_signs=signs.detach().tolist(),selected_weights=p.tolist(),
                 selected_entropy_nats=float(-(p*p.clamp_min(1e-300).log()).sum(-1).mean()),
                 expert_occupancy=[int((ids==i).sum()) for i in range(m.history.num_experts)])
        # Measure the original paired pre-history/activation in the
        # existing E8 and punctured-nine candidate charts, without
        # claiming these diagnostic features drove the actual router.
        a=x[:,None,:].expand_as(previous_h)
        if x.shape[-1]%8==0:
            row['e8_candidate_invariant_spectrum']=spectral(e8_router_invariants(a,previous_h))
        if x.shape[-1]%24==0:
            row['punctured_nine_candidate_spectrum']=spectral(pair_wave_router_features(a,previous_h,compact=True))
        row['candidate_chart_is_actual_lattice_membership']=False
        rows.append(row)
    with torch.no_grad():
        logits=m(pairs,selector,observer=observer) if is_learner else m(pairs)
    return logits,dict(logit_spectrum=spectral(logits),actual_router_steps=rows,
        actual_source_route_measured=is_learner,phase_causal_intervention_measured=False,
        semantic_interpretation_inferred=False,grokking_mechanism_certified=False)
