"""Independent residual generator fits on natural recurrent-history pairs.
Original Leech-LILA: Anatolii Kornienko; DASHI extension.
Fit only public training prefixes. No imposed group laws or final lockbox.
"""
from collections import defaultdict
import torch
from scripts.audit_x369_ordered_signed_task import endpoint
from scripts.train_x369_ordered_action import batch, encode_inputs

# Independently typed task actions on D; M and original prefix stay fixed.
ACTIONS={'R':('a',(1,0,0,0)), 'J':('j',(0,0,1,0)), 'N':('n',(0,0,0,1))}
ZERO=(0,0,0)


def transform(name,d):
    word,guards=ACTIONS[name]
    return endpoint(word,guards,ZERO,d)[1]


def key(r): return (r['word'],r['guards'],r['mean'])


def fit_natural_generators(rows,history,rtol=1e-10):
    """Fit a shared 3D residual chart, then unrestricted maps on its natural pairs.

    The chart is supervised by task D; generator fits use latent pairs rather
    than conjugating known matrices. Baselines are natural D=0 at each prefix.
    Evaluation rows are never used to choose the chart or fit any generator.
    """
    Z=history.detach().double().cpu()
    if Z.ndim!=2 or len(Z)!=len(rows) or not torch.isfinite(Z).all():
        raise ValueError('finite aligned complete histories required')
    lookup={(key(r),r['signed_label']):i for i,r in enumerate(rows)}
    if len(lookup)!=len(rows):raise ValueError('duplicate prefix/guard/endpoint')
    zeros=torch.tensor([lookup[key(r),ZERO] for r in rows]);X=Z-Z[zeros]
    D=torch.tensor([r['signed_label'] for r in rows],dtype=torch.float64)
    train=torch.tensor([r['split']=='train' for r in rows],dtype=torch.bool)
    if int(torch.linalg.matrix_rank(D[train],rtol=rtol))!=3:raise ValueError('incomplete training task residual span')
    U=torch.linalg.pinv(D[train],rtol=rtol)@X[train]
    _,sv,Vh=torch.linalg.svd(U,full_matrices=False)
    rank=int((sv>sv.max()*rtol).sum())
    if rank!=3:return dict(rank=rank,accepted=False)
    V=Vh[:3].T.contiguous();q=X@V
    # A nonzero full-rank projected training carrier is needed to fit maps.
    if int(torch.linalg.matrix_rank(q[train],rtol=rtol))!=3:return dict(rank=rank,accepted=False,reason='projected_rank')
    decoder=torch.linalg.pinv(q[train],rtol=rtol)@D[train]
    targets={name:torch.tensor([lookup[key(r),transform(name,r['signed_label'])] for r in rows]) for name in ACTIONS}
    A={name:torch.linalg.pinv(q[train],rtol=rtol)@q[ix[train]] for name,ix in targets.items()}
    R,J,N=(A[k] for k in ACTIONS);I=torch.eye(3,dtype=q.dtype)
    rel={'R3':R@R@R-I,'J2':J@J-I,'N2':N@N-I,'JRJ_R2':J@R@J-R@R,'NR_RN':N@R-R@N,'NJ_JN':N@J-J@N}
    # Row convention: chronological R then J is R@J.
    masks={'train':train}
    for t in sorted(set(len(r['word']) for r in rows)):
        mask=~train & torch.tensor([len(r['word'])==t for r in rows])
        if mask.any():masks[f'composition_t{t}']=mask
    metrics={}
    for split,mask in masks.items():
        def rms(x):return float(x.square().mean().sqrt())
        # Task-space comparisons avoid confusing an arbitrarily rescaled
        # chart with an improved geometric action. Full history is separate.
        group=dict(rows=int(mask.sum()),decode_rmse=rms(q[mask]@decoder-D[mask]),
                   outside_chart_rmse=rms(X[mask]-q[mask]@V.T),
                   generator_task_rmse={},natural_history_edit_rmse={})
        for name,ix in targets.items():
            pred=q[mask]@A[name]
            group['generator_task_rmse'][name]=rms(pred@decoder-D[ix[mask]])
            group['natural_history_edit_rmse'][name]=rms(Z[mask]+(pred-q[mask])@V.T-Z[ix[mask]])
        # Unseen multi-generator order: evaluate R then J vs J then R.
        rj=targets['J'][targets['R'][mask]];jr=targets['R'][targets['J'][mask]]
        group['ordered_RJ_rmse']=rms(q[mask]@R@J@decoder-D[rj]);group['ordered_JR_rmse']=rms(q[mask]@J@R@decoder-D[jr])
        metrics[split]=group
    # Relations measured in the frozen orthonormal chart; norms are
    # coordinate-dependent diagnostics, not general GL invariants.
    return dict(rank=rank,accepted=True,V=V,q=q,A=A,targets=targets,metrics=metrics,
                relation_frobenius={k:float(v.norm()) for k,v in rel.items()},
                nontriviality={k:float((v-I).norm()) for k,v in A.items()},
                commutator_frobenius=float((R@J-J@R).norm()))


def capture(model,rows,device,dtype):
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[len(r['word'])].append(i)
    Z={}
    with torch.no_grad():
        for ids in groups.values():
            for start in range(0,len(ids),256):
                ix=ids[start:start+256];s,o,_=batch([rows[i] for i in ix],'signed',device,dtype)
                Z.update(zip(ix,model.rollout(s,o)[1].flatten(1).cpu()))
    return torch.stack([Z[i] for i in range(len(rows))])


def audit_hidden_generators(model,rows,device,dtype):
    """Freeze independently fitted maps; intervene at a real post-prefix site.

    This is a diagnostic, not a same-action certification threshold. A chart
    can decode well yet omit a residual which the actual consumer requires.
    """
    Z=capture(model,rows,device,dtype);fit=fit_natural_generators(rows,Z)
    if not fit['accepted']:return dict(fit,learned_same_action_established=False)
    V,q,A,targets=(fit[k] for k in ('V','q','A','targets'));summary={k:v for k,v in fit.items() if k not in ('V','q','A','targets')}
    lookup={(key(r),r['residual']):i for i,r in enumerate(rows)}
    prefix=[]
    for j,r in enumerate(rows):
        if r['split']=='train' or len(r['word'])<2:continue
        pk=(r['word'][:-1],r['guards'],r['mean']);i=lookup.get((pk,r['residual']))
        if i is not None:prefix.append((i,j))
    shape=(-1,model.experts,2,model.width) if model.paired else (-1,model.experts,model.width)
    perm=torch.randperm(Z.shape[1],generator=torch.Generator().manual_seed(719))
    receipt={}
    with torch.no_grad():
        for name,ix in targets.items():
            pairs=[(i,int(ix[i]),j) for i,j in prefix if int(ix[i])!=i]
            errors=[0.]*3;hist=[0.]*3;ids_ok=[0]*3;signs_ok=[0]*3;weight_l1=[0.]*3
            for start in range(0,len(pairs),128):
                i,k,j=map(list,zip(*pairs[start:start+128]));h=Z[i];target=Z[k]
                delta=((q[i]@A[name]-q[i])@V.T).to(h)
                op=torch.tensor([encode_inputs(rows[t],'signed')[1][-1] for t in j],device=device,dtype=dtype)
                execute=lambda z:model.advance(op,z.to(device,dtype).reshape(shape))
                y0,h0,i0,s0,w0=execute(target);p0=model.consume(y0,h0)
                for a,z in enumerate((h,h+delta,h+delta[:,perm])):
                    y,h1,ii,ss,ww=execute(z);p=model.consume(y,h1)
                    errors[a]+=sum(float((u.softmax(-1)-v.softmax(-1)).abs().sum()) for u,v in zip(p,p0))
                    hist[a]+=float((h1-h0).flatten(1).norm(dim=-1).sum());ids_ok[a]+=int((ii==i0).all(-1).sum());signs_ok[a]+=int((ss==s0).flatten(1).all(-1).sum());weight_l1[a]+=float((ww-w0).abs().sum())
            n=len(pairs);receipt[name]=dict(nonfixed_pairs=n,probability_l1_per_row=[v/max(n,1) for v in errors],next_history_l2_per_row=[v/max(n,1) for v in hist],same_ordered_ids=ids_ok,same_all_signs=signs_ok,weights_l1_per_row=[v/max(n,1) for v in weight_l1])
    return dict(summary,intervention_order=['unedited','fitted_generator_delta','permuted_delta_null'],interventions=receipt,learned_same_action_established=False)
