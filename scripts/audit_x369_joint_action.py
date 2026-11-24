"""Joint signed mean/residual action on a closed mean orbit; no F5 data."""
from itertools import product
from collections import defaultdict
import hashlib,json
import torch
from scripts.audit_x369_ordered_signed_task import endpoint,TRIADS,WORDS,GUARDS,TRAIN_WORDS

ZERO=(0,0,0)
MEANS=(ZERO,)+tuple(tuple(s*int(i==j) for i in range(3)) for j in range(3) for s in (-1,1))
ACTIONS={'R':('a',(1,0,0,0)),'J':('j',(0,0,1,0)),'N':('n',(0,0,0,1))}

def joint_records():
    for w,g,m,d in product(WORDS,GUARDS,MEANS,TRIADS):
        a,b,q=endpoint(w,g,m,d)
        yield dict(word=w,guards=g,mean=m,residual=d,mean_label=a,signed_label=b,gram_label=q,split='train' if w in TRAIN_WORDS else 'composition')

def task_digest(rows):
    h=hashlib.sha256()
    for r in rows:h.update((json.dumps(r,sort_keys=True,separators=(',',':'))+'\n').encode())
    return h.hexdigest()

def context(r):return (r['word'],r['guards'])
def state(r):return (r['mean_label'],r['signed_label'])
def acted(name,y):return endpoint(*ACTIONS[name],*y)[:2]

def select_contexts(rows,n):
    if n<1:raise ValueError('positive context budget required')
    groups=defaultdict(list)
    for r in rows:groups[context(r)].append(r)
    out=[]
    for split in ('train','composition'):
        keys=sorted(k for k in groups if groups[k][0]['split']==split)
        if not keys:raise ValueError('missing split')
        chosen=keys if len(keys)<=n else [keys[j*len(keys)//n] for j in range(n)]
        for k in chosen:out.extend(groups[k])
    return out

def fit_joint(rows,Z,rtol=1e-10):
    Z=Z.detach().double().cpu()
    if Z.ndim!=2 or len(Z)!=len(rows) or not torch.isfinite(Z).all():raise ValueError('bad history matrix')
    lookup={(context(r),state(r)):i for i,r in enumerate(rows)}
    if len(lookup)!=len(rows):raise ValueError('duplicate context state')
    zeros=torch.tensor([lookup[context(r),(ZERO,ZERO)] for r in rows]);X=Z-Z[zeros]
    Y=torch.tensor([sum(state(r),()) for r in rows],dtype=torch.float64)
    tr=torch.tensor([r['split']=='train' for r in rows]);ev=~tr
    if not tr.any() or not ev.any() or int(torch.linalg.matrix_rank(Y[tr],rtol=rtol))!=6:raise ValueError('missing full training span')
    U=torch.linalg.pinv(Y[tr],rtol=rtol)@X[tr];_,s,Vh=torch.linalg.svd(U,full_matrices=False)
    rank=int((s>s.max()*rtol).sum())
    if rank!=6:return dict(rank=rank,admitted=False)
    V=Vh[:6].T.contiguous();q=X@V
    if int(torch.linalg.matrix_rank(q[tr],rtol=rtol))!=6:return dict(rank=rank,admitted=False,reason='projected_rank')
    decoder=torch.linalg.pinv(q[tr],rtol=rtol)@Y[tr]
    ix={g:torch.tensor([lookup[context(r),acted(g,state(r))] for r in rows]) for g in ACTIONS}
    A={g:torch.linalg.pinv(q[tr],rtol=rtol)@q[t[tr]] for g,t in ix.items()}
    R,J,N=(A[g] for g in ACTIONS);I=torch.eye(6,dtype=q.dtype)
    rel={'R3':R@R@R-I,'J2':J@J-I,'N2':N@N-I,'JRJ_R2':J@R@J-R@R,'NR_RN':N@R-R@N,'NJ_JN':N@J-J@N}
    rms=lambda x:float(x.square().mean().sqrt())
    def metrics(mask):
        e=q[mask]@decoder-Y[mask]
        return dict(rows=int(mask.sum()),mean_rmse=rms(e[:,:3]),signed_rmse=rms(e[:,3:]),dc_rmse=rms(e.reshape(-1,2,3).sum(-1)/3**.5),outside_chart_rmse=rms(X[mask]-q[mask]@V.T),generator_rmse={g:rms(q[mask]@A[g]@decoder-Y[t[mask]]) for g,t in ix.items()},ordered_RJ_rmse=rms(q[mask]@R@J@decoder-Y[ix['J'][ix['R'][mask]]]),ordered_JR_rmse=rms(q[mask]@J@R@decoder-Y[ix['R'][ix['J'][mask]]]))
    return dict(rank=rank,admitted=True,V=V,q=q,A=A,targets=ix,train=metrics(tr),composition=metrics(ev),relations={g:float(v.norm()) for g,v in rel.items()},nontriviality={g:float((a-I).norm()) for g,a in A.items()},commutator=float((R@J-J@R).norm()))

def audit_joint(model,rows,device,dtype,contexts=12,pairs=128):
    from scripts.audit_x369_hidden_generators import capture
    from scripts.train_x369_ordered_action import encode_inputs
    from scripts.audit_x369_consumer_closure import factorial_states,factorial_effect
    if model.training or pairs<1:raise ValueError('eval and positive pair budget required')
    rows=select_contexts(rows,contexts);Z=capture(model,rows,device,dtype).double();f=fit_joint(rows,Z)
    if not f['admitted']:return dict(f,learned_same_action_established=False)
    V,q,A,ix=(f[k] for k in ('V','q','A','targets'));report={k:v for k,v in f.items() if k not in ('V','q','A','targets')}
    shape=(-1,model.experts,2,model.width) if model.paired else (-1,model.experts,model.width)
    results={}
    with torch.no_grad():
        for g,t in ix.items():
            elig=[i for i,r in enumerate(rows) if r['split']=='composition' and i!=int(t[i])]
            chosen=elig if len(elig)<=pairs else [elig[j*len(elig)//pairs] for j in range(pairs)]
            if not chosen:continue
            j=t[chosen];arms=factorial_states(Z[chosen],Z[j],V,q[chosen],q[j],A[g]);err=[0.]*4;hist=[0.]*4;route=[0]*4;sign=[0]*4;weight=[0.]*4;mix=0.;n=0
            for letter in 'abjn':
                op=torch.tensor([encode_inputs(dict(rows[i],word=letter),'signed')[1][0] for i in chosen],device=device,dtype=dtype)
                obs=[]
                for h in arms:
                    y,h,ids,ss,ww=model.advance(op,h.to(device,dtype).reshape(shape));p=torch.cat([z.softmax(-1) for z in model.consume(y,h)],-1);obs.append((p,h,ids,ss,ww))
                p0,h0,i0,s0,w0=obs[-1]
                for k,(p,h,i,s,w) in enumerate(obs):
                    err[k]+=float((p-p0).abs().sum());hist[k]+=float((h-h0).flatten(1).norm(dim=-1).sum());route[k]+=int((i!=i0).any(-1).sum());sign[k]+=int((s!=s0).flatten(1).any(-1).sum());weight[k]+=float((w-w0).abs().sum())
                mix+=float(factorial_effect([z[0] for z in obs])[2].abs().sum());n+=len(chosen)
            results[g]=dict(eligible=len(elig),comparisons=n,probability_l1=[v/n for v in err],history_l2=[v/n for v in hist],route_mismatch=route,sign_mismatch=sign,weight_l1=[v/n for v in weight],mixed_probability_l1=mix/n)
    return dict(report,intervention_arms=['fitted_old','oracle_chart_old','fitted_natural_residual','natural_full'],interventions=results,learned_same_action_established=False)
