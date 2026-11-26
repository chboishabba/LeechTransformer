"""Frozen translated-source repair: exact seed sector split + projected affine solve.
Original Leech-LILA Anatolii Kornienko; DASHI extension.
Native recurrence only. Train-only linear readout fit, no optimizer steps.
"""
import argparse,io,json,math,subprocess,time
from collections import defaultdict
from pathlib import Path
from hashlib import sha256
import torch
from models.geometric_reflection_fibre import GROUP,inv
from models.geometric_stabilizer_codec import same_bits
from scripts.x369_source_geometry import SourceGeometryOrderedLearner
from scripts.x369_source_acceptance import source_identity,read_checkpoint,model_digest,execution_contract,translation_provenance
from scripts.x369_translate_paired_checkpoint import exclusive
from scripts.audit_x369_joint_action import joint_records,select_contexts,task_digest
from scripts.train_x369_ordered_action import batch,HEADS


def exact_seed_split(seed,experts,width):
    """Reynolds projection on Hom(task, paired histories), exact rationals.
    Retain B=B_eq+B_res; this does not change the fixed source seed.
    """
    import sympy as S
    from models.geometric_paired_history import pair_action
    A=S.Matrix(seed['constructed_basis']);B=S.Matrix(seed['historical_basis'])
    L=S.Matrix(seed['constructed_coordinates'])
    assert A.shape==B.shape==(6,experts*2*width) and A*L==S.eye(6)
    def action(M,g):
        rows=[]
        for row in M.tolist():
            out=[]
            for e in range(experts):
                i=e*2*width;out.extend(sum(pair_action(g,(tuple(row[i:i+width]),tuple(row[i+width:i+2*width]))),()))
            rows.append(out)
        return S.Matrix(rows)
    C={g:action(A,g)*L for g in GROUP}
    E=sum((C[inv(g)]*action(B,g) for g in GROUP),S.zeros(*B.shape))/len(GROUP)
    R=B-E
    assert all(C[g]*E==action(E,g) for g in GROUP)
    assert sum(e*r for e,r in zip(E,R))==0
    assert E+R==B
    mat=lambda M:[[str(v) for v in row] for row in M.tolist()]
    return dict(equivariant_rank=E.rank(),residual_rank=R.rank(),original_rank=B.rank(),
        equivariant_basis=mat(E),retained_residual=mat(R),
        equivariant_energy=str(sum(v*v for v in E)),residual_energy=str(sum(v*v for v in R)),
        original_energy=str(sum(v*v for v in B)),orthogonal=True,reversible_sum=True,
        fixed_source_seed_modified=False,global_recurrence_conjugacy=False)


def projection_groups(X,Y,masks,table):
    """X includes bias. Y in local label coordinates. CPU FP64 linear algebra.
    Group by actual stabilizer, with symmetric idempotent ideal projector P.
    Objective is sum ||X B P-Y||^2, not unprojected regression.
    """
    if X.ndim!=2 or Y.ndim!=2 or len(X)!=len(Y) or masks.shape!=(len(X),len(table)):
        raise ValueError('invalid projection data')
    eye=torch.eye(Y.shape[1],dtype=X.dtype,device=X.device)
    groups=[]
    for mask in torch.unique(masks,dim=0):
        ids=(masks==mask).all(1);ix=mask.nonzero().flatten().tolist()
        if not ix:raise ValueError('empty stabilizer')
        P=sum((eye[:,table[i]] for i in ix),torch.zeros_like(eye))/len(ix)
        if not torch.allclose(P,P.T,atol=1e-12,rtol=0) or not torch.allclose(P@P,P,atol=1e-12,rtol=0):raise ValueError('invalid subgroup projection')
        x,y=X[ids],Y[ids];G=x.T@x;H=x.T@(y@P)
        groups.append((G,H,P,x,y))
    return groups


def solve_projected(X,Y,masks,table,B0,ridge=1e-3,max_iter=160,tol=1e-9):
    """Unique ridge-regularized correction to original affine weights.
    Normal equation on delta; no gradient optimizer, no heldout selection.
    Reports residual and an a posteriori distance bound residual/ridge
    for the ideal arithmetic normal system (not a floating error theorem).
    """
    if not math.isfinite(ridge) or ridge<=0 or max_iter<1 or not 0<tol<1:raise ValueError('invalid solve scope')
    groups=projection_groups(X,Y,masks,table)
    if B0.shape!=(X.shape[1],Y.shape[1]):raise ValueError('readout shape')
    def normal(D):return ridge*D+sum((G@D@P for G,_,P,_,_ in groups),torch.zeros_like(D))
    rhs=sum((H-G@B0@P for G,H,P,_,_ in groups),torch.zeros_like(B0))
    D=torch.zeros_like(B0);r=rhs.clone();p=r.clone();rr=(r*r).sum();initial=float(rr.sqrt());n=0
    for n in range(1,max_iter+1):
        if float(rr.sqrt())<=tol*max(1.,initial):n-=1;break
        ap=normal(p);den=(p*ap).sum()
        if not bool(torch.isfinite(den)) or not bool(den>0):raise RuntimeError('nonpositive normal system')
        alpha=rr/den;D=D+alpha*p;r=r-alpha*ap;nn=(r*r).sum();p=r+(nn/rr)*p;rr=nn
    # Recompute residual, rather than trust accumulated CG residual.
    residual=float((rhs-normal(D)).norm())
    def loss(B):return sum(float(((x@B@P-y)**2).sum()) for _,_,P,x,y in groups)
    floor=sum(float(((y-y@P)**2).sum()) for _,_,P,_,y in groups)
    before=loss(B0);after=loss(B0+D);objective=after+ridge*float((D*D).sum())
    if not all(math.isfinite(v) for v in (residual,before,after,objective,floor)):raise RuntimeError('nonfinite solve')
    return B0+D,dict(iterations=n,normal_residual=residual,initial_residual=initial,
        converged=residual<=tol*max(1.,initial),conditional_distance_bound=residual/ridge,
        original_sse=before,solved_sse=after,regularized_objective=objective,stabilizer_irreducible_sse=floor,
        ridge=ridge,groups=len(groups),rows=len(X))


@torch.no_grad()
def capture(model,rows,batch_size,seconds):
    """Only original controlled source supplies features and marks."""
    p=next(model.parameters());end=time.monotonic()+seconds;groups=defaultdict(list)
    for i,r in enumerate(rows):groups[r['split'],len(r['word'])].append(i)
    features={};masks={};marks={};labels={}
    for _,ix in sorted(groups.items()):
        for start in range(0,len(ix),batch_size):
            if time.monotonic()>=end:raise TimeoutError('incomplete declared capture; no readout fit')
            ids=ix[start:start+batch_size];s,o,y=batch([rows[i] for i in ids],'signed',p.device,p.dtype)
            from scripts.x369_source_geometry import literal_controls
            c,t=literal_controls(o);_,trace=model.rollout_group(s,c,t);z=trace[-1]
            f=torch.cat((z.local_output.flatten(1),z.local_history.flatten(1)),1).cpu().double()
            for i,a,b,d,e in zip(ids,f,z.stabilizer.cpu(),z.world_to_local.cpu(),y.cpu()):
                features[i]=a;masks[i]=b;marks[i]=d;labels[i]=e
    return tuple(torch.stack([m[i] for i in range(len(rows))]) for m in (features,masks,marks,labels))


@torch.no_grad()
def score_native(model,rows,batch_size,seconds):
    p=next(model.parameters());end=time.monotonic()+seconds;groups=defaultdict(list);result={}
    for r in rows:groups[r['split'],len(r['word'])].append(r)
    for (split,_),rs in sorted(groups.items()):
        stat=result.setdefault(split,dict(rows=0,correct=[0]*3,ties=[0]*3,sse=[0.]*3))
        for start in range(0,len(rs),batch_size):
            if time.monotonic()>=end:raise TimeoutError('incomplete native score')
            s,o,y=batch(rs[start:start+batch_size],'signed',p.device,p.dtype);pred=model(s,o);stat['rows']+=len(y)
            for j,v in enumerate(pred):
                target=torch.nn.functional.one_hot(y[:,j],v.shape[-1]).to(v.dtype)
                top=v==v.amax(-1,keepdim=True);stat['correct'][j]+=int((top.gather(1,y[:,j,None]).squeeze(1)&(top.sum(1)==1)).sum());stat['ties'][j]+=int((top.sum(1)>1).sum())
                stat['sse'][j]+=float(((v.double()-target.double())**2).sum())
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('checkpoint','translation-receipt','output','receipt'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('sha256','translation-receipt-sha256','source-head'):p.add_argument('--'+k,required=True)
    p.add_argument('--contexts',type=int,default=5);p.add_argument('--guards',type=int,nargs=4,default=(1,2,1,1))
    p.add_argument('--batch-size',type=int,default=32);p.add_argument('--seconds',type=float,default=90);p.add_argument('--ridge',type=float,default=1e-3)
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    if min(a.contexts,a.batch_size)<1 or not 0<a.seconds<=90 or not math.isfinite(a.ridge) or a.ridge<=0:raise ValueError('invalid scope')
    if len({a.checkpoint.resolve(),a.translation_receipt.resolve(),a.output.resolve(),a.receipt.resolve()})!=4 or a.output.exists() or a.receipt.exists():raise ValueError('fresh distinct outputs required')
    from scripts.x369_source_acceptance import guard_controls
    guard_controls(a.guards);source=source_identity(root,a.source_head)
    data=a.translation_receipt.read_bytes()
    if sha256(data).hexdigest()!=a.translation_receipt_sha256:raise ValueError('parent receipt digest mismatch')
    parent=json.loads(data);prior=parent['source']
    # Source additions only: existing parent implementation must be unchanged.
    subprocess.run(['git','-C',str(root),'merge-base','--is-ancestor',prior['head'],source['head']],check=True)
    if any(source['files'].get(k)!=v for k,v in prior['files'].items()):raise ValueError('parent Python implementation changed')
    if not torch.cuda.is_available() or not torch.version.hip or not getattr(torch.cuda.get_device_properties(0),'gcnArchName','').startswith('gfx803'):raise RuntimeError('native gfx803 HIP required')
    dev=torch.device('cuda',0);cfg=parent['config'];dt=getattr(torch,cfg['dtype'])
    runtime=dict(torch=str(torch.__version__),hip=torch.version.hip,arch=torch.cuda.get_device_properties(dev).gcnArchName)
    model=SourceGeometryOrderedLearner(cfg['width'],cfg['experts'],expert_width=cfg['expert_width']).to(dev,dt).eval()
    model.load_state_dict(read_checkpoint(a.checkpoint,a.sha256,dt),strict=True);model.history.source.history_cap=cfg['cap']
    parent_digest=model_digest(model);execution=execution_contract(model)
    translation_provenance(a.translation_receipt,a.translation_receipt_sha256,a.sha256,prior,cfg,parent_digest,execution,runtime)
    if 'readout_repair' in parent:raise ValueError('use original translated candidate, no iterative heldout selection')
    seed=exact_seed_split(parent['seed'],cfg['experts'],cfg['width'])
    rows=select_contexts([r for r in joint_records() if r['guards']==tuple(a.guards)],a.contexts)
    train=[r for r in rows if r['split']=='train'];held=[r for r in rows if r['split']=='composition']
    frozen={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if not k.startswith('readout.')}
    original=score_native(model,rows,a.batch_size,a.seconds)
    F,K,M,T=capture(model,train,a.batch_size,a.seconds);X=torch.cat((F,torch.ones(len(F),1,dtype=F.dtype)),1)
    fit=[];solutions=[];off=0
    for j,n in enumerate(HEADS):
        tab=getattr(model,f'label_perm_{j}').cpu()
        Y=torch.nn.functional.one_hot(T[:,j],n).double().gather(1,tab[M])
        B0=torch.cat((model.readout.weight[off:off+n].detach().cpu().double().T,model.readout.bias[off:off+n].detach().cpu().double()[None]),0)
        B,rep=solve_projected(X,Y,K,tab,B0,a.ridge)
        fit.append(rep);solutions.append(B);off+=n
    # Accept/reject only on the training objective; heldout never selects.
    if any(not r['converged'] or r['regularized_objective']>r['original_sse']+1e-8*max(1.,r['original_sse']) for r in fit):
        exclusive(a.receipt,(json.dumps(dict(kind='x369_readout_repair_negative',status='linear_solve_not_admitted',parent_translation_sha256=a.translation_receipt_sha256,fit=fit,seed_split=seed,training_steps=0),indent=2)+'\n').encode());return
    with torch.no_grad():
        B=torch.cat(solutions,1).to(dev,dt);model.readout.weight.copy_(B[:-1].T);model.readout.bias.copy_(B[-1])
    repaired=score_native(model,rows,a.batch_size,a.seconds)
    for k,v in frozen.items():
        if not same_bits(v,model.state_dict()[k].cpu()):raise RuntimeError('non-readout state changed')
    if execution_contract(model)!=execution or source_identity(root,a.source_head)!=source:raise RuntimeError('execution/source changed')
    good=all(b<=c for b,c in zip(repaired['train']['sse'],original['train']['sse']))
    result=dict(kind='x369_translated_source_candidate',status='translated_candidate' if good else 'native_readout_not_improved',
        source=source,config=cfg,runtime=runtime,execution_contract=execution,historical_sha256=parent['historical_sha256'],
        parent_candidate_sha256=a.sha256,parent_translation_receipt_sha256=a.translation_receipt_sha256,
        seed=parent['seed'],seed_split=seed,readout_repair=dict(method='train_only_stabilizer_projected_ridge',fit=fit,
            baseline=original,repaired=repaired,train_task_sha256=task_digest(train),heldout_task_sha256=task_digest(held),
            guards=a.guards,contexts=a.contexts,heldout_used_for_selection=False,nonreadout_bitwise=True,linear_fit_performed=True),
        candidate_model_sha256=model_digest(model),candidate_sha256=None,training_steps=0,optimizer_migrated=False,
        original_new_source_training=False,release_admitted=False,universal_full_function_certified=False)
    if good:
        buf=io.BytesIO();torch.save({k:v.detach().cpu().clone() for k,v in model.state_dict().items()},buf);raw=buf.getvalue()
        result['candidate_sha256']=sha256(raw).hexdigest();exclusive(a.output,raw)
        check=read_checkpoint(a.output,result['candidate_sha256'],dt)
        if any(not same_bits(v,model.state_dict()[k].cpu()) for k,v in check.items()):raise RuntimeError('reopened repair differs')
    data=(json.dumps(result,indent=2,allow_nan=False)+'\n').encode();exclusive(a.receipt,data)
    print(json.dumps(dict(status=result['status'],candidate_sha256=result['candidate_sha256'],receipt_sha256=sha256(data).hexdigest(),training_steps=0,readout_linear_fit=True)))


if __name__=='__main__':main()
