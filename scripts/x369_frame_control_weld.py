"""Native seed/frame/control repair, using actual frozen paired donor.
Original Leech-LILA Anatolii Kornienko; DASHI extension.
Six predeclared arms, no optimizer, no heldout selection. New architecture
diagnostics, not an ordinary SourceGeometryOrderedLearner checkpoint.
"""
import argparse,json,math,subprocess,time
from collections import defaultdict
from pathlib import Path
from hashlib import sha256
from fractions import Fraction
from types import SimpleNamespace
import torch
from models.geometric_reflection_fibre import GROUP
from models.geometric_orbit_descent import (act_tensor,canonical_controlled,
    apply_group_indices,project_stabilizer,decode_frame,conjugate_control)
from models.geometric_stabilizer_codec import same_bits
from scripts.x369_source_geometry import SourceGeometryOrderedLearner,literal_controls,signed_seed
from scripts.x369_source_acceptance import (source_identity,read_checkpoint,model_digest,
    execution_contract,translation_provenance,guard_controls)
from scripts.x369_translate_paired_checkpoint import restore,exclusive,reopen_signed_seed
from scripts.x369_translation_repair import exact_seed_split
from scripts.x369_projected_direct import solve_direct,collision_bounds
from scripts.audit_x369_joint_action import joint_records,select_contexts,task_digest
from scripts.train_x369_ordered_action import batch,HEADS

SEEDS=('canonical','equivariant','marked_historical')
CONTROLS=('static','conditioned')
FIT_GUARDS=((0,0,0,0),(1,2,1,1),(2,1,1,1))
GENERATORS={'R':(2,0,0),'J':(2,1,0),'N':(0,0,1)}


def transformed_seed(g,s):
    return torch.cat((act_tensor(g,s[:,:3]),(-1)**g[1]*act_tensor(g,s[:,3:6]),s[:,6:]),1)


def seed_matrix(values,device,dtype):
    return torch.tensor([[float(Fraction(x)) for x in row] for row in values],device=device,dtype=dtype)


def initial(model,s,c,seed_mode,E,B):
    """Causal initial lift; first control only, never future word or labels.
    Marked historical lift evaluates B on the canonical seed then projects
    its actual stabilizer. Unprojected local B and signed residual remain
    reconstructible from parent matrices and input, not claimed as world B.
    """
    h=signed_seed(s,model.experts,model.width)
    if seed_mode=='canonical':return h
    if seed_mode not in SEEDS:raise ValueError('unknown seed policy')
    _,ch,m,K=canonical_controlled(c[:,0],h,True)
    q=reopen_signed_seed(ch)[:,:6]
    local=(q@(E if seed_mode=='equivariant' else B)).reshape_as(h)
    return decode_frame(project_stabilizer(local,K,True),m,True)


def features(h):
    x=h.flatten(1)
    return torch.cat((x,torch.ones_like(x[:,:1])),1)


def fit_control(X,Y,base,ridge):
    """Minimize ||X D-(Y-base)||^2+ridge||D||^2, using SVD, no labels.
    Bound the correction by the observed training displacement radius.
    """
    if X.dtype!=torch.float64 or Y.dtype!=X.dtype or X.ndim!=2 or Y.ndim!=2 or len(X)!=len(Y) or not len(X):raise ValueError('bad control data')
    if not math.isfinite(ridge) or ridge<=0 or not all(torch.isfinite(v).all() for v in (X,Y,base)):raise ValueError('bad control fit')
    y=Y-base;U,s,Vh=torch.linalg.svd(X,full_matrices=False)
    D=Vh.T@((s/(s.square()+ridge))[:,None]*(U.T@y))
    residual=float((X.T@(X@D-y)+ridge*D).norm());rhs=float((X.T@y).norm())
    radius=float(y.norm(dim=1).max());unbounded=X@D
    bounded=bound_delta(unbounded,radius)
    before=float(y.square().sum());after=float((unbounded-y).square().sum())
    rep=dict(rows=len(X),rank=int((s>s.max()*max(X.shape)*torch.finfo(X.dtype).eps).sum()),
        normal_residual=residual,initial_residual=rhs,converged=residual<=1e-9*max(1.,rhs),
        static_sse=before,unbounded_sse=after,regularized_objective=after+ridge*float(D.square().sum()),
        bounded_sse=float((bounded-y).square().sum()),correction_radius=radius,ridge=ridge)
    if not all(math.isfinite(v) for v in rep.values() if isinstance(v,(float,int))):raise ValueError('nonfinite control fit')
    return D,rep


def bound_delta(x,radius):
    n=x.norm(dim=-1,keepdim=True)
    return x*torch.clamp(torch.as_tensor(radius,device=x.device,dtype=x.dtype)/n.clamp_min(torch.finfo(x.dtype).tiny),max=1.)


def groups(rows):
    g=defaultdict(list)
    for r in rows:g[r['split'],len(r['word'])].append(r)
    return [v for _,v in sorted(g.items())]


@torch.no_grad()
def capture_donor(old,model,rows,batch_size,seconds,ridge):
    """Actual historical training-prefix activations; not endpoint labels."""
    p=next(old.parameters());end=time.monotonic()+seconds;data=defaultdict(lambda:[[],[]]);steps=0
    for rs in groups(rows):
        for start in range(0,len(rs),batch_size):
            if time.monotonic()>=end:raise TimeoutError('donor capture budget')
            s,o,_=batch(rs[start:start+batch_size],'signed',p.device,p.dtype)
            c,t=literal_controls(o);h=old.initial(s)
            for j in range(c.shape[1]):
                x=old.embed(o[:,j]);cc,ch,m,_=canonical_controlled(c[:,j],h,True)
                desired=apply_group_indices(x,m)
                for k in torch.unique(t[:,j]*len(GROUP)+cc).tolist():
                    mask=t[:,j]*len(GROUP)+cc==k
                    data[k][0].append(features(ch[mask]).cpu().double())
                    data[k][1].append(desired[mask].cpu().double())
                h=old.history.step(x,h,train_sign=False).history;steps+=len(s)
    fits={};params={}
    for k,(xs,ys) in sorted(data.items()):
        X,Y=torch.cat(xs),torch.cat(ys);base=model.control_embed.weight[k].cpu().double()
        D,r=fit_control(X,Y,base,ridge)
        if not r['converged'] or r['regularized_objective']>r['static_sse']+1e-8*max(1.,r['static_sse']):raise RuntimeError('control fit not admitted')
        fits[str(k)]=r;params[k]=(D.to(p.device,p.dtype),r['correction_radius'])
    expected={0,8,12,20,24,34,36,37}
    if set(params)!=expected:raise ValueError('training scope must cover all eight canonical legal keys')
    return params,dict(rows=len(rows),prefix_steps=steps,task_sha256=task_digest(rows),fits=fits,
        coefficients={str(k):dict(delta=D.cpu().tolist(),radius=r) for k,(D,r) in params.items()},
        fitted_from_historical_activations_only=True)


def local_activation(model,tag,cc,ch,policy,params):
    keys=tag*len(GROUP)+cc;x=model.control_embed(keys)
    if policy=='static':return x
    if policy!='conditioned':raise ValueError('unknown control policy')
    x=x.clone();covered=torch.zeros_like(keys,dtype=torch.bool)
    for k,(D,r) in params.items():
        m=keys==k
        if bool(m.any()):
            x[m]=x[m]+bound_delta(features(ch[m])@D,r);covered|=m
    if not bool(covered.all()):raise ValueError('unsupported canonical control')
    return x


@torch.no_grad()
def rollout(model,s,c,t,seed_mode,policy,E,B,params):
    h=initial(model,s,c,seed_mode,E,B);trace=[]
    for j in range(c.shape[1]):
        if policy=='static':
            z=model.history.step_controlled(c[:,j],h,lambda u:model.control_embed(t[:,j]*len(GROUP)+u),train_sign=False)
        else:
            cc,ch,m,K=canonical_controlled(c[:,j],h,True)
            x=local_activation(model,t[:,j],cc,ch,policy,params)
            # Existing original step and projection; no world mark as learned input.
            z=model.history._execute(x,ch,m,K,h,False,cc)
        trace.append(z);h=z.history
    return model.consume(trace[-1]),trace


def measured(a,b):
    if a.shape!=b.shape or not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):raise ValueError('nonfinite or unequal shape')
    return dict(max_abs=float((a.double()-b.double()).abs().max()),numerically_equal=bool(torch.allclose(a,b,rtol=2e-5,atol=2e-6)) if a.is_floating_point() else bool(torch.equal(a,b)))


@torch.no_grad()
def covariance(model,rows,seed_mode,policy,E,B,params):
    """Actual conjugation square vs intentionally fixed control comparison.
    Fixed-context comparisons outside the centralizer are NOT equivariance tests.
    """
    p=next(model.parameters());out={}
    for rs in groups(rows):
        s,o,_=batch(rs,'signed',p.device,p.dtype);c,t=literal_controls(o)
        y,tr=rollout(model,s,c,t,seed_mode,policy,E,B,params)
        for name,g in GENERATORS.items():
            cg=conjugate_control(g,c);x=transformed_seed(g,s)
            z,tt=rollout(model,x,cg,t,seed_mode,policy,E,B,params)
            fixed,ft=rollout(model,x,c,t,seed_mode,policy,E,B,params)
            rec=out.setdefault(name,dict(rows=0,commuting_rows=0,conjugated={},fixed={}))
            rec['rows']+=len(s);rec['commuting_rows']+=int((cg==c).all(1).sum())
            def add(scope,k,a,b):
                r=measured(a,b);v=rec[scope].setdefault(k,dict(max_abs=0.,all_close=True))
                v['max_abs']=max(v['max_abs'],r['max_abs']);v['all_close']&=r['numerically_equal']
            for scope,trace,logits in (('conjugated',tt,z),('fixed',ft,fixed)):
                for a,b in zip(tr,trace):
                    add(scope,'history',act_tensor(g,a.history,True),b.history)
                    add(scope,'output',act_tensor(g,a.output,True),b.output)
                    add(scope,'ids',a.ids,b.ids);add(scope,'local_signs',a.local_signs,b.local_signs);add(scope,'weights',a.weights,b.weights)
                for j,(a,b) in enumerate(zip(y,logits)):
                    add(scope,'logits'+str(j),a[:,getattr(model,f'label_perm_{j}')[GROUP.index(g)]],b)
    return out


@torch.no_grad()
def capture_arm(model,rows,seed_mode,policy,E,B,params,batch_size,seconds):
    p=next(model.parameters());end=time.monotonic()+seconds;out=[]
    for rs in groups(rows):
        for start in range(0,len(rs),batch_size):
            if time.monotonic()>=end:raise TimeoutError('arm capture budget')
            r=rs[start:start+batch_size];s,o,y=batch(r,'signed',p.device,p.dtype);c,t=literal_controls(o)
            _,tr=rollout(model,s,c,t,seed_mode,policy,E,B,params);z=tr[-1]
            f=torch.cat((z.local_output.flatten(1),z.local_history.flatten(1)),1)
            out.append((r,f.cpu(),z.stabilizer.cpu(),z.world_to_local.cpu(),y.cpu()))
    return out


def unpack(parts,split):
    r=[];F=[];K=[];M=[];Y=[]
    for rows,f,k,m,y in parts:
        if rows[0]['split']!=split:continue
        r.extend(rows);F.append(f);K.append(k);M.append(m);Y.append(y)
    return r,torch.cat(F),torch.cat(K),torch.cat(M),torch.cat(Y)


def leverage(X,H,ridge):
    """Measure train-span and ridge leverage on heldout features, not fit."""
    _,s,Vh=torch.linalg.svd(X,full_matrices=(len(X)<X.shape[1]))
    s=torch.nn.functional.pad(s,(0,X.shape[1]-len(s)));q=H@Vh.T
    rank=int((s>s.max()*max(X.shape)*torch.finfo(X.dtype).eps).sum())
    lev=(q.square()/(s.square()+ridge)).sum(1)
    return dict(rank=rank,train_features=X.shape[1],heldout_rows=len(H),
        outside_train_span_rms=float(q[:,rank:].square().sum(1).mean().sqrt()),
        heldout_ridge_leverage_mean=float(lev.mean()),heldout_ridge_leverage_max=float(lev.max()))


@torch.no_grad()
def score_captured(model,parts,coefficients):
    """Use actual original native affine consumer on captured terminal fields."""
    p=next(model.parameters());w=model.width;out={}
    before={k:v.detach().clone() for k,v in model.readout.state_dict().items()}
    if coefficients is not None:
        B=torch.cat(coefficients,1).to(p.device,p.dtype)
        model.readout.weight.copy_(B[:-1].T);model.readout.bias.copy_(B[-1])
    try:
        for rows,f,k,m,y in parts:
            x=f.to(p.device,p.dtype);target=y.to(p.device);cut=2*w
            z=SimpleNamespace(local_output=x[:,:cut].reshape(-1,2,w),
                local_history=x[:,cut:].reshape(-1,model.experts,2,w),stabilizer=k.to(p.device),world_to_local=m.to(p.device))
            pred=model.consume(z);stat=out.setdefault(rows[0]['split'],dict(rows=0,correct=[0]*3,ties=[0]*3,sse=[0.]*3))
            stat['rows']+=len(rows)
            for j,v in enumerate(pred):
                if not bool(torch.isfinite(v).all()):raise ValueError('nonfinite score')
                top=v==v.amax(1,keepdim=True);stat['correct'][j]+=int((top.gather(1,target[:,j,None]).flatten()&(top.sum(1)==1)).sum())
                stat['ties'][j]+=int((top.sum(1)>1).sum());stat['sse'][j]+=float((v.double()-torch.nn.functional.one_hot(target[:,j],v.shape[1]).double()).square().sum())
    finally:model.readout.load_state_dict(before,strict=True)
    return out


def arm_fit(model,parts,ridge):
    train,F,K,M,T=unpack(parts,'train');held,H,HK,HM,HT=unpack(parts,'composition')
    X=features(F.double());HX=features(H.double());fit=[];solutions=[];off=0
    for j,n in enumerate(HEADS):
        tab=getattr(model,f'label_perm_{j}').cpu()
        Y=torch.nn.functional.one_hot(T[:,j],n).double().gather(1,tab[M])
        HY=torch.nn.functional.one_hot(HT[:,j],n).double().gather(1,tab[HM])
        B0=torch.cat((model.readout.weight[off:off+n].detach().cpu().double().T,model.readout.bias[off:off+n].detach().cpu().double()[None]),0)
        B,r=solve_direct(X,Y,K,tab,B0,ridge);off+=n;solutions.append(B)
        r['train_collisions']=collision_bounds(X,Y,K,tab,train)
        r['heldout_collisions']=collision_bounds(HX,HY,HK,tab,held)
        r['joint_collisions']=collision_bounds(torch.cat((X,HX)),torch.cat((Y,HY)),torch.cat((K,HK)),tab,train+held)
        fit.append(r)
    base=score_captured(model,parts,None)
    good=all(r['converged'] and r['regularized_objective']<=r['original_sse']+1e-8*max(1.,r['original_sse']) for r in fit)
    repaired=score_captured(model,parts,solutions) if good else None
    admitted=good and all(a<=b for a,b in zip(repaired['train']['sse'],base['train']['sse']))
    return dict(status='train_admitted' if admitted else 'readout_not_admitted',fit=fit,baseline=base,repaired=repaired,
        leverage=leverage(X,HX,ridge),train_task_sha256=task_digest(train),heldout_task_sha256=task_digest(held),
        readout_coefficients=[v.tolist() for v in solutions] if admitted else None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('checkpoint','translation-receipt','donor','phase-manifest','phase-receipt','output'):
        p.add_argument('--'+k,type=Path,required=True)
    for k in ('sha256','translation-receipt-sha256','donor-sha256','phase-manifest-sha256','source-head'):
        p.add_argument('--'+k,required=True)
    p.add_argument('--guards',type=int,nargs=4,default=(1,2,1,1));p.add_argument('--contexts',type=int,default=5);p.add_argument('--heldout-contexts',type=int,default=2)
    p.add_argument('--batch-size',type=int,default=32);p.add_argument('--seconds',type=float,default=90)
    p.add_argument('--ridge',type=float,default=.001);p.add_argument('--control-ridge',type=float,default=.01)
    p.add_argument('--arm',choices=[s+':'+c for s in SEEDS for c in CONTROLS],nargs='+')
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    guard_controls(a.guards)
    if min(a.contexts,a.heldout_contexts,a.batch_size)<1 or not 0<a.seconds<=90 or any(not math.isfinite(x) or x<=0 for x in (a.ridge,a.control_ridge)):raise ValueError('scope')
    arms=a.arm or [s+':'+c for s in SEEDS for c in CONTROLS]
    if len(arms)!=len(set(arms)) or a.output.exists():raise ValueError('duplicate arms or existing output')
    src=source_identity(root,a.source_head);data=a.translation_receipt.read_bytes()
    if sha256(data).hexdigest()!=a.translation_receipt_sha256:raise ValueError('translation receipt digest')
    parent=json.loads(data);prior=parent['source']
    subprocess.run(['git','-C',str(root),'merge-base','--is-ancestor',prior['head'],src['head']],check=True)
    if any(src['files'].get(k)!=v for k,v in prior['files'].items()):raise ValueError('parent source changed')
    if parent['historical_sha256']!=a.donor_sha256 or parent.get('phase_manifest_sha256')!=a.phase_manifest_sha256 or sha256(a.phase_receipt.read_bytes()).hexdigest()!=parent.get('phase_receipt_sha256'):raise ValueError('wrong donor identity')
    if not torch.cuda.is_available() or not torch.version.hip or not getattr(torch.cuda.get_device_properties(0),'gcnArchName','').startswith('gfx803'):raise RuntimeError('native gfx803 HIP required')
    dev=torch.device('cuda',0);cfg=parent['config'];dt=getattr(torch,cfg['dtype'])
    runtime=dict(torch=str(torch.__version__),hip=torch.version.hip,arch=torch.cuda.get_device_properties(dev).gcnArchName)
    model=SourceGeometryOrderedLearner(cfg['width'],cfg['experts'],expert_width=cfg['expert_width']).to(dev,dt).eval()
    model.load_state_dict(read_checkpoint(a.checkpoint,a.sha256,dt),strict=True);model.history.source.history_cap=cfg['cap']
    frozen=model_digest(model);execution=execution_contract(model)
    translation_provenance(a.translation_receipt,a.translation_receipt_sha256,a.sha256,prior,cfg,frozen,execution,runtime)
    if 'readout_repair' in parent or not parent['historical_metadata']['joint_mean']:raise ValueError('use original joint donor translation')
    a.output.mkdir(parents=True,exist_ok=False)
    def save(name,obj):exclusive(a.output/name,(json.dumps(obj,indent=2,allow_nan=False)+'\n').encode())
    split=exact_seed_split(parent['seed'],cfg['experts'],cfg['width'])
    E=seed_matrix(split['equivariant_basis'],dev,dt);B=seed_matrix(parent['seed']['historical_basis'],dev,dt)
    allrows=list(joint_records());domain=[r for r in allrows if r['guards']==tuple(a.guards)]
    rows=[r for r in select_contexts(domain,a.contexts) if r['split']=='train']+[r for r in select_contexts(domain,a.heldout_contexts) if r['split']=='composition']
    # Diagnostic witnesses contain zero/DC/phase and reflected noncommuting words.
    probe=[r for r in rows if r['mean'] in ((0,0,0),(1,0,0)) and r['residual'] in ((0,0,0),(1,-1,0),(1,1,1))]
    meta=dict(kind='x369_frame_control_weld',source=src,runtime=runtime,config=cfg,checkpoint_sha256=a.sha256,
        parent_translation_sha256=a.translation_receipt_sha256,donor_sha256=a.donor_sha256,seed_split=split,arms=arms,
        task_sha256=task_digest(rows),covariance_task_sha256=task_digest(probe),fit_guards=FIT_GUARDS,
        training_steps=0,heldout_used_for_selection=False,release_admitted=False,ordinary_checkpoint_export=False)
    save('manifest.json',meta);params={}
    if any(x.endswith(':conditioned') for x in arms):
        donor_args=SimpleNamespace(checkpoint=a.donor,sha256=a.donor_sha256,phase_manifest=a.phase_manifest,
            phase_manifest_sha256=a.phase_manifest_sha256,phase_receipt=a.phase_receipt)
        old,_,_,_=restore(donor_args,root,dev);od=model_digest(old)
        donor_rows=[r for r in allrows if r['split']=='train' and r['guards'] in FIT_GUARDS]
        try:params,fit=capture_donor(old,model,donor_rows,a.batch_size,a.seconds,a.control_ridge)
        except TimeoutError as e:
            save('controller.json',dict(status='time_budget',error=str(e)));fit=None
        if model_digest(old)!=od:raise RuntimeError('historical model changed')
        del old
        if fit is not None:save('controller.json',dict(status='complete',**fit))
    for arm in arms:
        s,c=arm.split(':');name=arm.replace(':','_')+'.json'
        if c=='conditioned' and not params:save(name,dict(status='controller_unavailable',arm=arm));continue
        try:
            parts=capture_arm(model,rows,s,c,E,B,params,a.batch_size,a.seconds)
            result=arm_fit(model,parts,a.ridge)
            result['covariance']=covariance(model,probe,s,c,E,B,params)
        except TimeoutError as e:result=dict(status='time_budget',error=str(e))
        if model_digest(model)!=frozen or execution_contract(model)!=execution or source_identity(root,a.source_head)!=src:raise RuntimeError('frozen source changed')
        result.update(arm=arm,training_steps=0,heldout_used_for_selection=False,release_admitted=False)
        save(name,result);print(json.dumps(dict(arm=arm,status=result['status'])),flush=True)


if __name__=='__main__':main()
