"""Translate existing trained paired FairLearner states, never train or migrate optimizer.
Original Leech-LILA: Anatolii Kornienko; DASHI extension.
Exact marked historical path and projected controlled-source candidate are distinct.
Native execution only; no F5. Public task only.
"""
import argparse,io,json,os,time,copy
from pathlib import Path
from collections import defaultdict
from fractions import Fraction
from hashlib import sha256
import torch
from torch import nn
from models.geometric_fair_residual import FairResidualLinear,fair_arm
from models.geometric_orbit_descent import canonical_controlled,apply_group_indices,decode_frame,act_tensor
from models.geometric_reflection_fibre import GROUP,mul,inv
from models.geometric_stabilizer_codec import same_bits,encode_history,decode_history
from scripts.x369_source_geometry import SourceGeometryOrderedLearner,literal_controls,signed_seed
from scripts.x369_ordered_checkpoint import fingerprint,load_checkpoint,trace_probe
from scripts.x369_source_acceptance import source_identity,model_digest,execution_contract,choose
from scripts.train_x369_ordered_action import batch,records
from scripts.audit_x369_joint_action import joint_records,task_digest,select_contexts,fit_joint,MEANS
from scripts.audit_x369_ordered_signed_task import TRIADS
from scripts.run_x369_fair_ordered_gpu import FairLearner


def canonical_index(i):
    u=GROUP[i]
    return max(GROUP.index(mul(mul(g,u),inv(g))) for g in GROUP)


def effective_dense(model):
    """Use the original stored eval expression on the original device/dtype."""
    if model.training:raise ValueError('eval only')
    dense=copy.deepcopy(model);folded=[]
    with torch.no_grad():
        for name,layer in list(dense.named_modules()):
            if type(layer) is not FairResidualLinear:continue
            if layer.training:raise ValueError('eval residual required')
            w=layer.base+layer.delta()
            if not torch.isfinite(w).all():raise ValueError('nonfinite effective weight')
            lin=nn.Linear(w.shape[1],w.shape[0],bias=layer.bias is not None,device=w.device,dtype=w.dtype)
            lin.weight.copy_(w)
            if layer.bias is not None:lin.bias.copy_(layer.bias)
            lin.eval();parent,key=name.rsplit('.',1);setattr(dense.get_submodule(parent),key,lin)
            folded.append(dict(path=name,weight_shape=list(w.shape),weight_sha256=sha256(w.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),ternary=layer.ternary,scale=layer.scale))
    return dense.eval(),folded


def seed_weld(old,new):
    """Exact dyadic six-coordinate seed map and partial inverse on its image.
    Not a global nonlinear conjugacy; signed group action tested independently.
    """
    import sympy as S
    from scripts.x369_common_hom import exact,target_action
    p=next(old.parameters());basis=torch.eye(6,device=p.device,dtype=p.dtype)
    seeds=torch.cat((basis,torch.zeros_like(basis[:,:3])),1)
    with torch.no_grad():
        a=signed_seed(seeds,new.experts,new.width)
        b=old.initial(seeds)
    A=exact(a.flatten(1));B=exact(b.flatten(1));L=A.T*(A*A.T).inv()
    assert A*L==S.eye(6)
    rank=B.rank();reverse=B.T*(B*B.T).inv() if rank==6 else None
    if reverse is not None:assert B*reverse==S.eye(6)
    mat=lambda x:[[str(v) for v in row] for row in x.tolist()]
    T=target_action();res={}
    for k,g in (('R',GROUP[8]),('J',GROUP[10]),('N',GROUP[1])):
        D=exact(act_tensor(g,b,True).flatten(1))-T[k]*B
        res[k]=dict(rank=D.rank(),first_nonzero=next(([i,j,str(D[i,j])] for i in range(D.rows) for j in range(D.cols) if D[i,j]),None))
    return dict(rank=rank,constructed_basis=mat(A),historical_basis=mat(B),constructed_coordinates=mat(L),
                historical_coordinates=None if reverse is None else mat(reverse),bidirectional_on_seed_image=reverse is not None,
                standard_action_defects=res,global_nonlinear_conjugacy_certified=False)


def project_controls(observations):
    """Exact Euclidean LS by canonical literal key. Preserve first collision.
    Observations are (tag, canonical index, desired polar local vector, witness).
    """
    groups=defaultdict(list)
    for t,c,v,w in observations:groups[t,c].append((tuple(Fraction(float(x)) for x in v),w))
    means={};report=[]
    for (t,c),xs in sorted(groups.items()):
        if not xs or any(len(v)!=len(xs[0][0]) for v,_ in xs):raise ValueError('control width')
        n=len(xs);mean=tuple(sum((v[j] for v,_ in xs),Fraction(0))/n for j in range(len(xs[0][0])))
        means[t,c]=mean
        diff=next(((v,w) for v,w in xs[1:] if v!=xs[0][0]),None)
        err=sum((sum((a-b)**2 for a,b in zip(v,mean)) for v,_ in xs),Fraction(0))
        report.append(dict(tag=t,canonical=c,count=n,mean=list(map(str,mean)),squared_error=str(err),exact=diff is None,
          collision=None if diff is None else dict(left=xs[0][1],right=diff[1],left_value=list(map(str,xs[0][0])),right_value=list(map(str,diff[0])))))
    return means,report


@torch.no_grad()
def translate_controls(old,new,means):
    """Ten original legal literals, original training seed domain only.
    Project transformed old embeddings onto the actual canonical control keys.
    No heldout endpoint, target prediction or new-source optimizer.
    """
    p=next(old.parameters());obs=[];raw=[];fallback=defaultdict(list)
    for t in range(4):
        for v in range(3 if t<2 else 2):
            op=tuple(int(t==j) for j in range(4))+tuple(int(v==j) for j in range(3))
            o=torch.tensor([[op]],device=p.device,dtype=p.dtype)
            u,tag=literal_controls(o);c=canonical_index(int(u.item()))
            x=old.embed(o[:,0]);raw.append(dict(tag=t,guard=v,control=int(u.item()),canonical=c,activation=x[0].tolist()))
            fallback[t].append(x[0].detach())
            pairs=[(m,d) for m in means for d in TRIADS]
            s=torch.tensor([m+d+(0,0,0) for m,d in pairs],device=p.device,dtype=p.dtype)
            h=old.initial(s);cc,_,mark,_=canonical_controlled(u.flatten().expand(len(s)),h,True)
            if not bool((cc==c).all()):raise RuntimeError('canonical class mismatch')
            z=apply_group_indices(x.expand(len(s),-1),mark).tolist()
            for (m,d),g,y in zip(pairs,mark.tolist(),z):
                obs.append((t,c,y,dict(mean=m,residual=d,guard=v,mark=g)))
    table,report=project_controls(obs);unsupported=[]
    for t in range(4):
        default=torch.stack(fallback[t]).double().mean(0).to(p.dtype)
        for i in range(len(GROUP)):
            c=canonical_index(i)
            if (t,c) in table:
                v=torch.tensor([float(x) for x in table[t,c]],device=p.device,dtype=p.dtype)
            else:
                v=default;unsupported.append([t,i])
            new.control_embed.weight[t*len(GROUP)+i].copy_(v)
    return dict(raw_literals=raw,projected_keys=report,unsupported_table_slots=unsupported,
                scope='ten legal training one-step literals over original seed domain; no heldout selection',
                full_polar_embedding_equivariant=False,
                full_polar_obstruction='central sign commutes with every control; a mark-independent polar embedding invariant under it must vanish')


@torch.no_grad()
def build_candidate(old,dense,means):
    p=next(old.parameters());src=dense.history.source
    if (not old.paired or src.route_geometry!='c2_c3_residual' or src.top_k!=2 or src.temperature!=1 or
        src.router_selection!='topk' or src.transition_rank is not None or getattr(src,'route_probe_action',None) is not None):
        raise ValueError('unsupported source recurrence configuration')
    width=src.experts[0][0].out_features
    new=SourceGeometryOrderedLearner(old.width,old.experts,expert_width=width).to(p.device,p.dtype).eval()
    new.history.source.load_state_dict(src.state_dict(),strict=True)
    new.history.source.history_cap=src.history_cap
    new.readout.load_state_dict(dense.readout.state_dict(),strict=True)
    for a,b in zip(src.state_dict().values(),new.history.source.state_dict().values()):
        if not same_bits(a,b):raise RuntimeError('recurrent materialization copy differs')
    return new,seed_weld(old,new),translate_controls(old,new,means)


class Metrics:
    def __init__(self):self.fields={}
    def add(self,name,a,b,rows):
        if a.shape!=b.shape or a.dtype!=b.dtype or not torch.isfinite(a).all() or not torch.isfinite(b).all():raise ValueError('bad comparison '+name)
        if len(a)!=len(rows):raise ValueError('row mismatch')
        eq=(a.contiguous().view(torch.int32 if a.dtype==torch.float32 else torch.int64)==b.contiguous().view(torch.int32 if b.dtype==torch.float32 else torch.int64)) if a.is_floating_point() else a==b
        eq=eq.flatten(1).all(1);error=(a.double()-b.double()).abs().flatten(1).amax(1)
        v=self.fields.setdefault(name,dict(rows=0,bitwise_rows=0,max_abs=0.,first_difference=None))
        v['rows']+=len(rows);v['bitwise_rows']+=int(eq.sum());v['max_abs']=max(v['max_abs'],float(error.max()))
        if v['first_difference'] is None and not bool(eq.all()):
            i=int((~eq).nonzero()[0]);v['first_difference']=dict(row=rows[i],max_abs=float(error[i]))
    def pair(self,name,a,b,rows):
        for k in ('branches','history','ids','signs','weights'):self.add(name+'.'+k,getattr(a,k),getattr(b,k),rows)


@torch.no_grad()
def compare(old,dense,new,rows,*,batch_size=32,seconds=90):
    """Matched natural paths and same-history counterfactuals.
    Stop on complete batch boundaries. No positive beyond observed scope.
    """
    p=next(old.parameters());end=time.monotonic()+seconds;metrics=Metrics();groups=defaultdict(list)
    for i,r in enumerate(rows):groups[r['split'],len(r['word'])].append(i)
    hist=[{},{}];scores={};done=0;marked_ok=True;fold_ok=True
    for (split,length),ix in sorted(groups.items()):
        for start in range(0,len(ix),batch_size):
            if time.monotonic()>=end:break
            ids=ix[start:start+batch_size];rs=[rows[i] for i in ids];s,o,target=batch(rs,'signed',p.device,p.dtype)
            controls,tags=literal_controls(o)
            h=old.initial(s);d=dense.initial(s);n=signed_seed(s,new.experts,new.width)
            metrics.add('seed.same_coordinates',h,n,rs)
            for t in range(length):
                u=controls[:,t];tag=tags[:,t];x=old.embed(o[:,t]);rs0=[dict(r,prefix=r['word'][:t+1]) for r in rs]
                a=old.history.step(x,h,train_sign=False);b=dense.history.step(dense.embed(o[:,t]),d,train_sign=False)
                metrics.pair('materialized_natural',a,b,rs0)
                fold_ok=fold_ok and all(same_bits(getattr(a,k),getattr(b,k)) if getattr(a,k).is_floating_point() else torch.equal(getattr(a,k),getattr(b,k)) for k in ('branches','history','ids','signs','weights'))
                cc,ch,mark,K=canonical_controlled(u,h,True)
                # Lossless historical wrapper: canonical packet + mark, reopen
                # world before executing original source. No equivariance assumption.
                reopened=decode_frame(decode_history(encode_history(ch)),mark,True)
                if not same_bits(reopened,h):raise RuntimeError('marked history lost bytes')
                m=old.history.step(x,reopened,train_sign=False);metrics.pair('marked_original',a,m,rs0)
                marked_ok=marked_ok and all(same_bits(getattr(a,k),getattr(m,k)) if getattr(a,k).is_floating_point() else torch.equal(getattr(a,k),getattr(m,k)) for k in ('branches','history','ids','signs','weights'))
                # Same historical history: isolate frame, control projection,
                # stabilizer projection and classifier from seed/path drift.
                lx=apply_group_indices(x,mark);v=dense.history.step(lx,ch,train_sign=False)
                shadow=new.history.step_controlled(u,h,lambda c:new.control_embed(tag*len(GROUP)+c),train_sign=False)
                if not torch.equal(shadow.world_to_local,mark):raise RuntimeError('frame mismatch')
                metrics.add('frame.output',a.branches,decode_frame(v.branches,mark,True),rs0)
                metrics.add('frame.history',a.history,decode_frame(v.history,mark,True),rs0)
                metrics.add('frame.ids',a.ids,v.ids,rs0)
                metrics.add('control.embedding',lx,new.control_embed(tag*len(GROUP)+cc),rs0)
                metrics.add('control.local_output',v.branches,shadow.raw_local_output,rs0)
                metrics.add('control.local_history',v.history,shadow.raw_local_history,rs0)
                metrics.add('control.ids',v.ids,shadow.ids,rs0)
                metrics.add('projection.output',decode_frame(shadow.raw_local_output,mark,True),shadow.output,rs0)
                metrics.add('projection.history',decode_frame(shadow.raw_local_history,mark,True),shadow.history,rs0)
                z=new.history.step_controlled(u,n,lambda c:new.control_embed(tag*len(GROUP)+c),train_sign=False)
                metrics.add('natural.history',a.history,z.history,rs0);metrics.add('natural.output',a.branches,z.output,rs0)
                h,d,n=a.history,b.history,z.history
            oldp=old.consume(a.branches,h);densep=dense.consume(b.branches,d);newp=new.consume(z)
            world=dense.consume(shadow.output,shadow.history);local=new.consume(shadow)
            for j,(v,w,k,f,g) in enumerate(zip(oldp,densep,newp,world,local)):
                metrics.add(f'materialized.logits{j}',v,w,rs);metrics.add(f'natural.logits{j}',v,k,rs)
                metrics.add(f'readout.projection{j}',f,g,rs);fold_ok=fold_ok and same_bits(v,w)
                for name,logits in (('historical',v),('materialized',w),('translated',k)):
                    stat=scores.setdefault(split,{}).setdefault(name,dict(rows=0,correct=[0]*3,ties=[0]*3))
                    if j==0:stat['rows']+=len(rs)
                    top=logits==logits.amax(-1,keepdim=True);stat['correct'][j]+=int((top.gather(1,target[:,j,None]).squeeze(1)&(top.sum(-1)==1)).sum());stat['ties'][j]+=int((top.sum(-1)>1).sum())
            for i,v,w in zip(ids,h.flatten(1).cpu(),n.flatten(1).cpu()):hist[0][i]=v;hist[1][i]=w
            done+=len(rs)
        if time.monotonic()>=end:break
    return dict(status='complete' if done==len(rows) else 'time_budget',rows=done,requested=len(rows),fields=metrics.fields,
                task_scores=scores,materialized_bitwise_on_sample=fold_ok and done>0,marked_bitwise_on_sample=marked_ok and done>0,
                universal_full_function_certified=False),hist


def hom_comparison(rows,hist):
    from scripts.x369_common_hom import fitted_receipt
    out={}
    for name,H in zip(('historical','translated'),hist):
        if len(H)!=len(rows):out[name]=dict(status='incomplete');continue
        f=fit_joint(rows,torch.stack([H[i] for i in range(len(rows))]))
        out[name]=dict(fit={k:v for k,v in f.items() if k not in ('V','q','A','targets')})
        if f['admitted']:out[name]['common_hom']=fitted_receipt(f)
    return out


def restore(a,root,device):
    """Use existing original final-state or phase-snapshot authority, never loose keys."""
    dt=torch.float32
    runtime=dict(torch=str(torch.__version__),hip=torch.version.hip,arch=torch.cuda.get_device_properties(device).gcnArchName)
    if a.phase_manifest:
        from scripts.run_x369_phase_trajectory_gpu import PHASE_SOURCES,reopen
        manifest_bytes=a.phase_manifest.read_bytes()
        if sha256(manifest_bytes).hexdigest()!=a.phase_manifest_sha256:raise ValueError('phase manifest digest mismatch')
        manifest=json.loads(manifest_bytes);r=json.loads(a.phase_receipt.read_bytes())
        identity=r['identity']
        if (manifest['kind']!='x369_phase_trajectory' or manifest['sources']!=fingerprint(root,PHASE_SOURCES) or manifest['runtime']!=runtime or
            identity['manifest_sha256']!=a.phase_manifest_sha256 or identity['settings']!=manifest['settings'] or identity['arm'] not in manifest['arms'] or
            r['sha256']!=a.sha256 or r['step'] not in manifest['settings']['schedule'] or r['step']<=0):
            raise ValueError('phase identity or source mismatch')
        q=reopen(a.checkpoint,r,identity);m=dict(identity['arm'],width=identity['settings']['width'],experts=4,joint_mean=True,steps=r['step'])
        rows=list(joint_records());state=q['model'];probe=r['probe'];expected_task=manifest['task_sha256']
    else:
        q=load_checkpoint(a.checkpoint,a.sha256,fingerprint(root));m=q['metadata'];state=q['state'];probe=m['trace_probe'];expected_task=m['task_sha256']
        if (not m.get('fixed_horizon_completed') or not m.get('public_task_only') or m.get('dtype')!='float32' or m.get('device')!='cuda' or
            any(m.get(k)!=v for k,v in runtime.items()) or m.get('steps',0)<=0):raise ValueError('original final-state provenance mismatch')
        rows=list(joint_records() if m['joint_mean'] else records())
    if m['paired'] is not True:raise ValueError('paired historical source required; no implicit unpaired lifting')
    if task_digest(rows)!=expected_task:raise ValueError('original task digest mismatch')
    if not all(isinstance(v,torch.Tensor) and v.layout==torch.strided and torch.isfinite(v).all() and (not v.is_floating_point() or v.dtype==dt) for v in state.values()):
        raise ValueError('original tensor dtype/layout mismatch')
    old=FairLearner(m['width'],m['experts'],True,'dense').to(device,dt)
    old.history.source=fair_arm(old.history.source,m['mode'],m['null'],m['ternary'],m['seed'])
    old.load_state_dict(state,strict=True);old.eval()
    if trace_probe(old,rows,device,dt)!=probe:raise RuntimeError('original stored trace mismatch')
    return old,m,rows,probe


def exclusive(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--sha256',required=True);p.add_argument('--source-head',required=True)
    p.add_argument('--output',type=Path,required=True,help='new raw translated candidate .pt');p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--phase-manifest',type=Path);p.add_argument('--phase-manifest-sha256');p.add_argument('--phase-receipt',type=Path)
    p.add_argument('--contexts',type=int,default=1);p.add_argument('--batch-size',type=int,default=32);p.add_argument('--seconds',type=float,default=90)
    a=p.parse_args()
    if a.contexts<1 or a.batch_size<1 or not 0<a.seconds<=90 or len({a.output.resolve(),a.receipt.resolve(),a.checkpoint.resolve()})!=3 or a.output.exists() or a.receipt.exists():p.error('invalid scope or existing output')
    if len({bool(a.phase_manifest),bool(a.phase_manifest_sha256),bool(a.phase_receipt)})!=1:p.error('phase inputs must be supplied together')
    root=Path(__file__).resolve().parents[1];source=source_identity(root,a.source_head)
    if not torch.cuda.is_available() or not torch.version.hip or not getattr(torch.cuda.get_device_properties(0),'gcnArchName','').startswith('gfx803'):raise RuntimeError('native gfx803 HIP required')
    device=torch.device('cuda',0);old,m,rows,probe=restore(a,root,device);before=model_digest(old)
    dense,folded=effective_dense(old);fold_probe=trace_probe(dense,rows,device,torch.float32)
    if fold_probe!=probe:raise RuntimeError('materialized original trace differs; retain historical computation, no translated export')
    new,seed,control=build_candidate(old,dense,MEANS if m['joint_mean'] else ((1,0,0),))
    frozen=model_digest(new);execution=execution_contract(new)
    # Full joint contexts are independent public diagnostics; for a fixed-mean
    # donor this is out-of-distribution on the extra mean states.
    sample=select_contexts(list(joint_records()),a.contexts)
    comparison,hist=compare(old,dense,new,sample,batch_size=a.batch_size,seconds=a.seconds)
    result=dict(kind='x369_translated_source_candidate',historical_sha256=a.sha256,historical_metadata=m,
        historical_trace=probe,materialized_trace=fold_probe,folded=folded,source=source,seed=seed,control=control,
        comparison=comparison,common_hom=hom_comparison(sample,hist),probe_task_sha256=task_digest(sample),
        probe_scope='joint public contexts; extra mean states OOD for fixed-mean donor',training_steps=0,
        optimizer_migrated=False,original_new_source_training=False,release_admitted=False,universal_full_function_certified=False,
        candidate_model_sha256=frozen,execution_contract=execution,
        config=dict(width=new.width,experts=new.experts,expert_width=new.expert_width,cap=new.history.source.history_cap,dtype='float32'),
        candidate_sha256=None)
    if model_digest(old)!=before or model_digest(new)!=frozen or execution_contract(new)!=execution or source_identity(root,a.source_head)!=source:raise RuntimeError('source mutated during translation')
    if not comparison['materialized_bitwise_on_sample'] or not comparison['marked_bitwise_on_sample']:
        result['status']='materialized_or_marked_mismatch'
    else:
        buf=io.BytesIO();torch.save({k:v.detach().cpu().clone() for k,v in new.state_dict().items()},buf);data=buf.getvalue()
        result['candidate_sha256']=sha256(data).hexdigest();result['status']='translated_candidate' if comparison['status']=='complete' else 'translated_candidate_bounded_diagnostics'
        exclusive(a.output,data)
    data=(json.dumps(result,indent=2,allow_nan=False)+'\n').encode();exclusive(a.receipt,data)
    print(json.dumps(dict(status=result['status'],candidate_sha256=result['candidate_sha256'],receipt_sha256=sha256(data).hexdigest(),training_steps=0),indent=2))


if __name__=='__main__':main()
