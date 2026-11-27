"""GPU-native matched public ordered-action learning campaign.
Original Leech-LILA: Anatolii Kornienko; DASHI extension.
No F5, historical optimizer migration, held-out selection or CPU training.
"""
import argparse,hashlib,json,random,time
from collections import defaultdict
from pathlib import Path
import torch
from torch.nn import functional as F
from scripts.run_x369_phase_trajectory_gpu import snapshot,receipts,checkpoint_paths,schedule
from scripts.run_x369_ordered_campaign import exclusive_json,sha
from scripts.x369_translate_paired_checkpoint import restore,effective_dense,build_candidate,exclusive
from scripts.x369_source_acceptance import source_identity,model_digest,execution_contract
from scripts.x369_source_geometry import SourceGeometryOrderedLearner,literal_controls
from scripts.x369_constructive_finite_source import decode_history
from scripts.x369_source_geometry import known_action_register
from scripts.run_x369_fair_ordered_gpu import FairLearner
from models.geometric_fair_residual import fair_arm
from scripts.audit_x369_joint_action import joint_records,task_digest,select_contexts,MEANS
from scripts.audit_x369_ordered_signed_task import endpoint
from scripts.train_x369_ordered_action import batch
from models.geometric_reflection_fibre import GROUP

ARMS=('fresh','warm','historical')
RELATIONS=(('aaa',''),('jj',''),('nn',''),('jaj','aa'),('na','an'),('nj','jn'))

def groups(rows):
    result=defaultdict(list)
    for r in rows:result[len(r['word'])].append(r)
    return dict(sorted(result.items()))

def training_batch(train,rng,n):
    rs=rng.choice(list(train.values()))
    return rng.choices(rs,k=n)

def setup(arm,seed,old,meta,device):
    torch.manual_seed(seed)
    if arm=='fresh':
        model=SourceGeometryOrderedLearner(old.width,old.experts,expert_width=old.history.source.experts[0][0].out_features).to(device,torch.float32)
        model.history.source.history_cap=old.history.source.history_cap
        return model
    if arm=='warm':
        dense,_=effective_dense(old)
        model,_,_=build_candidate(old,dense,MEANS if meta['joint_mean'] else ((1,0,0),),'representative')
        return model
    model=FairLearner(old.width,old.experts,True,'dense').to(device,torch.float32)
    model.history.source=fair_arm(model.history.source,meta['mode'],meta['null'],meta['ternary'],seed)
    model.history.source.history_cap=old.history.source.history_cap
    return model

def state_observation(model,s,o):
    if isinstance(model,SourceGeometryOrderedLearner):
        c,t=literal_controls(o);p,trace=model.rollout_group(s,c,t)
        return p,trace[-1].history
    y,h=model.rollout(s,o)
    return model.consume(y,h),h

@torch.no_grad()
def probe(model,rows,device):
    model.eval()
    h=hashlib.sha256()
    for length,rs in groups(rows).items():
        sample=[rs[j*(len(rs)-1)//7] for j in range(8)]
        s,o,_=batch(sample,'signed',device,torch.float32)
        logits,state=state_observation(model,s,o)
        for v in (*logits,state):
            a=v.detach().cpu().contiguous()
            if not torch.isfinite(a).all():raise RuntimeError('nonfinite trace')
            h.update(str((a.dtype,tuple(a.shape))).encode()+a.numpy().tobytes())
    return h.hexdigest()

@torch.no_grad()
def metrics(model,rows,device):
    model.eval()
    total=0;correct=[0]*3;ties=[0]*3;ce=[0.]*3;semantic_max=0.;semantic_sse=0.
    typed=isinstance(model,SourceGeometryOrderedLearner)
    for rs in groups(rows).values():
        for start in range(0,len(rs),128):
            chunk=rs[start:start+128]
            s,o,y=batch(chunk,'signed',device,torch.float32)
            p,h=state_observation(model,s,o)
            for i,v in enumerate(p):
                if not torch.isfinite(v).all():raise RuntimeError('nonfinite evaluation')
                top=v==v.amax(-1,keepdim=True);unique=top.sum(-1)==1
                correct[i]+=int((unique&top.gather(1,y[:,i,None]).squeeze(1)).sum())
                ties[i]+=int((~unique).sum())
                ce[i]+=float(F.cross_entropy(v,y[:,i],reduction='sum'))
            if typed:
                actual=torch.cat(decode_history(h),-1)
                want=torch.tensor([r['mean_label']+r['signed_label'] for r in chunk],device=device,dtype=actual.dtype)
                err=actual-want;semantic_max=max(semantic_max,float(err.abs().max()));semantic_sse+=float(err.square().sum())
            total+=len(chunk)
    return dict(rows=total,unique_correct=correct,ties=ties,ce=[v/total for v in ce],semantic_max=semantic_max if typed else None,semantic_rmse=(semantic_sse/(6*total))**.5 if typed else None)

@torch.no_grad()
def relation_audit(model,device):
    from scripts.x369_source_geometry import signed_seed
    from scripts.train_x369_ordered_action import encode_inputs
    from scripts.audit_x369_ordered_signed_task import TRIADS
    from itertools import product
    model.eval();typed=isinstance(model,SourceGeometryOrderedLearner)
    rows=[dict(mean=m,residual=d,guards=(1,2,1,1)) for m,d in product(MEANS,TRIADS)]
    s=torch.tensor([r['mean']+r['residual']+(0,0,0) for r in rows],device=device,dtype=torch.float32)
    def run(word):
        h=signed_seed(s,model.experts,model.width) if typed else model.initial(s)
        if not word:return h
        ops=torch.tensor([encode_inputs(dict(r,word=word),'signed')[1] for r in rows],device=device,dtype=s.dtype)
        if typed:return model.rollout_group(s,*literal_controls(ops))[1][-1].history
        return model.rollout(s,ops)[1]
    result={}
    for left,right in RELATIONS:
        for r in rows:
            assert endpoint(left,r['guards'],r['mean'],r['residual'])==endpoint(right,r['guards'],r['mean'],r['residual'])
        a,b=run(left),run(right)
        # Compare histories and identical subsequent control predictions.
        op=torch.tensor([encode_inputs(dict(r,word='b'),'signed')[1][0] for r in rows],device=device,dtype=s.dtype)
        def continuation(h):
            if typed:
                c,t=literal_controls(op[:,None,:]);tag=t[:,0]
                out=model.history.step_controlled(c[:,0],h,lambda k:model.control_embed(tag*len(GROUP)+k),train_sign=False)
                return model.consume(out)
            y,n,*_=model.advance(op,h);return model.consume(y,n)
        pa,pb=continuation(a),continuation(b)
        v=(a-b).flatten(1);entry=dict(rows=len(rows),history_max=float(v.abs().max()),history_rmse=float(v.square().mean().sqrt()),history_equal_rows=int((v==0).all(-1).sum()))
        entry['continuation_logit_max']=[float((x-y).abs().max()) for x,y in zip(pa,pb)]
        entry['continuation_decision_equal']=[int((x.argmax(-1)==y.argmax(-1)).sum()) for x,y in zip(pa,pb)]
        if typed:
            err=torch.cat(decode_history(a),-1)-torch.cat(decode_history(b),-1)
            entry['semantic_max']=float(err.abs().max())
        result[left+'='+right]=entry
    return result

def train_arm(d,identity,old,meta,rows,device):
    cfg=identity['settings'];steps=cfg['schedule'];arm=identity['arm'];seed=identity['seed']
    found=receipts(d,steps,identity)
    model=setup(arm,seed,old,meta,device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=.1)
    rng=random.Random(seed);start=0;elapsed=0.;peak=0
    train=groups([r for r in rows if r['split']=='train'])
    probe_rows=[r for r in rows if r['split']=='train']
    def save(t):
        model.eval();p,r=checkpoint_paths(d,t)
        payload=dict(schema=1,kind='x369_geometry_training_snapshot',identity=identity,step=t,probe=probe(model,probe_rows,device),model=model.state_dict(),optimizer=opt.state_dict(),python_rng=rng.getstate(),torch_rng=torch.get_rng_state(),device_rng=torch.cuda.get_rng_state(device),train_seconds=elapsed,peak_allocated_bytes=peak,execution=execution_contract(model) if arm!='historical' else dict(cap=model.history.source.history_cap))
        digest=snapshot(p,payload)
        exclusive_json(r,dict(identity=identity,step=t,sha256=digest,probe=payload['probe']))
        print(json.dumps(dict(arm=arm,seed=seed,step=t,sha256=digest,train_seconds=elapsed)),flush=True)
    if found:
        r=found[-1];start=r['step'];p,_=checkpoint_paths(d,start)
        if sha(p)!=r['sha256']:raise ValueError('snapshot digest')
        q=torch.load(p,map_location='cpu',weights_only=True)
        if q['identity']!=identity or q['step']!=start or q['kind']!='x369_geometry_training_snapshot':raise ValueError('snapshot identity')
        model.load_state_dict(q['model'],strict=True);opt.load_state_dict(q['optimizer'])
        current=execution_contract(model) if arm!='historical' else dict(cap=model.history.source.history_cap)
        if current!=q['execution']:raise RuntimeError('execution contract changed')
        if probe(model,probe_rows,device)!=r['probe'] or q['probe']!=r['probe']:raise RuntimeError('replay mismatch')
        rng.setstate(q['python_rng']);torch.set_rng_state(q['torch_rng']);torch.cuda.set_rng_state(q['device_rng'],device)
        elapsed=q['train_seconds'];peak=q['peak_allocated_bytes'];del q
    else:save(0)
    torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device);begin=time.perf_counter()
    for t in range(start+1,steps[-1]+1):
        model.train()
        rs=training_batch(train,rng,cfg['batch_size']);s,o,y=batch(rs,'signed',device,torch.float32)
        opt.zero_grad(set_to_none=True)
        if cfg['semantic_weight'] and arm!='historical':
            c,tags=literal_controls(o);pred,trace=model.rollout_group(s,c,tags)
            prefix=[]
            for j,out in enumerate(trace):
                m,_,d=known_action_register(s,c[:,:j+1])
                prefix.append(F.mse_loss(torch.cat(decode_history(out.history),-1),torch.cat((m,d),-1)))
            aux=sum(prefix)/len(prefix)
        else:
            pred=model(s,o);aux=0.
        loss=sum(F.cross_entropy(v,y[:,i]) for i,v in enumerate(pred))+cfg['semantic_weight']*aux
        if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise RuntimeError('nonfinite gradient')
        opt.step()
        if t in steps:
            torch.cuda.synchronize(device)
            elapsed+=time.perf_counter()-begin;peak=max(peak,torch.cuda.max_memory_allocated(device))
            save(t)
            torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device);begin=time.perf_counter()
    return model,opt

@torch.no_grad()
def trajectory(model,d,identity,rows,device):
    result=[]
    for r in receipts(d,identity['settings']['schedule'],identity):
        p,_=checkpoint_paths(d,r['step'])
        q=torch.load(p,map_location='cpu',weights_only=True)
        if q['identity']!=identity or q['probe']!=r['probe'] or q['step']!=r['step']:raise ValueError('trajectory identity')
        model.load_state_dict(q['model'],strict=True)
        result.append(dict(step=r['step'],checkpoint_sha256=r['sha256'],train_seconds=q['train_seconds'],peak_train_allocated_bytes=q['peak_allocated_bytes'],train=metrics(model,[x for x in rows if x['split']=='train'],device),composition=metrics(model,[x for x in rows if x['split']=='composition'],device)))
        del q
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','phase-manifest','phase-receipt','output'):
        p.add_argument('--'+name,type=Path,required=name in ('checkpoint','output'))
    for name in ('sha256','source-head','phase-manifest-sha256'):
        p.add_argument('--'+name,required=name!='phase-manifest-sha256')
    p.add_argument('--seeds',type=int,nargs='+',default=[41,42,43])
    p.add_argument('--steps',type=int,default=2000)
    p.add_argument('--interval',type=int,default=250)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--contexts',type=int,default=8)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--semantic-weight',type=float,default=0.,help='optional actual-history prefix supervision, new arms only; separate experiment')
    p.add_argument('--plan',action='store_true')
    a=p.parse_args()
    if len(set(a.seeds))!=len(a.seeds) or min(a.steps,a.interval,a.batch_size,a.contexts)<1 or not 0<a.lr<float('inf') or not 0<=a.semantic_weight<float('inf') or len({bool(a.phase_manifest),bool(a.phase_manifest_sha256),bool(a.phase_receipt)})!=1:p.error('invalid scope')
    if a.plan:
        print(json.dumps(dict(arms=list(ARMS),seeds=a.seeds,steps=a.steps,schedule=schedule(a.steps,a.interval),batch_size=a.batch_size,semantic_weight=a.semantic_weight)));return
    if not torch.cuda.is_available() or not torch.version.hip or not torch.cuda.get_device_properties(0).gcnArchName.startswith('gfx803'):raise RuntimeError('native gfx803 required')
    device=torch.device('cuda',0);root=Path(__file__).resolve().parents[1];source=source_identity(root,a.source_head)
    old,meta,_,_=restore(a,root,device);before=model_digest(old)
    rows=list(joint_records())
    cfg=dict(schedule=schedule(a.steps,a.interval),batch_size=a.batch_size,contexts=a.contexts,lr=a.lr,semantic_weight=a.semantic_weight)
    manifest=dict(kind='x369_geometry_training',source=source,donor_sha256=a.sha256,phase_manifest_sha256=a.phase_manifest_sha256,phase_receipt_sha256=sha(a.phase_receipt) if a.phase_receipt else None,task_sha256=task_digest(rows),settings=cfg,arms=list(ARMS),seeds=a.seeds,runtime=dict(torch=str(torch.__version__),hip=torch.version.hip,arch=torch.cuda.get_device_properties(0).gcnArchName),release_admitted=False)
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);mp=out/'manifest.json'
    if mp.exists():
        if json.loads(mp.read_text())!=manifest:raise ValueError('manifest mismatch')
    else:exclusive_json(mp,manifest)
    sample=select_contexts(rows,a.contexts);summary=[]
    for seed in a.seeds:
        for arm in ARMS:
            identity=dict(arm=arm,seed=seed,settings=cfg,manifest_sha256=sha(mp))
            d=out/f'{arm}_{seed}';d.mkdir(exist_ok=True)
            model,opt=train_arm(d,identity,old,meta,rows,device)
            final=receipts(d,cfg['schedule'],identity)[-1]
            saved=torch.load(checkpoint_paths(d,cfg['schedule'][-1])[0],map_location='cpu',weights_only=True)
            train_seconds=saved['train_seconds'];peak=saved['peak_allocated_bytes'];del saved
            curve=trajectory(model,d,identity,sample,device)
            report=dict(trajectory=curve,identity=identity,checkpoint_sha256=final['sha256'],train_seconds=train_seconds,peak_train_allocated_bytes=peak,parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),train=metrics(model,[r for r in sample if r['split']=='train'],device),composition=metrics(model,[r for r in sample if r['split']=='composition'],device),relations=relation_audit(model,device),release_admitted=False)
            rp=d/'evaluation.json'
            if rp.exists():
                if json.loads(rp.read_text())!=report:raise ValueError('evaluation mismatch')
            else:exclusive_json(rp,report)
            summary.append(report);del model,opt
            if model_digest(old)!=before or source_identity(root,a.source_head)!=source:raise RuntimeError('source or donor mutated')
    sp=out/'summary.json'
    result=dict(manifest_sha256=sha(mp),arms=summary,heldout_selected=False,release_admitted=False)
    if sp.exists():
        if json.loads(sp.read_text())!=result:raise ValueError('summary mismatch')
    else:exclusive_json(sp,result)

if __name__=='__main__':main()
