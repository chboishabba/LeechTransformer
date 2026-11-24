"""GPU-first fair residual experiment, same existing ordered task and source step.
No CPU training fallback or release lockbox. Original Leech-LILA: Anatolii Kornienko.
"""
import argparse,copy,json,random,time
from collections import defaultdict
import torch
from torch import nn
from torch.nn import functional as F
from scripts.train_x369_ordered_action import OrderedLearner,batch,score,final_action_audit,HEADS,records
from models.geometric_fair_residual import fair_arm
from scripts.audit_x369_hidden_chart import fit_chart,chart_metrics,transport_delta


class FairLearner(OrderedLearner):
    def initial(self,seed):
        b=len(seed);m=self.mean_seed(seed[:,:3]).reshape(b,self.experts,self.width);d=self.residual_seed(seed[:,3:]).reshape_as(m)
        return torch.stack((m+d,m-d),2) if self.paired else m+d
    def advance(self,op,h):
        x=self.embed(op)
        if self.paired:
            a=self.history.step(x,h,train_sign=self.training);return a.branches,a.history,a.ids,a.signs,a.weights
        return self.history.step(x,h,train_sign=self.training)
    def rollout(self,seed,ops):
        h=self.initial(seed)
        for t in range(ops.shape[1]):y,h,*_=self.advance(ops[:,t],h)
        return y,h
    def consume(self,y,h):return self.readout(torch.cat((y.flatten(1),h.flatten(1)),1)).split(HEADS,-1)
    def forward(self,seed,ops):return self.consume(*self.rollout(seed,ops))


def audit_hidden(model,rows,device,dtype):
    """Training-only chart, frozen public composition scores by time; signed input only."""
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[len(r['word'])].append(i)
    zs={}
    with torch.no_grad():
        for ix in groups.values():
            for start in range(0,len(ix),256):
                ids=ix[start:start+256];s,o,_=batch([rows[i] for i in ids],'signed',device,dtype)
                h=model.rollout(s,o)[1].flatten(1).cpu()
                zs.update(zip(ids,h))
    Z=torch.stack([zs[i] for i in range(len(rows))]);Y=torch.tensor([r['mean_label']+r['signed_label'] for r in rows],dtype=torch.float64)
    train=torch.tensor([r['split']=='train' for r in rows]);chart=fit_chart(Y[train],Z[train])
    metrics={'train':chart_metrics(chart,Y[train],Z[train])}
    for t,ix in groups.items():
        ix=[i for i in ix if not train[i]]
        if ix:metrics[f'composition_t{t}']=chart_metrics(chart,Y[ix],Z[ix])
    result=dict(task_rank=chart['task_rank'],chart_rank=chart['chart_rank'],metrics=metrics,learned_same_action_established=False)
    if not chart['full_rank']:return dict(result,intervention='rank_deficient')
    # Natural antipodal counterfactual versus chart edit at post-prefix history.
    # Only nonzero D and public composition extensions with observed prefix.
    key=lambda r,w,d:(w,r['guards'],r['mean'],d)
    lookup={key(r,r['word'],r['residual']):i for i,r in enumerate(rows)}
    pairs=[]
    for j,r in enumerate(rows):
        if r['split']=='train' or len(r['word'])<2 or not any(r['residual']):continue
        d=r['residual'];nd=tuple(-v for v in d);w=r['word'][:-1]
        i=lookup.get(key(r,w,d));k=lookup.get(key(r,w,nd));l=lookup.get(key(r,r['word'],nd))
        if None not in (i,k,l):pairs.append((i,k,j,l))
    count=0;err=[0.,0.,0.];route=[0,0,0];history=[0.,0.,0.];sign=[0,0,0];weight=[0.,0.,0.]
    gen=torch.Generator().manual_seed(719);perm=torch.randperm(Z.shape[1],generator=gen)
    with torch.no_grad():
        for start in range(0,len(pairs),128):
            chunk=pairs[start:start+128];i,k,j,l=map(list,zip(*chunk));h=Z[i];cf=Z[k];delta=transport_delta(chart,Y[i],Y[k]).to(h)
            # Last literal only, grouped by word length is unnecessary here.
            from scripts.train_x369_ordered_action import encode_inputs
            op=torch.tensor([encode_inputs(rows[q],'signed')[1][-1] for q in j],device=device,dtype=dtype)
            shape=(-1,model.experts,2,model.width) if model.paired else (-1,model.experts,model.width)
            execute=lambda z:model.advance(op,z.to(device,dtype).reshape(shape))
            target=execute(cf);ty,th,ti,ts,tw=target;tp=model.consume(ty,th)
            for n,z in enumerate((h,h+delta,h+delta[:,perm])):
                y,hh,ids,ss,ww=execute(z);p=model.consume(y,hh)
                err[n]+=sum(float((a.softmax(-1)-b.softmax(-1)).abs().sum()) for a,b in zip(p,tp))
                route[n]+=int((ids==ti).all(-1).sum());sign[n]+=int((ss==ts).flatten(1).all(-1).sum());weight[n]+=float((ww-tw).abs().sum());history[n]+=float((hh-th).flatten(1).norm(dim=-1).sum())
            count+=len(chunk)
    return dict(result,intervention_pairs=count,intervention_order=['unedited','chart_delta','coordinate_permuted_delta_null'],probability_l1_sum_per_row=[v/max(count,1) for v in err],same_ordered_ids=route,same_all_signs=sign,weights_l1_per_row=[v/max(count,1) for v in weight],next_history_l2_per_row=[v/max(count,1) for v in history],null_is_norm_matched_not_independent_gauge_proof=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('c3','s3','full','entry'),default='c3');p.add_argument('--paired',action='store_true');p.add_argument('--seed',type=int,default=41);p.add_argument('--width',type=int,default=12);p.add_argument('--steps',type=int,default=2000);p.add_argument('--batch-size',type=int,default=128);p.add_argument('--require-gfx803',action='store_true');p.add_argument('--consumer-closure',action='store_true');p.add_argument('--closure-pairs',type=int,default=128);p.add_argument('--checkpoint-dir',type=str);p.add_argument('--joint-mean',action='store_true');p.add_argument('--arm',choices=('00','01','10','11'));p.add_argument('--defer-audits',action='store_true');p.add_argument('--eval-every',type=int,default=100);a=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('GPU required')
    if a.require_gfx803 and (not torch.version.hip or not getattr(torch.cuda.get_device_properties(0),'gcnArchName','').startswith('gfx803')):raise RuntimeError('gfx803 HIP required')
    if a.width<3 or a.width%3 or min(a.steps,a.batch_size,a.eval_every)<1:raise ValueError('invalid dimensions or horizon')
    torch.manual_seed(a.seed);dev=torch.device('cuda');dt=torch.float32
    from scripts.audit_x369_joint_action import joint_records,task_digest
    if a.defer_audits and not a.checkpoint_dir:raise ValueError('deferred audits require checkpoint')
    base=FairLearner(a.width,4,a.paired,'dense').to(dev,dt);rows=list(joint_records() if a.joint_mean else records());task_sha=task_digest(rows);groups={k:defaultdict(list) for k in ('train','composition')}
    for r in rows:groups[r['split']][len(r['word'])].append(r)
    pre_s,pre_o,_=batch(groups['train'][1][:32],'signed',dev,dt);base.eval()
    with torch.no_grad():expected=base(pre_s,pre_o)
    for null in (False,True):
        for ternary in (False,True):
            if a.arm is not None and a.arm!=f'{int(null)}{int(ternary)}':continue
            model=copy.deepcopy(base);source=model.history.source if a.paired else model.history;replacement=fair_arm(source,a.mode,null,ternary,a.seed)
            if a.paired:model.history.source=replacement
            else:model.history=replacement
            model.eval()
            with torch.no_grad():
                if not all(torch.equal(x,y) for x,y in zip(expected,model(pre_s,pre_o))):raise RuntimeError('initial function mismatch on preflight')
            opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.1);rng=random.Random(a.seed);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            arm=dict(mode=a.mode,null=null,ternary=ternary,paired=a.paired,seed=a.seed,parameters=sum(v.numel() for v in model.parameters() if v.requires_grad),buffer_bytes=sum(v.numel()*v.element_size() for v in model.buffers()))
            print(json.dumps(dict(kind='arm',**arm,initial_preflight=True,torch=torch.__version__,hip=torch.version.hip,physical_work_equal_not_claimed=True)),flush=True)
            for t in range(1,a.steps+1):
                model.train();rs=rng.choice(list(groups['train'].values()));rs=rng.choices(rs,k=a.batch_size);s,o,y=batch(rs,'signed',dev,dt);opt.zero_grad(set_to_none=True);pred=model(s,o);loss=sum(F.cross_entropy(v,y[:,i]) for i,v in enumerate(pred));loss.backward();opt.step()
                if (t==1 or t%a.eval_every==0 or t==a.steps) and not (a.defer_audits and t==a.steps):
                    model.eval();torch.cuda.synchronize();print(json.dumps(dict(kind='trajectory',**arm,step=t,train=score(model,groups['train'],'signed',dev,dt),composition=score(model,groups['composition'],'signed',dev,dt),elapsed_seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),grokking_established=False)),flush=True)
            model.eval()
            if a.checkpoint_dir:
                from pathlib import Path
                from scripts.x369_ordered_checkpoint import save_checkpoint,fingerprint,trace_probe
                root=Path(__file__).resolve().parents[1]
                meta=dict(**arm,width=a.width,experts=4,steps=a.steps,batch_size=a.batch_size,device=str(dev),dtype='float32',torch=str(torch.__version__),hip=torch.version.hip,arch=getattr(torch.cuda.get_device_properties(dev),'gcnArchName',''),fixed_horizon_completed=True,public_task_only=True,joint_mean=a.joint_mean,task_sha256=task_sha,trace_probe=trace_probe(model,rows,dev,dt))
                target=Path(a.checkpoint_dir)/f"ordered_{a.mode}_paired{int(a.paired)}_seed{a.seed}_null{int(null)}_ternary{int(ternary)}.pt"
                sha=save_checkpoint(target,model,meta,fingerprint(root))
                print(json.dumps(dict(kind='final_checkpoint',path=str(target),sha256=sha,metadata=meta)),flush=True)
            if a.defer_audits:
                del model,opt
                continue
            from scripts.audit_x369_hidden_generators import audit_hidden_generators
            print(json.dumps(dict(kind='final',**arm,output_action=final_action_audit(model,rows,'signed',dev,dt),hidden=audit_hidden(model,rows,dev,dt),hidden_generators=audit_hidden_generators(model,rows,dev,dt))),flush=True)
            if a.joint_mean:
                from scripts.audit_x369_joint_action import audit_joint
                print(json.dumps(dict(kind='joint_action',**arm,**audit_joint(model,rows,dev,dt))),flush=True)
            if a.consumer_closure:
                from scripts.audit_x369_consumer_closure import audit_consumer_closure
                print(json.dumps(dict(kind='consumer_closure',**arm,**audit_consumer_closure(model,rows,dev,dt,max_pairs=a.closure_pairs))),flush=True)
            del model,opt
if __name__=='__main__':main()
