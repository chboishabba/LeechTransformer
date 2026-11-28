"""Independent residual memory benchmark with an exact semantic-only ceiling.
Original Leech-LILA: Anatolii Kornienko; DASHI extension.
Synthetic public task, not original LILA language quality.
"""
import argparse,json,random,time
from itertools import product
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from scripts.x369_split_carrier import LearnedGeneratorSplitCarrier
from scripts.x369_learned_generator_weld import fit_generators
from scripts.audit_x369_joint_action import joint_records,MEANS
from scripts.audit_x369_ordered_signed_task import TRIADS,endpoint
from scripts.train_x369_ordered_action import encode_inputs
from scripts.x369_source_acceptance import source_identity
from scripts.run_x369_ordered_campaign import exclusive_json,sha
from scripts.run_x369_phase_trajectory_gpu import snapshot
from scripts.x369_campaign_progress import CampaignProgress

def records():
    # Same guards, seed, suffix, and semantic endpoint; distinct ordered prefix.
    # Training and composition use disjoint common suffixes.
    for split,suffixes in (('train',('','a','b','j','n')),('composition',('jn','nj','aj','ja','abjn'))):
        for suffix,m,d,guards in product(suffixes,MEANS,TRIADS,((1,2,1,1),(2,1,1,0))):
            a='ab'+suffix;b='ba'+suffix
            assert endpoint(a,guards,m,d)==endpoint(b,guards,m,d)
            for label,w in enumerate((a,b)):
                yield dict(split=split,word=w,guards=guards,mean=m,residual=d,label=label)

def prepare(rows,device):
    s,o=zip(*(encode_inputs(r,'signed') for r in rows))
    return (torch.tensor(s,device=device,dtype=torch.float32),
            torch.tensor(o,device=device,dtype=torch.float32),
            torch.tensor([r['label'] for r in rows],device=device,dtype=torch.long))

class MemoryLearner(nn.Module):
    def __init__(self,matrices):
        super().__init__()
        self.carrier=LearnedGeneratorSplitCarrier(matrices)
        source=self.carrier.source
        source.readout.requires_grad_(False)
        self.head=nn.Linear(source.experts*2*source.width,2)
    def forward(self,s,o):
        _,trace=self.carrier.rollout(s,o)
        # Query invariant under signed frame: squared magnitude is insufficient
        # in general, so retain full signed carrier for this declared task.
        return self.head(trace[-1].state.sparse.flatten(1))

def grouped(rows):
    out={}
    for r in rows:out.setdefault(len(r['word']),[]).append(r)
    return out

@torch.no_grad()
def evaluate(model,groups,device):
    model.eval();total=correct=0;loss=0.
    for rs in groups.values():
        for i in range(0,len(rs),128):
            s,o,y=prepare(rs[i:i+128],device);v=model(s,o)
            correct+=int(((v.argmax(-1)==y)&(v==v.amax(-1,keepdim=True)).sum(-1).eq(1)).sum())
            loss+=float(F.cross_entropy(v,y,reduction='sum'));total+=len(y)
    return dict(rows=total,unique_correct=correct,ce=loss/total,semantic_only_ceiling=total//2)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-head',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=2000);p.add_argument('--interval',type=int,default=250)
    p.add_argument('--seed',type=int,default=41);p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--progress-seconds',type=float,default=10.)
    a=p.parse_args()
    if min(a.steps,a.interval,a.batch_size)<1 or not 0<a.progress_seconds<float('inf'):p.error('invalid scope')
    if not torch.cuda.is_available() or not torch.version.hip or not torch.cuda.get_device_properties(0).gcnArchName.startswith('gfx803'):raise RuntimeError('native gfx803 required')
    dev=torch.device('cuda',0);root=Path(__file__).resolve().parents[1];source=source_identity(root,a.source_head)
    matrices=fit_generators(joint_records());rows=list(records());train=grouped(r for r in rows if r['split']=='train');test=grouped(r for r in rows if r['split']=='composition')
    torch.manual_seed(a.seed);rng=random.Random(a.seed);model=MemoryLearner(matrices).to(dev)
    opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=.001,weight_decay=.1)
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    manifest=dict(source=source,seed=a.seed,steps=a.steps,interval=a.interval,batch=a.batch_size,task='first_literal_same_semantic_v1')
    mp=out/'manifest.json'
    if mp.exists():
        if json.loads(mp.read_text())!=manifest:raise ValueError('manifest mismatch')
    else:exclusive_json(mp,manifest)
    schedule=sorted({0,a.steps,*range(a.interval,a.steps,a.interval)})
    start=0
    for step in schedule:
        cp=out/f'step_{step:08d}.pt';rp=cp.with_suffix('.json')
        if cp.exists()!=rp.exists():raise ValueError('orphan snapshot')
        if not cp.exists():break
        receipt=json.loads(rp.read_text())
        if receipt!=dict(step=step,sha256=sha(cp),manifest_sha256=sha(mp)):raise ValueError('snapshot receipt')
        q=torch.load(cp,map_location='cpu',weights_only=True)
        if q['step']!=step or q['manifest_sha256']!=sha(mp):raise ValueError('snapshot identity')
        model.load_state_dict(q['model'],strict=True);opt.load_state_dict(q['optimizer'])
        rng.setstate(q['rng']);torch.set_rng_state(q['torch_rng']);torch.cuda.set_rng_state(q['device_rng'],dev)
        start=step;del q
    def save(step):
        cp=out/f'step_{step:08d}.pt'
        digest=snapshot(cp,dict(step=step,manifest_sha256=sha(mp),model=model.state_dict(),optimizer=opt.state_dict(),rng=rng.getstate(),torch_rng=torch.get_rng_state(),device_rng=torch.cuda.get_rng_state(dev)))
        exclusive_json(cp.with_suffix('.json'),dict(step=step,sha256=digest,manifest_sha256=sha(mp)))
    if not (out/'step_00000000.pt').exists():save(0)
    key=('residual',a.seed);progress=CampaignProgress([key],a.steps,0,out,a.progress_seconds);progress.seed(key,start);progress.begin(key,'train')
    for t in range(start+1,a.steps+1):
        model.train();rs=rng.choices(rng.choice(list(train.values())),k=a.batch_size)
        s,o,y=prepare(rs,dev);opt.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model(s,o),y)
        if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise RuntimeError('nonfinite gradient')
        opt.step();progress.update(key,'train',t)
        if t in schedule:save(t);progress.emit(key,'train',force=True,checkpoint_step=t)
    progress.begin(key,'audit')
    report=dict(kind='residual_memory',manifest_sha256=sha(mp),train=evaluate(model,train,dev),composition=evaluate(model,test,dev),release_admitted=False)
    if source!=source_identity(root,a.source_head):raise RuntimeError('source changed')
    path=out/'summary.json'
    if path.exists():
        if json.loads(path.read_text())!=report:raise ValueError('evaluation mismatch')
    else:exclusive_json(path,report)
    progress.update(key,'audit',1,force=True);progress.complete(key)
if __name__=='__main__':main()
