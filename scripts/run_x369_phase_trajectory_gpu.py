"""Native time-indexed phase trajectory, separate from existing final-only baseline.
Fixed schedules, matched arms, no CPU training, no final reserve or selection.
"""
import argparse,copy,hashlib,io,json,os,random
from collections import defaultdict
from pathlib import Path
import torch
from torch.nn import functional as F
from scripts.run_x369_ordered_campaign import arms,arm_name,exclusive_json,sha
from scripts.x369_ordered_checkpoint import fingerprint,SOURCES,trace_probe
from scripts.audit_x369_joint_action import joint_records,task_digest,select_contexts
from scripts.run_x369_fair_ordered_gpu import FairLearner
from scripts.train_x369_ordered_action import batch,score
from models.geometric_fair_residual import fair_arm
from scripts.audit_x369_learning_phases import phase_audit,cross_time

PHASE_SOURCES=SOURCES+('scripts/x369_phase_geometry.py','scripts/audit_x369_learning_phases.py','scripts/run_x369_phase_trajectory_gpu.py')


def schedule(steps,interval):
    if min(steps,interval)<1:raise ValueError('positive horizon and interval required')
    return sorted({0,1,steps,*range(interval,steps,interval)})


def cpu_tree(x):
    if isinstance(x,torch.Tensor):return x.detach().cpu().clone()
    if isinstance(x,dict):return {k:cpu_tree(v) for k,v in x.items()}
    if isinstance(x,list):return [cpu_tree(v) for v in x]
    if isinstance(x,tuple):return tuple(cpu_tree(v) for v in x)
    return x


def snapshot(path,payload):
    """Atomic exclusive checkpoint publication. No overwrite."""
    buf=io.BytesIO();torch.save(cpu_tree(payload),buf);data=buf.getvalue()
    tmp=path.with_suffix('.tmp')
    with tmp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    try:os.link(tmp,path)
    finally:tmp.unlink()
    return hashlib.sha256(data).hexdigest()


def reopen(path,receipt,expected):
    data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=receipt['sha256']:raise ValueError('snapshot SHA mismatch')
    p=torch.load(io.BytesIO(data),map_location='cpu',weights_only=True)
    if p.get('schema')!=1 or p.get('kind')!='x369_phase_snapshot' or p.get('identity')!=expected or p.get('step')!=receipt['step'] or p.get('probe')!=receipt['probe']:raise ValueError('snapshot identity mismatch')
    return p


def setup(a,width,device):
    torch.manual_seed(a['seed'])
    base=FairLearner(width,4,a['paired'],'dense').to(device,torch.float32)
    model=copy.deepcopy(base)
    src=model.history.source if a['paired'] else model.history
    replacement=fair_arm(src,a['mode'],a['null'],a['ternary'],a['seed'])
    if a['paired']:model.history.source=replacement
    else:model.history=replacement
    return base,model


def checkpoint_paths(d,t):
    return d/f'step_{t:08d}.pt',d/f'step_{t:08d}.json'


def receipts(d,steps,identity):
    found=[];missing=False
    for t in steps:
        p,r=checkpoint_paths(d,t)
        if p.exists()!=r.exists():raise RuntimeError('orphan snapshot or receipt; preserve for explicit recovery')
        if not p.exists():missing=True;continue
        if missing:raise RuntimeError('noncontiguous checkpoint schedule')
        v=json.loads(r.read_text())
        if v.get('step')!=t or v.get('identity')!=identity or sha(p)!=v.get('sha256'):raise RuntimeError('foreign snapshot receipt')
        found.append(v)
    return found


def train_arm(d,identity,rows,device):
    a=identity['arm'];cfg=identity['settings'];steps=cfg['schedule']
    old=receipts(d,steps,identity)
    if old and old[-1]['step']==steps[-1]:return
    base,model=setup(a,cfg['width'],device)
    groups=defaultdict(list)
    for r in rows:
        if r['split']=='train':groups[len(r['word'])].append(r)
    ss,oo,_=batch(groups[1][:32],'signed',device,torch.float32)
    base.eval();model.eval()
    with torch.no_grad():
        if not all(torch.equal(x,y) for x,y in zip(base(ss,oo),model(ss,oo))):raise RuntimeError('common initial function mismatch')
    del base
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.1);rng=random.Random(a['seed']);start=0
    if old:
        r=old[-1];start=r['step'];p=reopen(checkpoint_paths(d,start)[0],r,identity)
        model.load_state_dict(p['model'],strict=True);opt.load_state_dict(p['optimizer'])
        model.eval()
        if trace_probe(model,rows,device,torch.float32)!=p['probe']:raise RuntimeError('restored trace mismatch')
        rng.setstate(p['python_rng']);torch.set_rng_state(p['torch_rng']);torch.cuda.set_rng_state(p['device_rng'],device);del p
    def save(t):
        model.eval();probe=trace_probe(model,rows,device,torch.float32)
        p,r=checkpoint_paths(d,t)
        payload=dict(schema=1,kind='x369_phase_snapshot',identity=identity,step=t,probe=probe,model=model.state_dict(),optimizer=opt.state_dict(),python_rng=rng.getstate(),torch_rng=torch.get_rng_state(),device_rng=torch.cuda.get_rng_state(device))
        digest=snapshot(p,payload)
        exclusive_json(r,dict(identity=identity,step=t,sha256=digest,probe=probe))
        print(json.dumps(dict(kind='phase_snapshot',arm=arm_name(a),step=t,sha256=digest)),flush=True)
    if not old:save(0)
    for t in range(start+1,steps[-1]+1):
        model.train();rs=rng.choice(list(groups.values()));rs=rng.choices(rs,k=cfg['batch_size'])
        s,o,y=batch(rs,'signed',device,torch.float32);opt.zero_grad(set_to_none=True);pred=model(s,o)
        loss=sum(F.cross_entropy(v,y[:,i]) for i,v in enumerate(pred))
        if not torch.isfinite(loss):raise RuntimeError('nonfinite training loss')
        loss.backward();opt.step()
        if t in steps:save(t)


def audit_arm(d,identity,rows,device):
    cfg=identity['settings'];old=receipts(d,cfg['schedule'],identity);previous=None
    if len(old)!=len(cfg['schedule']):raise RuntimeError('complete scheduled training required before evaluation')
    base,model=setup(identity['arm'],cfg['width'],device);del base;model.eval()
    sampled=select_contexts(rows,cfg['contexts']);groups={k:defaultdict(list) for k in ('train','composition')}
    for r in sampled:groups[r['split']][len(r['word'])].append(r)
    for r in old:
        t=r['step'];path=d/f'audit_{t:08d}.json'
        if path.exists():
            current=json.loads(path.read_text())
            if current.get('identity')!=identity or current.get('snapshot_sha256')!=r['sha256'] or current.get('step')!=t or 'phase' not in current:raise RuntimeError('foreign audit')
        else:
            p=reopen(checkpoint_paths(d,t)[0],r,identity);model.load_state_dict(p['model'],strict=True);del p
            if trace_probe(model,rows,device,torch.float32)!=r['probe']:raise RuntimeError('snapshot trace mismatch')
            phase=phase_audit(model,rows,device,torch.float32,cfg['contexts'],cfg['pairs'],cfg['horizon'])
            current=dict(kind='x369_phase_audit',identity=identity,step=t,snapshot_sha256=r['sha256'],phase=phase,train=score(model,groups['train'],'signed',device,torch.float32),composition=score(model,groups['composition'],'signed',device,torch.float32),cross_time=cross_time(previous['phase'],phase) if previous else None,release_admitted=False)
            exclusive_json(path,current)
        previous=current
        print(json.dumps(dict(kind='phase_audit_complete',arm=arm_name(identity['arm']),step=t)),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--stage',choices=('plan','train','audit','both'),default='both');p.add_argument('--seeds',type=int,nargs='+',default=[41,42,43]);p.add_argument('--modes',nargs='+',choices=('c3','s3','full','entry'),default=['c3','s3','full','entry']);p.add_argument('--families',nargs='+',choices=('single','paired'),default=['single','paired']);p.add_argument('--width',type=int,default=12);p.add_argument('--steps',type=int,default=2000);p.add_argument('--interval',type=int,default=250);p.add_argument('--batch-size',type=int,default=128);p.add_argument('--contexts',type=int,default=4);p.add_argument('--pairs',type=int,default=32);p.add_argument('--horizon',type=int,default=2);a=p.parse_args()
    if any(len(v)!=len(set(v)) for v in (a.seeds,a.modes,a.families)) or a.width%3 or min(a.width,a.steps,a.interval,a.batch_size,a.contexts,a.pairs)<1 or not 1<=a.horizon<=3:raise ValueError('invalid campaign scope')
    cfg=dict(width=a.width,batch_size=a.batch_size,schedule=schedule(a.steps,a.interval),contexts=a.contexts,pairs=a.pairs,horizon=a.horizon)
    plan=arms(a.seeds,a.modes,a.families);manifest=dict(kind='x369_phase_trajectory',settings=cfg,arms=plan,release_admitted=False)
    if a.stage=='plan':print(json.dumps(manifest,indent=2));return
    if not torch.cuda.is_available() or not torch.version.hip or not getattr(torch.cuda.get_device_properties(0),'gcnArchName','').startswith('gfx803'):raise RuntimeError('native gfx803 HIP required; no CPU fallback')
    device=torch.device('cuda',0);root=Path(__file__).resolve().parents[1];rows=list(joint_records())
    manifest.update(sources=fingerprint(root,PHASE_SOURCES),task_sha256=task_digest(rows),runtime=dict(torch=str(torch.__version__),hip=torch.version.hip,arch=getattr(torch.cuda.get_device_properties(device),'gcnArchName','')))
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);mp=out/'manifest.json'
    if mp.exists():
        if json.loads(mp.read_text())!=manifest:raise RuntimeError('phase manifest changed; use new output')
    else:exclusive_json(mp,manifest)
    for arm in plan:
        if fingerprint(root,PHASE_SOURCES)!=manifest['sources']:raise RuntimeError('source revision changed')
        d=out/arm_name(arm);d.mkdir(exist_ok=True)
        identity=dict(arm=arm,settings=cfg,manifest_sha256=sha(mp))
        if a.stage in ('train','both'):train_arm(d,identity,rows,device)
        if a.stage in ('audit','both'):audit_arm(d,identity,rows,device)
if __name__=='__main__':main()
