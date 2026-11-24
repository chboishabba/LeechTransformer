"""One native, checkpoint-resumable ordered LILA development campaign.
No CI, CPU training fallback, reserve access, or model selection.
"""
import argparse,hashlib,itertools,json,subprocess,sys
from pathlib import Path


def arms(seeds,modes,families):
    return [dict(seed=s,mode=m,paired=f=='paired',null=bool(n),ternary=bool(t)) for s,m,f,n,t in itertools.product(seeds,modes,families,(0,1),(0,1))]

def arm_name(a):
    return f"ordered_{a['mode']}_paired{int(a['paired'])}_seed{a['seed']}_null{int(a['null'])}_ternary{int(a['ternary'])}"

def exclusive_json(path,obj):
    data=json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+'\n'
    with path.open('x') as f:f.write(data)

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def checkpoint_receipt(directory,name,a,settings):
    target=directory/(name+'.pt')
    if not target.exists():return None
    matches=[]
    for log in sorted(directory.glob('train.*.log')):
        for line in log.read_text(errors='replace').splitlines():
            try:r=json.loads(line)
            except ValueError:continue
            if isinstance(r,dict) and r.get('kind')=='final_checkpoint':matches.append(r)
    actual=sha(target)
    matches=[r for r in matches if r.get('sha256')==actual and Path(r.get('path','')).resolve()==target.resolve()]
    if len(matches)!=1:raise RuntimeError('checkpoint missing unique training receipt: '+name)
    r=matches[0];m=r['metadata'];expected=dict(**a,width=settings['width'],steps=settings['steps'],batch_size=settings['batch_size'],joint_mean=True,fixed_horizon_completed=True,public_task_only=True)
    if any(m.get(k)!=v for k,v in expected.items()):raise RuntimeError('checkpoint arm or horizon mismatch: '+name)
    return r

def invoke(root,dir,kind,cmd):
    attempt=0
    while (dir/f'{kind}.{attempt}.log').exists():attempt+=1
    log=dir/f'{kind}.{attempt}.log'
    with log.open('x') as f:
        proc=subprocess.run([sys.executable,'-m',*cmd],cwd=root,stdout=f,stderr=subprocess.STDOUT)
    if proc.returncode:raise RuntimeError(f'{kind} failed ({proc.returncode}); see {log}')

def replay_valid(path,checkpoint,scope):
    if not path.exists():return False
    r=json.loads(path.read_text());required=('train','composition','output_action','hidden_chart','hidden_generators','consumer_closure','reachable_bins','joint_action','precision_guards')
    if r.get('kind')!='ordered_checkpoint_replay' or r.get('sha256')!=checkpoint['sha256'] or r.get('metadata')!=checkpoint['metadata'] or r.get('trace_probe_equal') is not True or r.get('scope')!=scope or any(k not in r for k in required):
        raise RuntimeError('existing replay receipt is foreign or incomplete: '+str(path))
    return True

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--seeds',type=int,nargs='+',default=[41,42,43]);p.add_argument('--modes',nargs='+',choices=('c3','s3','full','entry'),default=['c3','s3','full','entry']);p.add_argument('--families',nargs='+',choices=('single','paired'),default=['single','paired']);p.add_argument('--width',type=int,default=12);p.add_argument('--steps',type=int,default=2000);p.add_argument('--batch-size',type=int,default=128);p.add_argument('--eval-every',type=int,default=250);p.add_argument('--contexts',type=int,default=8);p.add_argument('--joint-contexts',type=int,default=12);p.add_argument('--pairs',type=int,default=128);p.add_argument('--precision-rows',type=int,default=256);p.add_argument('--horizon',type=int,default=2);p.add_argument('--quantum',type=float,default=.125);p.add_argument('--plan',action='store_true');a=p.parse_args()
    if any(len(v)!=len(set(v)) for v in (a.seeds,a.modes,a.families)) or a.width%3 or min(a.width,a.steps,a.batch_size,a.eval_every,a.contexts,a.joint_contexts,a.pairs,a.precision_rows)<1 or not 1<=a.horizon<=3 or not 0<a.quantum<float('inf'):raise ValueError('invalid or duplicated campaign configuration')
    plan=arms(a.seeds,a.modes,a.families);settings={k:getattr(a,k) for k in ('width','steps','batch_size','eval_every')};scope={k:getattr(a,k) for k in ('quantum','contexts','horizon','joint_contexts','pairs','precision_rows')}
    manifest=dict(schema=1,kind='x369_native_joint_campaign',settings=settings,scope=scope,arms=plan,joint_mean=True,public_task_only=True,release_admitted=False)
    if a.plan:print(json.dumps(manifest,indent=2));return
    import torch
    from scripts.x369_ordered_checkpoint import fingerprint
    if not torch.cuda.is_available() or not torch.version.hip or not getattr(torch.cuda.get_device_properties(0),'gcnArchName','').startswith('gfx803'):raise RuntimeError('native gfx803 HIP required; no CPU fallback')
    root=Path(__file__).resolve().parents[1];out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    manifest.update(sources=fingerprint(root),runtime=dict(torch=str(torch.__version__),hip=torch.version.hip,arch=getattr(torch.cuda.get_device_properties(0),'gcnArchName','')))
    mp=out/'manifest.json'
    if mp.exists():
        if json.loads(mp.read_text())!=manifest:raise RuntimeError('campaign configuration or sources changed; use new output directory')
    else:exclusive_json(mp,manifest)
    completed=[]
    for arm in plan:
        name=arm_name(arm);d=out/name;d.mkdir(exist_ok=True);cp=checkpoint_receipt(d,name,arm,settings)
        if cp is None:
            cmd=['scripts.run_x369_fair_ordered_gpu','--require-gfx803','--joint-mean','--defer-audits','--checkpoint-dir',str(d),'--arm',f"{int(arm['null'])}{int(arm['ternary'])}",'--mode',arm['mode'],'--seed',str(arm['seed'])]
            for k,v in settings.items():cmd+=['--'+k.replace('_','-'),str(v)]
            if arm['paired']:cmd+=['--paired']
            invoke(root,d,'train',cmd);cp=checkpoint_receipt(d,name,arm,settings)
            if cp is None:raise RuntimeError('training returned without checkpoint')
        receipt=d/'replay.json'
        if not replay_valid(receipt,cp,scope):
            cmd=['scripts.replay_x369_ordered_checkpoint','--checkpoint',str(d/(name+'.pt')),'--sha256',cp['sha256'],'--output',str(receipt)]
            for k,v in scope.items():cmd+=['--'+k.replace('_','-'),str(v)]
            invoke(root,d,'replay',cmd)
            if not replay_valid(receipt,cp,scope):raise RuntimeError('missing replay receipt')
        result=json.loads(receipt.read_text());completed.append(dict(arm=arm,checkpoint_sha256=cp['sha256'],receipt_sha256=sha(receipt),train=result['train'],composition=result['composition']))
        print(json.dumps(dict(kind='arm_complete',arm=name,completed=len(completed),planned=len(plan)),flush=True))
    summary=dict(kind='campaign_receipts_complete',manifest_sha256=sha(mp),completed=completed,release_admitted=False,universal_closure_certified=False)
    sp=out/'summary.json'
    if sp.exists():
        if json.loads(sp.read_text())!=summary:raise RuntimeError('summary mismatch')
    else:exclusive_json(sp,summary)
if __name__=='__main__':main()
