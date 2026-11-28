"""Registered complete six-arm x three-seed X369 learning campaign.

One bounded native chunk per invocation; deterministic round-robin by
training progress, never by observed heldout results. Reuses the accepted
checkpointed runner. Fixed schedule, immutable chain and source identity.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from scripts.run_x369_mod3_checkpointed import NAMES,identity,sha,run
from scripts.x369_r4_execution_device import resolve_device

SEEDS=(41,42,43)


def require(ok,msg):
    if not ok:raise ValueError(msg)


def write_json(path,obj):
    with path.open('x') as f:
        json.dump(obj,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')


def load(path):
    return json.loads(path.read_text())


def registration(a):
    dev=resolve_device(a.device)
    arms=[]
    for seed in SEEDS:
        for arm in NAMES:
            spec=SimpleNamespace(seed=seed,arm=arm,width=a.width,experts=a.experts,
                dtype=a.dtype,steps=a.steps,report_every=a.report_every,lr=a.lr,weight_decay=a.weight_decay)
            arms.append(dict(name=f'{arm}-{seed}',contract=identity(spec,dev)))
    return dict(schema=1,arms=arms,seed_order=list(SEEDS),arm_order=list(NAMES),
        fixed_horizon=a.steps,heldout_selection=False,runner_device=a.device,
        scheduler_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def inspect_arm(root,entry,*,verify_all=False):
    d=root/entry['name'];d.mkdir(exist_ok=True)
    files=sorted(d.glob('chunk-*.json'))
    last=None;parent=None;step=0;seconds=0.;peak=None
    for k,p in enumerate(files,1):
        require(p.name==f'chunk-{k:06d}.json','missing or unexpected receipt ordinal')
        r=load(p);cp=p.with_suffix('.pt')
        require(cp.is_file() and r['contract']==entry['contract'] and
                r['parent_checkpoint_sha256']==parent and r['first_step']==step+1 and
                step<r['last_step']<=entry['contract']['horizon'] and
                r['heldout_never_trained'] is True and r['heldout_checkpoint_selection'] is False,
                'learning receipt source, horizon or parent chain mismatch')
        require(len(r['per_step_training_losses'])==r['last_step']-step,'missing training losses')
        require(r['status']==('FIXED_HORIZON_COMPLETE' if r['last_step']==entry['contract']['horizon']
                else 'CHUNK_CHECKPOINTED'),'false completion marker')
        if last:
            require(r['initial_state_sha256']==last['initial_state_sha256'] and
                    r['sampled_trajectory'][:len(last['sampled_trajectory'])]==last['sampled_trajectory'],
                    'source trajectory changed during resume')
        if verify_all or k==len(files):
            require(sha(cp)==r['checkpoint_sha256'],'checkpoint file digest mismatch')
        step=r['last_step'];parent=r['checkpoint_sha256'];last=r
        seconds+=r['chunk_training_and_diagnostic_seconds']
        v=r['chunk_peak_gpu_bytes']
        if v is not None:peak=v if peak is None else max(peak,v)
    expected={p.with_suffix('.pt') for p in files}
    require(set(d.glob('chunk-*.pt'))==expected and not list(d.glob('*.partial')),
            'orphan checkpoint or partial chunk; inspect before continuation')
    return dict(entry=entry,directory=d,count=len(files),last=last,step=step,parent=parent,
                measured_training_and_diagnostic_seconds=seconds,peak_gpu_bytes=peak)


def report(states,plan):
    arms=[]
    for s in states:
        r=s['last'];last=r['sampled_trajectory'][-1] if r and r['sampled_trajectory'] else None
        complete=s['step']==plan['fixed_horizon']
        a=s['entry']['contract']
        arms.append(dict(arm=a['arm'],seed=a['seed'],completed_steps=s['step'],
            complete=complete,final_state_sha256=r['final_model_sha256'] if r else None,
            initial_state_sha256=r['initial_state_sha256'] if r else None,
            final_observed_audit=last['audit'] if complete and last else None,
            final_observed_geometry=last.get('geometry') if complete and last else None,
            classification=r['classification'] if r else None,
            measured_training_and_diagnostic_seconds=s['measured_training_and_diagnostic_seconds'],
            peak_gpu_bytes=s['peak_gpu_bytes'],
            horizon_complete_required_for_quality_comparison=True))
    # Four computational arms must start from precisely the same learner.
    for seed in SEEDS:
        hashes={r['initial_state_sha256'] for r in arms if r['seed']==seed and
                r['arm'] in NAMES[:4] and r['initial_state_sha256'] is not None}
        require(len(hashes)<=1,'matched computational arms started differently')
    return dict(status='FIXED_CAMPAIGN_COMPLETE' if all(r['complete'] for r in arms)
                else 'CAMPAIGN_IN_PROGRESS',arms_required=len(arms),
                arms_complete=sum(r['complete'] for r in arms),arms=arms,
                heldout_selection=False,learned_mechanism_certified=False,
                native_fitted_LILA_certified=False)


def advance(a):
    root=a.directory;root.mkdir(parents=True,exist_ok=True)
    path=root/'registration.json';plan=registration(a)
    if path.exists():require(load(path)==plan,'registered campaign differs; use new directory')
    else:write_json(path,plan)
    states=[inspect_arm(root,entry,verify_all=a.verify_all) for entry in plan['arms']]
    remaining=[s for s in states if s['step']<plan['fixed_horizon']]
    if remaining and not a.audit_only:
        # Stable sort preserves predeclared seed/arm order on ties.
        s=min(remaining,key=lambda v:v['step'])
        c=s['entry']['contract'];stem=s['directory']/f'chunk-{s["count"]+1:06d}'
        previous=s['directory']/f'chunk-{s["count"]:06d}.pt' if s['count'] else None
        run(SimpleNamespace(arm=c['arm'],seed=c['seed'],device=a.device,
            dtype=a.dtype,width=a.width,experts=a.experts,steps=a.steps,
            report_every=a.report_every,chunk_steps=a.chunk_steps,seconds=a.seconds,
            lr=a.lr,weight_decay=a.weight_decay,resume=previous,
            resume_sha256=s['parent'],output=stem.with_suffix('.pt'),receipt=stem.with_suffix('.json')))
        states=[inspect_arm(root,e) for e in plan['arms']]
    return report(states,plan)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--device',choices=('gfx803','cpu'),default='gfx803')
    p.add_argument('--dtype',choices=('float32','float64'),default='float32')
    p.add_argument('--width',type=int,default=384);p.add_argument('--experts',type=int,default=4)
    p.add_argument('--steps',type=int,default=2000);p.add_argument('--report-every',type=int,default=100)
    p.add_argument('--chunk-steps',type=int,default=100);p.add_argument('--seconds',type=float,default=80)
    p.add_argument('--lr',type=float,default=.001);p.add_argument('--weight-decay',type=float,default=.1)
    p.add_argument('--audit-only',action='store_true');p.add_argument('--verify-all',action='store_true')
    p.add_argument('--report',type=Path)
    a=p.parse_args();r=advance(a)
    if a.report:
        a.report.parent.mkdir(parents=True,exist_ok=True);write_json(a.report,r)
    print(json.dumps(r,sort_keys=True,indent=2,allow_nan=False))


if __name__=='__main__':main()
