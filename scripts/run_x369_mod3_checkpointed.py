"""Checkpointed execution of the existing six-arm native mod3 campaign.

Reuses original learner, controls, independent task labels and classification.
One arm resident at a time, fixed horizon, <=90s cooperative work chunk.
No heldout optimization/selection; no original fitted LILA claim.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from time import monotonic
import torch
from torch.nn import functional as F

from scripts.run_x369_mod3_maxcut_gpu import (
    ARMS, AdditiveLinearControl, PairMLPControl, exact_fourier_law_logits, state_sha, classify)
from scripts.train_x369_mod3_codec_gpu import Learner
from scripts.audit_x369_typed_mod3_prediction_consumer import (
    TASK_PAIRS,TASK_LABELS,TRAIN_INDICES,HELDOUT_INDEX,typed_mod3_prediction_consumer)
from scripts.x369_r4_execution_device import resolve_device,runtime

ROOT=Path(__file__).resolve().parents[1]
OWNERS=('scripts/run_x369_mod3_maxcut_gpu.py','scripts/train_x369_mod3_codec_gpu.py',
        'scripts/audit_x369_typed_mod3_prediction_consumer.py','models/history_orbit_moe.py',
        'models/trialectic_wave_prior.py','models/x369_shared_head.py',
        'models/x369_two_channel_codec.py','models/x369_differential_backward.py',
        'scripts/x369_r4_execution_device.py','scripts/run_x369_mod3_checkpointed.py')
NAMES=tuple(ARMS)+('additive_linear','pair_mlp')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def identity(a,device):
    return dict(schema=1,seed=a.seed,arm=a.arm,width=a.width,experts=a.experts,
        dtype=a.dtype,horizon=a.steps,report_every=a.report_every,lr=a.lr,
        weight_decay=a.weight_decay,source_owners={p:sha(ROOT/p) for p in OWNERS},
        task_pairs=[list(p) for p in TASK_PAIRS],task_labels=list(TASK_LABELS),
        train_indices=list(TRAIN_INDICES),heldout_index=HELDOUT_INDEX,
        execution_runtime=runtime(device))


def make_arm(a,device):
    dt=getattr(torch,a.dtype)
    torch.manual_seed(a.seed)
    if a.arm in ARMS:
        m=Learner(a.width,a.experts).to(device,dt)
        with torch.no_grad():m.history.c3_residual_weight.normal_(std=.003)
        return m,ARMS[a.arm],True
    # Same independent control initialization sequence as original runner.
    torch.manual_seed(a.seed+1000003)
    additive=AdditiveLinearControl(a.width).to(device,dt)
    if a.arm=='additive_linear':return additive,None,False
    del additive
    return PairMLPControl(a.width).to(device,dt),None,False


def checkpoint_write(path,payload):
    if path.exists():raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.partial')
    with tmp.open('xb') as f:
        torch.save(payload,f);f.flush();os.fsync(f.fileno())
    # No silent replacement of a different checkpoint.
    os.link(tmp,path);tmp.unlink()
    return sha(path)


def run(a):
    if (a.arm not in NAMES or a.width<6 or a.width%3 or a.experts<2 or
        not 0<a.seconds<=90 or min(a.steps,a.report_every,a.chunk_steps)<1 or
        a.lr<=0 or a.weight_decay<0 or bool(a.resume)!=bool(a.resume_sha256)):
        raise ValueError('invalid registered campaign scope')
    if a.output.exists() or a.receipt.exists() or a.output.resolve()==a.receipt.resolve():
        raise FileExistsError('immutable chunk outputs required')
    dev=resolve_device(a.device);dt=getattr(torch,a.dtype)
    contract=identity(a,dev)
    assert HELDOUT_INDEX not in TRAIN_INDICES
    assert tuple(TRAIN_INDICES)==tuple(range(8)) and HELDOUT_INDEX==8
    m,selector,is_learner=make_arm(a,dev)
    initial_hash=state_sha(m)
    opt=torch.optim.AdamW(m.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    completed=0;rows=[]
    if a.resume:
        if sha(a.resume)!=a.resume_sha256:raise ValueError('checkpoint SHA mismatch')
        prior=torch.load(a.resume,map_location='cpu',weights_only=True)
        if prior['contract']!=contract or prior['initial_state_sha256']!=initial_hash:
            raise ValueError('registered campaign or source identity changed')
        completed=prior['completed_steps'];rows=list(prior['rows'])
        if not 0<=completed<a.steps:raise ValueError('checkpoint horizon already reached or invalid')
        m.load_state_dict(prior['model'],strict=True);opt.load_state_dict(prior['optimizer'])
        torch.set_rng_state(prior['cpu_rng'])
        if dev.type=='cuda':torch.cuda.set_rng_state(prior['device_rng'],dev)
        if state_sha(m)!=prior['model_sha256']:raise ValueError('checkpoint model digest mismatch')
    pairs=torch.tensor(TASK_PAIRS,device=dev,dtype=torch.long)
    labels=torch.tensor(TASK_LABELS,device=dev,dtype=torch.long)
    train=torch.tensor(TRAIN_INDICES,device=dev,dtype=torch.long)
    law=typed_mod3_prediction_consumer(exact_fourier_law_logits(pairs,dt))
    if not law['full_task_correct'] or law['hard_label_action_defect_count']!=0:
        raise RuntimeError('independent analytic task control failed')
    start=completed;deadline=monotonic()+a.seconds;losses=[]
    m.train()
    for t in range(start+1,min(a.steps,start+a.chunk_steps)+1):
        if monotonic()>=deadline:break
        opt.zero_grad(set_to_none=True)
        logits=m(pairs[train],selector) if is_learner else m(pairs[train])
        loss=F.cross_entropy(logits,labels[train])
        if not bool(torch.isfinite(loss)):raise RuntimeError('nonfinite training loss')
        loss.backward();opt.step()
        if dev.type=='cuda':torch.cuda.synchronize(dev)
        losses.append(float(loss.detach()));completed=t
        if t==1 or t%a.report_every==0 or t==a.steps:
            m.eval()
            with torch.no_grad():
                all_logits=m(pairs,selector) if is_learner else m(pairs)
                audit=typed_mod3_prediction_consumer(all_logits)
                rows.append(dict(step=t,audit=audit,
                    postupdate_train_ce=float(F.cross_entropy(all_logits[train],labels[train])),
                    predictions=all_logits.argmax(-1).tolist()))
            m.train()
    if completed==start:raise TimeoutError('no complete optimizer step in chunk')
    model_sha=state_sha(m)
    payload=dict(contract=contract,completed_steps=completed,rows=rows,
        initial_state_sha256=initial_hash,model_sha256=model_sha,
        model=m.state_dict(),optimizer=opt.state_dict(),cpu_rng=torch.get_rng_state(),
        device_rng=torch.cuda.get_rng_state(dev) if dev.type=='cuda' else None)
    output_sha=checkpoint_write(a.output,payload)
    r=dict(status='FIXED_HORIZON_COMPLETE' if completed==a.steps else 'CHUNK_CHECKPOINTED',
        contract=contract,checkpoint_sha256=output_sha,parent_checkpoint_sha256=a.resume_sha256,
        first_step=start+1,last_step=completed,per_step_training_losses=losses,
        initial_state_sha256=initial_hash,final_model_sha256=model_sha,
        classification=classify(rows),sampled_trajectory=rows,
        heldout_never_trained=True,heldout_checkpoint_selection=False,
        law_is_nonlearned=True,analytic_control=law,
        physical_gpu_executed=dev.type=='cuda',original_fitted_model=False,
        grokking_certified=False)
    a.receipt.parent.mkdir(parents=True,exist_ok=True)
    with a.receipt.open('x') as f:json.dump(r,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    return r


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--arm',choices=NAMES,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--device',choices=('gfx803','cpu'),default='gfx803')
    p.add_argument('--dtype',choices=('float32','float64'),default='float32')
    p.add_argument('--width',type=int,default=384);p.add_argument('--experts',type=int,default=4)
    p.add_argument('--steps',type=int,default=2000);p.add_argument('--report-every',type=int,default=100)
    p.add_argument('--chunk-steps',type=int,default=100);p.add_argument('--seconds',type=float,default=80)
    p.add_argument('--lr',type=float,default=.001);p.add_argument('--weight-decay',type=float,default=.1)
    p.add_argument('--resume',type=Path);p.add_argument('--resume-sha256')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True)
    print(json.dumps(run(p.parse_args()),indent=2,sort_keys=True))


if __name__=='__main__':main()
