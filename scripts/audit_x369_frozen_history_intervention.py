"""Frozen-checkpoint history intervention, geometric vs matched nuisance.

No registered owner changes. Actual second-step history intervention
with rerouting and separately fixed original selected IDs/weights.
Signs are recomputed by original selected expert execution in both
cases. This is causal for the frozen finite task, not semantic
opposition, a fitted LILA replication, or a training mechanism claim.
"""
import argparse
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.nn import functional as F
from scripts.run_x369_mod3_checkpointed import identity,make_arm,sha
from scripts.run_x369_mod3_maxcut_gpu import state_sha
from scripts.x369_r4_execution_device import resolve_device,runtime
from scripts.audit_x369_actual_leech_neural_action import matrix,transform,execute,rawbytes
from scripts.audit_x369_typed_mod3_prediction_consumer import TASK_PAIRS,TASK_LABELS,typed_mod3_prediction_consumer
from scripts.x369_grokking_geometry_trajectory import spectral


def require(ok,msg):
    if not ok:raise ValueError(msg)


def matched_nuisance(h,moved,seed):
    # Each actual expert history independently: same norm and inner
    # product with source, hence same displacement, up to dtype rounding.
    # No cross-example information or label enters the construction.
    a=h.double();b=moved.double()
    n2=a.square().sum(-1,keepdim=True)
    inner=(a*b).sum(-1,keepdim=True)
    safe=torch.where(n2>0,n2,torch.ones_like(n2))
    parallel=inner/safe
    perp2=(b.square().sum(-1,keepdim=True)-parallel.square()*n2).clamp_min(0)
    gen=torch.Generator(device='cpu');gen.manual_seed(seed)
    v=torch.randn(a.shape,generator=gen,dtype=torch.float64).to(a.device)
    v=v-(v*a).sum(-1,keepdim=True)/safe*a
    vn=v.square().sum(-1,keepdim=True).sqrt()
    require(bool(((vn>0)|(perp2==0)).all()),'degenerate matched nuisance')
    c=parallel*a+perp2.sqrt()*v/vn.clamp_min(1e-300)
    return c.to(h.dtype)


def norm_audit(h,a,b):
    def n(x):return x.double().square().sum(-1)
    return dict(max_squared_norm_difference=float((n(a)-n(b)).abs().max()),
        max_squared_displacement_difference=float((n(a-h)-n(b-h)).abs().max()))


def fixed(m,x,h,ids,w):
    idx=ids.unsqueeze(-1).expand(-1,-1,m.history.d_model)
    y,u,s=m.history.execute_selected(x,ids,h.gather(1,idx),w,train_sign=False)
    return y,h.scatter(1,idx,u),ids,s,w


def consumer(m,x,out,base):
    y,_,ids,signs,w=out
    logits=m.readout(y+x)
    lp=F.log_softmax(logits.double(),dim=-1)
    bp=F.log_softmax(base.double(),dim=-1)
    labels=torch.tensor(TASK_LABELS,device=x.device)
    return dict(audit=typed_mod3_prediction_consumer(logits),
        ce_train=float(F.nll_loss(lp[:8],labels[:8])),ce_full=float(F.nll_loss(lp,labels)),
        KL_from_baseline=float((bp.exp()*(bp-lp)).sum(-1).clamp_min(0).mean()),
        predictions=logits.argmax(-1).tolist(),ids=ids.tolist(),signs=signs.tolist(),weights=w.tolist(),
        logit_spectrum=spectral(logits))


def probe(m,selector,R,pairs,seeds):
    with torch.no_grad():
        h=m.embed.weight.new_zeros(len(pairs),m.history.num_experts,m.history.d_model)
        x0=m.embed(pairs[:,0]);_,h,_,_,_=execute(m,selector,x0,h)
        x=m.embed(pairs[:,1]);base=execute(m,selector,x,h)
        logits=m.readout(base[0]+x)
        require(torch.equal(rawbytes(logits),rawbytes(m(pairs,selector))),
                'original two-step source replay')
        rows={}
        def test(v):
            return dict(reroute=consumer(m,x,execute(m,selector,x,v),logits),
                fixed_ids_weights=consumer(m,x,fixed(m,x,v,base[2],base[4]),logits))
        rows['baseline']=test(h)
        rows['zero_history']=test(torch.zeros_like(h))
        rows['antipode']=test(-h)
        geo=transform(h,R)
        rows['literal_leech']=test(geo)
        controls=[]
        for seed in seeds:
            n=matched_nuisance(h,geo,seed)
            controls.append(dict(seed=seed,matching=norm_audit(h,geo,n),result=test(n)))
        return dict(interventions=rows,matched_nuisance=controls,actual_history_spectrum=spectral(h),
            original_source_replay_exact=True,signs_recomputed=True,
            no_cross_example_intervention=True,training_or_checkpoint_selection=False,
            semantic_opposition_inferred=False,causal_grokking_mechanism_certified=False)


def run(a):
    dev=resolve_device(a.device)
    require(sha(a.checkpoint)==a.checkpoint_sha256 and sha(a.lift)==a.lift_sha256,'pinned actual sources')
    p=torch.load(a.checkpoint,map_location='cpu',weights_only=True);c=p['contract']
    spec=SimpleNamespace(arm=c['arm'],seed=c['seed'],width=c['width'],experts=c['experts'],
        dtype=c['dtype'],steps=c['horizon'],report_every=c['report_every'],lr=c['lr'],weight_decay=c['weight_decay'])
    require(identity(spec,dev)==c,'registered source changed')
    m,selector,learner=make_arm(spec,dev);require(learner,'recurrent learner required')
    m.load_state_dict(p['model'],strict=True);m.eval()
    require(state_sha(m)==p['model_sha256'],'actual model identity')
    lift=json.loads(a.lift.read_text())
    require(lift['status']=='actual_Gx0_real_Leech_lift' and lift['forward']['quotient_basis_checks']==24,
            'actual real source action')
    R=matrix(lift['forward']['real_matrix'],dev)
    seeds=(3701,3702,3703,3704)
    r=probe(m,selector,R,torch.tensor(TASK_PAIRS,device=dev,dtype=torch.long),seeds)
    require(state_sha(m)==p['model_sha256'],'intervention mutated original model')
    return dict(status='actual_frozen_history_causal_intervention',checkpoint_sha256=a.checkpoint_sha256,
        model_sha256=p['model_sha256'],arm=c['arm'],seed=c['seed'],step=p['completed_steps'],
        lift_sha256=a.lift_sha256,source_word=lift['word'],runtime=runtime(dev),
        nuisance_seeds=list(seeds),result=r)


def main():
    p=argparse.ArgumentParser()
    for key in ('checkpoint','lift'):
        p.add_argument('--'+key,type=Path,required=True);p.add_argument('--'+key+'-sha256',required=True)
    p.add_argument('--device',choices=('gfx803','cpu'),default='gfx803')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=run(a);a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(r,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    print(r['status'],r['arm'],r['step'])

if __name__=='__main__':main()
