"""L7: native C3 geometric bias discrimination, separately frozen from L6.
Supplied output action is not learned latent equivariance. No reserve access.
"""
import argparse,copy,json,time
from pathlib import Path
import torch
from torch.nn import functional as F
from scripts.train_x369_mod3_codec_gpu import Learner
from scripts.run_x369_mod3_maxcut_gpu import state_sha
from scripts.audit_x369_typed_mod3_prediction_consumer import TASK_PAIRS,TASK_LABELS,TRAIN_INDICES,HELDOUT_INDEX,typed_mod3_prediction_consumer
from scripts.x369_phase_geometry import mod3_spectrum
from scripts.x369_mod3_geometric_control import orbit_forward,training_edges,edge_loss,transport_metrics
from scripts.run_x369_phase_trajectory_gpu import snapshot,PHASE_SOURCES,schedule
from scripts.x369_ordered_checkpoint import fingerprint
from scripts.run_x369_ordered_campaign import exclusive_json,sha

ARMS=('unconstrained','train_edge_consistency','hard_c3','hard_wrong_orientation')
SOURCES=tuple(dict.fromkeys(PHASE_SOURCES+('scripts/train_x369_mod3_codec_gpu.py','scripts/run_x369_mod3_maxcut_gpu.py','scripts/audit_x369_typed_mod3_prediction_consumer.py','scripts/x369_mod3_geometric_control.py','scripts/run_x369_mod3_geometric_gpu.py','models/x369_shared_head.py','models/x369_two_channel_codec.py','models/x369_differential_backward.py')))

def predict(m,p,arm):
    if arm=='hard_c3':return orbit_forward(m,p,1)
    if arm=='hard_wrong_orientation':return orbit_forward(m,p,-1)
    return m(p)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--seeds',nargs='+',type=int,default=[41,42,43]);p.add_argument('--arms',nargs='+',choices=ARMS,default=list(ARMS));p.add_argument('--width',type=int,default=384);p.add_argument('--experts',type=int,default=4);p.add_argument('--steps',type=int,default=2000);p.add_argument('--interval',type=int,default=100);p.add_argument('--lr',type=float,default=.001);p.add_argument('--weight-decay',type=float,default=.1);p.add_argument('--edge-weight',type=float,default=1.);p.add_argument('--plan',action='store_true');a=p.parse_args()
    if len(set(a.seeds))!=len(a.seeds) or len(set(a.arms))!=len(a.arms) or a.width<6 or a.width%3 or a.experts<2 or min(a.steps,a.interval)<1 or not 0<a.lr<float('inf') or not 0<=a.weight_decay<float('inf') or not 0<a.edge_weight<float('inf'):raise ValueError('invalid scope')
    cfg={k:v for k,v in vars(a).items() if k not in ('output','plan')};cfg['schedule']=schedule(a.steps,a.interval)
    manifest=dict(kind='x369_mod3_geometric_bias',settings=cfg,task_pairs=[list(x) for x in TASK_PAIRS],train_indices=list(TRAIN_INDICES),heldout_index=HELDOUT_INDEX,supplied_output_symmetry=True,learned_latent_equivariance_certified=False,release_admitted=False)
    if a.plan:print(json.dumps(manifest,indent=2));return
    if not torch.cuda.is_available() or not torch.version.hip or not getattr(torch.cuda.get_device_properties(0),'gcnArchName','').startswith('gfx803'):raise RuntimeError('native gfx803 HIP required; no CPU fallback')
    dev=torch.device('cuda',0);dt=torch.float32;root=Path(__file__).resolve().parents[1]
    manifest.update(sources=fingerprint(root,SOURCES),runtime=dict(torch=str(torch.__version__),hip=torch.version.hip,arch=getattr(torch.cuda.get_device_properties(dev),'gcnArchName','')))
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);mp=out/'manifest.json';exclusive_json(mp,manifest)
    pairs=torch.tensor(TASK_PAIRS,device=dev,dtype=torch.long);labels=torch.tensor(TASK_LABELS,device=dev,dtype=torch.long);train=torch.tensor(TRAIN_INDICES,device=dev,dtype=torch.long)
    tp,ty=pairs[train],labels[train];edges=training_edges(tp)
    assert len(edges)==14 and HELDOUT_INDEX not in TRAIN_INDICES
    for seed in a.seeds:
        torch.manual_seed(seed);initial=Learner(a.width,a.experts).to(dev,dt)
        with torch.no_grad():initial.history.c3_residual_weight.normal_(std=.003)
        init_sha=state_sha(initial)
        for arm in a.arms:
            if fingerprint(root,SOURCES)!=manifest['sources']:raise RuntimeError('source drift')
            m=copy.deepcopy(initial);assert state_sha(m)==init_sha
            opt=torch.optim.AdamW(m.parameters(),lr=a.lr,weight_decay=a.weight_decay);d=out/f'{arm}_seed{seed}';d.mkdir()
            ident=dict(manifest_sha256=sha(mp),arm=arm,seed=seed,initial_sha256=init_sha)
            # Identical parameters at start, NOT identical output function or executed work.
            first_train=first_joint=None
            for step in range(a.steps+1):
                if step:
                    m.train();opt.zero_grad(set_to_none=True);logits=predict(m,tp,arm);ce=F.cross_entropy(logits,ty)
                    penalty=edge_loss(logits,edges) if arm=='train_edge_consistency' else ce.new_zeros(())
                    loss=ce+a.edge_weight*penalty
                    if not torch.isfinite(loss):raise RuntimeError('nonfinite training loss')
                    loss.backward();opt.step()
                if step not in cfg['schedule']:continue
                m.eval()
                with torch.no_grad():
                    z=predict(m,pairs,arm);raw=m(pairs);audit=typed_mod3_prediction_consumer(z);metrics=transport_metrics(z);raw_metrics=transport_metrics(raw)
                    # No eval-dependent training or selection. Record actual raw and wrapped outputs.
                    if metrics['intended_margin_certificate'] and not metrics['p22_margin']>0:raise RuntimeError('margin transport violation')
                    if arm=='hard_c3' and metrics['first_action_max_error']!=0:raise RuntimeError('hard C3 output transport violated')
                    if arm=='hard_wrong_orientation' and transport_metrics(z,-1)['first_action_max_error']!=0:raise RuntimeError('wrong-character transport violated')
                    if metrics['strict_all_correct'] and arm=='hard_wrong_orientation':raise RuntimeError('wrong action strict fit impossible')
                checkpoint=d/f'step_{step:08d}.pt'
                ch=snapshot(checkpoint,dict(schema=1,kind='x369_mod3_geometric_snapshot',identity=ident,step=step,model=m.state_dict(),optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),device_rng=torch.cuda.get_rng_state(dev)))
                row=dict(identity=ident,step=step,checkpoint_sha256=ch,audit=audit,transport=metrics,spectrum=mod3_spectrum(z),raw_transport=raw_metrics,raw_spectrum=mod3_spectrum(raw),logits=z.cpu().tolist(),raw_logits=raw.cpu().tolist(),release_admitted=False)
                exclusive_json(d/f'audit_{step:08d}.json',row)
                if audit['training_eight_correct'] and first_train is None:first_train=step
                if audit['full_task_correct'] and first_joint is None:first_joint=step
                print(json.dumps(dict(kind='geometric_trajectory',arm=arm,seed=seed,step=step,training=audit['training_accuracy'],heldout=audit['heldout_p22_correct'],target_fraction=row['spectrum']['target_fraction'])),flush=True)
            exclusive_json(d/'summary.json',dict(identity=ident,first_observed_train_fit=first_train,first_observed_joint_fit=first_joint,final=row,hard_action_is_supplied=arm.startswith('hard_'),grokking_established=False))
            del m,opt
    exclusive_json(out/'completion.json',dict(manifest_sha256=sha(mp),arms=len(a.seeds)*len(a.arms),release_admitted=False))
if __name__=='__main__':main()
