"""Independent ordered checkpoint replay; public task only, no training or F5."""
import argparse,json
from pathlib import Path
import torch
from scripts.x369_ordered_checkpoint import fingerprint,load_checkpoint
from scripts.run_x369_fair_ordered_gpu import FairLearner
from models.geometric_fair_residual import fair_arm
from scripts.train_x369_ordered_action import records,final_action_audit
from scripts.audit_x369_reachable_bins import audit_reachable_bins

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--sha256',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--quantum',type=float,default=.125);p.add_argument('--contexts',type=int,default=8);p.add_argument('--horizon',type=int,default=2);a=p.parse_args()
    if a.output.exists():raise FileExistsError('never overwrite replay receipt')
    if not torch.cuda.is_available():raise RuntimeError('GPU required; no CPU fallback')
    payload=load_checkpoint(a.checkpoint,a.sha256,fingerprint(Path(__file__).resolve().parents[1]));m=payload['metadata'];dev=torch.device('cuda');dt=torch.float32
    if (m.get('fixed_horizon_completed') is not True or m.get('public_task_only') is not True or m.get('dtype')!='float32' or m.get('device')!='cuda' or m.get('torch')!=str(torch.__version__) or m.get('hip')!=torch.version.hip or m.get('arch')!=getattr(torch.cuda.get_device_properties(dev),'gcnArchName','')):
        raise ValueError('requires matching recorded runtime and completed public horizon metadata')
    model=FairLearner(m['width'],m['experts'],m['paired'],'dense').to(dev,dt)
    original=model.history.source if m['paired'] else model.history
    replacement=fair_arm(original,m['mode'],m['null'],m['ternary'],m['seed'])
    if m['paired']:model.history.source=replacement
    else:model.history=replacement
    model.load_state_dict(payload['state'],strict=True);model.eval();rows=list(records())
    result=dict(kind='ordered_checkpoint_replay',sha256=a.sha256,metadata=m,output_action=final_action_audit(model,rows,dev,dt),reachable_bins=audit_reachable_bins(model,rows,dev,dt,a.quantum,a.contexts,a.horizon),fitted_provenance_independently_authenticated=False,universal_closure_certified=False)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result,allow_nan=False))
if __name__=='__main__':main()
