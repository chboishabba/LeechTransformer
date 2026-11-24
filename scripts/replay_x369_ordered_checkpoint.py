"""Independent full ordered-state acceptance on the recorded native device.
Public task only; no training, model selection, F5, or global quotient claim.
"""
import argparse,json,time
from collections import defaultdict
from pathlib import Path
import torch
from scripts.x369_ordered_checkpoint import fingerprint,load_checkpoint,trace_probe
from scripts.run_x369_fair_ordered_gpu import FairLearner,audit_hidden
from models.geometric_fair_residual import fair_arm
from scripts.train_x369_ordered_action import records,score,final_action_audit
from scripts.audit_x369_joint_action import joint_records,task_digest,audit_joint
from scripts.audit_x369_hidden_generators import audit_hidden_generators
from scripts.audit_x369_consumer_closure import audit_consumer_closure
from scripts.audit_x369_reachable_bins import audit_reachable_bins
from scripts.audit_x369_precision_guards import audit_precision_guards

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--sha256',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--quantum',type=float,default=.125);p.add_argument('--contexts',type=int,default=8);p.add_argument('--horizon',type=int,default=2);p.add_argument('--joint-contexts',type=int,default=12);p.add_argument('--pairs',type=int,default=128);p.add_argument('--precision-rows',type=int,default=256);a=p.parse_args()
    if a.output.exists():raise FileExistsError('never overwrite replay receipt')
    if not torch.cuda.is_available():raise RuntimeError('GPU required; no CPU fallback')
    if min(a.contexts,a.joint_contexts,a.pairs,a.precision_rows)<1 or not 1<=a.horizon<=3 or not 0<a.quantum<float('inf'):raise ValueError('invalid audit scope')
    payload=load_checkpoint(a.checkpoint,a.sha256,fingerprint(Path(__file__).resolve().parents[1]));m=payload['metadata'];dev=torch.device('cuda');dt=torch.float32
    if (m.get('fixed_horizon_completed') is not True or m.get('public_task_only') is not True or m.get('dtype')!='float32' or m.get('device')!='cuda' or m.get('torch')!=str(torch.__version__) or m.get('hip')!=torch.version.hip or m.get('arch')!=getattr(torch.cuda.get_device_properties(dev),'gcnArchName','')):
        raise ValueError('requires matching recorded runtime and completed public horizon metadata')
    model=FairLearner(m['width'],m['experts'],m['paired'],'dense').to(dev,dt)
    original=model.history.source if m['paired'] else model.history
    replacement=fair_arm(original,m['mode'],m['null'],m['ternary'],m['seed'])
    if m['paired']:model.history.source=replacement
    else:model.history=replacement
    model.load_state_dict(payload['state'],strict=True);model.eval()
    rows=list(joint_records() if m['joint_mean'] else records())
    if task_digest(rows)!=m['task_sha256']:raise ValueError('task manifest differs')
    observed=trace_probe(model,rows,dev,dt)
    if observed!=m['trace_probe']:raise RuntimeError('fixed recurrent trace differs after strict restore')
    print(json.dumps(dict(kind='fixed_trace_replay',**observed)),flush=True)
    fixed=[r for r in rows if r['mean']==(1,0,0)]
    groups={k:defaultdict(list) for k in ('train','composition')}
    for r in rows:groups[r['split']][len(r['word'])].append(r)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    result=dict(kind='ordered_checkpoint_replay',sha256=a.sha256,metadata=m,trace_probe_equal=True,scope=dict(quantum=a.quantum,contexts=a.contexts,horizon=a.horizon,joint_contexts=a.joint_contexts,pairs=a.pairs,precision_rows=a.precision_rows),train=score(model,groups['train'],'signed',dev,dt),composition=score(model,groups['composition'],'signed',dev,dt),fitted_provenance_independently_authenticated=False,universal_closure_certified=False)
    def record(name,fn):
        result[name]=fn()
        print(json.dumps(dict(kind='audit_complete',audit=name),allow_nan=False),flush=True)
    record('output_action',lambda:final_action_audit(model,rows,'signed',dev,dt))
    record('hidden_chart',lambda:audit_hidden(model,fixed,dev,dt))
    record('hidden_generators',lambda:audit_hidden_generators(model,fixed,dev,dt))
    record('consumer_closure',lambda:audit_consumer_closure(model,fixed,dev,dt,max_pairs=a.pairs,horizon=a.horizon))
    record('reachable_bins',lambda:audit_reachable_bins(model,rows,dev,dt,a.quantum,a.contexts,a.horizon))
    if m['joint_mean']:record('joint_action',lambda:audit_joint(model,rows,dev,dt,a.joint_contexts,a.pairs))
    record('precision_guards',lambda:audit_precision_guards(model,rows,dev,a.precision_rows))
    torch.cuda.synchronize();result['audit_seconds']=time.perf_counter()-start;result['audit_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    data=json.dumps(result,indent=2,allow_nan=False)
    with a.output.open('x') as f:f.write(data+'\n')
    print(json.dumps(dict(kind='replay_complete',output=str(a.output),sha256=a.sha256)),flush=True)
if __name__=='__main__':main()
