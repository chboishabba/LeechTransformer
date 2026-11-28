"""Checkpoint identity helpers; integrity, not authentication or proof of training."""
import hashlib,io
from pathlib import Path
import torch

SOURCES=(
 'models/history_orbit_moe.py','models/trialectic_wave_prior.py',
 'models/punctured_nine_wave_prior.py','models/e8_root_prior.py',
 'models/c3_matched_router_control.py','models/geometric_cross_triads.py',
 'models/geometric_history_adapter.py','models/geometric_reflection_fibre.py',
 'models/geometric_paired_history.py','models/geometric_fair_residual.py',
 'scripts/train_x369_ordered_action.py','scripts/audit_x369_ordered_signed_task.py',
 'scripts/run_x369_fair_ordered_gpu.py','scripts/audit_x369_hidden_generators.py',
 'scripts/audit_x369_reachable_bins.py','scripts/x369_ordered_checkpoint.py',
 'scripts/audit_x369_joint_action.py','scripts/audit_x369_precision_guards.py',
 'scripts/audit_x369_hidden_chart.py','scripts/audit_x369_ordered_action_predictions.py',
 'scripts/audit_x369_consumer_closure.py','scripts/observe_x369_r4_original_topk.py',
 'scripts/replay_x369_ordered_checkpoint.py','scripts/run_x369_ordered_campaign.py')


def fingerprint(root,paths=SOURCES):
    return {p:hashlib.sha256((Path(root)/p).read_bytes()).hexdigest() for p in paths}


def save_checkpoint(path,model,metadata,source_hashes):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload=dict(schema=1,kind='x369_ordered_final_state',metadata=dict(metadata),sources=dict(source_hashes),state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
    buf=io.BytesIO();torch.save(payload,buf);data=buf.getvalue()
    with path.open('xb') as f:f.write(data)
    return hashlib.sha256(data).hexdigest()


def load_checkpoint(path,expected_sha,source_hashes):
    data=Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest()!=expected_sha:raise ValueError('checkpoint SHA mismatch')
    p=torch.load(io.BytesIO(data),map_location='cpu',weights_only=True)
    if p.get('schema')!=1 or p.get('kind')!='x369_ordered_final_state' or p.get('sources')!=source_hashes:raise ValueError('checkpoint schema or source mismatch')
    if not isinstance(p.get('metadata'),dict) or not isinstance(p.get('state'),dict) or not all(isinstance(v,torch.Tensor) and torch.isfinite(v).all() for v in p['state'].values()):raise ValueError('invalid checkpoint tensors or metadata')
    return p

def trace_probe(model,rows,device,dtype):
    """Fixed natural rows, every step; SHA over typed full trace, not just logits."""
    from scripts.train_x369_ordered_action import batch
    from collections import defaultdict
    if model.training:raise ValueError('eval required')
    groups=defaultdict(list)
    for r in rows:groups[len(r['word'])].append(r)
    h=hashlib.sha256();count=0
    def add(t):
        t=t.detach().cpu().contiguous()
        if not torch.isfinite(t).all():raise ValueError('nonfinite probe')
        h.update((str(t.dtype)+':'+str(tuple(t.shape))+':').encode());h.update(t.numpy().tobytes())
    with torch.no_grad():
        for length,rs in sorted(groups.items()):
            # First and last rows at each length, with deterministic evenly spaced interior.
            ix=sorted(set(j*(len(rs)-1)//15 for j in range(16)));rs=[rs[j] for j in ix]
            s,o,_=batch(rs,'signed',device,dtype);state=model.initial(s);add(state)
            for t in range(length):
                y,state,ids,signs,weights=model.advance(o[:,t],state)
                for v in (y,state,ids,signs,weights):add(v)
            for v in model.consume(y,state):add(v)
            count+=len(rs)
    return dict(rows=count,sha256=h.hexdigest())
