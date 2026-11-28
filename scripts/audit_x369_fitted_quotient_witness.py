"""Minimal collision discriminator for the ORIGINAL ordered checkpoint.
Anatolii Kornienko: original Leech-LILA; DASHI quotient extension.
No training, F5, checkpoint migration, or universal quotient promotion.
"""
from collections import defaultdict
from itertools import product
import argparse, hashlib, json
from pathlib import Path
import torch


def row_keys(x):
    """Byte equality, including signed zero, in the declared stored dtype."""
    if not torch.isfinite(x).all():
        raise ValueError('nonfinite observation')
    x=x.detach().cpu().contiguous()
    return [r.numpy().tobytes() for r in x.reshape(len(x),-1).view(torch.uint8)]


def representatives(keys):
    groups=defaultdict(list)
    for i,k in enumerate(keys): groups[k].append(i)
    return [(g[0],j) for g in groups.values() for j in g[1:]]


def candidate_keys(h, quantum):
    from scripts.audit_x369_reachable_bins import code_bins
    flat=h.flatten(1)
    result={'grid':code_bins(flat,quantum)}
    if h.ndim==4 and h.shape[2]==2:
        result['mean']=row_keys((h[:,:,0]+h[:,:,1])/2)
        a,b=row_keys(h[:,:,0]),row_keys(h[:,:,1])
        # Exact global branch-exchange quotient; no independent per-expert swap.
        result['unmarked_pair']=[tuple(sorted((x,y))) for x,y in zip(a,b)]
    return result


def discriminate(codes, labels, observe, horizon, states=None):
    """First reproducible separator per candidate; no quadratic pair list.
    A star in each collision class suffices by transitivity of equality.
    """
    if not 1<=horizon<=3: raise ValueError('invalid horizon')
    pairs={k:[(i,j) for i,j in representatives(v)
              if states is None or states[i]!=states[j]] for k,v in codes.items()}
    result={k:dict(states=len(codes[k]),classes=len(set(codes[k])),
                   representative_pairs=len(v),witness=None) for k,v in pairs.items()}
    live={k for k,v in pairs.items() if v}; checked=0
    for depth in range(1,horizon+1):
        for word in product(labels,repeat=depth):
            if not live: break
            obs={k:row_keys(v) for k,v in observe(word).items()};checked+=1
            hits={}
            for name in sorted(live):
                for i,j in pairs[name]:
                    fields=[k for k,v in obs.items() if v[i]!=v[j]]
                    if fields:
                        hits[name]=dict(i=i,j=j,word=list(word),fields=fields)
                        break
            if hits:
                again={k:row_keys(v) for k,v in observe(word).items()}
                if again!=obs: raise RuntimeError('separator replay not byte-identical')
                for name,w in hits.items(): result[name]['witness']=w;live.remove(name)
        if not live: break
    return dict(candidates=result,words_checked=checked,universal_closure_certified=False)


@torch.no_grad()
def main():
    from scripts.x369_ordered_checkpoint import fingerprint,load_checkpoint,trace_probe
    from scripts.run_x369_fair_ordered_gpu import FairLearner
    from models.geometric_fair_residual import fair_arm
    from scripts.train_x369_ordered_action import records,batch,encode_inputs
    from scripts.audit_x369_joint_action import joint_records,task_digest
    from scripts.audit_x369_hidden_generators import key
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--sha256',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--contexts',type=int,default=8);p.add_argument('--horizon',type=int,default=2);p.add_argument('--quantum',type=float,default=.125);a=p.parse_args()
    if a.output.exists() or a.contexts<1 or not 1<=a.horizon<=3 or not 0<a.quantum<float('inf'):raise ValueError('invalid scope or existing receipt')
    if not torch.cuda.is_available() or not torch.version.hip or not getattr(torch.cuda.get_device_properties(0),'gcnArchName','').startswith('gfx803'):raise RuntimeError('native gfx803 HIP required')
    payload=load_checkpoint(a.checkpoint,a.sha256,fingerprint(Path(__file__).resolve().parents[1]));m=payload['metadata'];dev=torch.device('cuda');dt=torch.float32
    if (m.get('fixed_horizon_completed') is not True or m.get('public_task_only') is not True or m.get('dtype')!='float32' or m.get('device')!='cuda' or m.get('torch')!=str(torch.__version__) or m.get('hip')!=torch.version.hip or m.get('arch')!=getattr(torch.cuda.get_device_properties(dev),'gcnArchName','')):raise ValueError('requires matching recorded runtime and completed public horizon')
    model=FairLearner(m['width'],m['experts'],m['paired'],'dense').to(dev,dt)
    original=model.history.source if m['paired'] else model.history
    replacement=fair_arm(original,m['mode'],m['null'],m['ternary'],m['seed'])
    if m['paired']:model.history.source=replacement
    else:model.history=replacement
    model.load_state_dict(payload['state'],strict=True);model.eval()
    rows=list(joint_records() if m['joint_mean'] else records())
    if task_digest(rows)!=m['task_sha256'] or trace_probe(model,rows,dev,dt)!=m['trace_probe']:raise RuntimeError('task or original trace replay mismatch')
    groups=defaultdict(list)
    for r in rows:
        if r['split']=='composition':groups[key(r)].append(r)
    keys=sorted(groups);chosen=keys if len(keys)<=a.contexts else [keys[j*len(keys)//a.contexts] for j in range(a.contexts)]
    reports=[]; refuted=set()
    expected={'grid','mean','unmarked_pair'} if m['paired'] else {'grid'}
    for context in chosen:
        if refuted==expected:break
        rs=groups[context];s,o,_=batch(rs,'signed',dev,dt);_,h=model.rollout(s,o)
        codes={k:v for k,v in candidate_keys(h,a.quantum).items() if k not in refuted}
        state_keys=row_keys(h)
        def observe(word):
            ops=torch.tensor([encode_inputs(dict(r,word=''.join(word)),'signed')[1] for r in rs],device=dev,dtype=dt)
            z=h.clone()
            for t in range(len(word)):y,z,ids,signs,weights=model.advance(ops[:,t],z)
            obs=dict(output=y,ids=ids,signs=signs,weights=weights)
            obs.update({f'logits_{j}':v for j,v in enumerate(model.consume(y,z))})
            return obs
        report=discriminate(codes,tuple('abjn'),observe,a.horizon,state_keys)
        for name,v in report['candidates'].items():
            w=v['witness']
            if w:
                refuted.add(name)
                i,j=w['i'],w['j'];w['initial_residuals']=[rs[i]['residual'],rs[j]['residual']]
                w['history_sha256']=[hashlib.sha256(k).hexdigest() for k in (state_keys[i],state_keys[j])]
        reports.append(dict(context=context,**report))
    result=dict(kind='fitted_quotient_first_separator',checkpoint_sha256=a.sha256,metadata=m,trace_probe_equal=True,quantum=a.quantum,horizon=a.horizon,contexts_total=len(keys),contexts_sampled=len(reports),refuted=sorted(refuted),reports=reports,universal_closure_certified=False,training_steps=0)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(dict(kind='complete',contexts=len(reports),output=str(a.output))))


if __name__=='__main__':main()
