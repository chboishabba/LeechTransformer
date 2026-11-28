"""Native frozen-checkpoint consumer intervention; no optimizer or fitted head.
Original Leech-LILA: Anatolii Kornienko; DASHI extension.
"""
import argparse,json
from pathlib import Path
import torch
from scripts.run_x369_geometry_training_gpu import configure_consumer,metrics,probe,relation_audit,groups
from scripts.run_x369_ordered_campaign import sha,exclusive_json
from scripts.x369_source_geometry import SourceGeometryOrderedLearner
from scripts.x369_source_acceptance import source_identity,execution_contract,model_digest
from scripts.audit_x369_joint_action import joint_records,select_contexts,task_digest
from scripts.x369_split_carrier import SplitCarrierLearner
from scripts.train_x369_ordered_action import batch

@torch.no_grad()
def split_reference(model,rows,device):
    ref=SplitCarrierLearner(model).eval();total=0;correct=[0]*3;ties=[0]*3;worst=0.
    for rs in groups(rows).values():
        for start in range(0,len(rs),128):
            chunk=rs[start:start+128];s,o,y=batch(chunk,'signed',device,torch.float32)
            logits,trace=ref.rollout(s,o)
            z=trace[-1].state.semantic
            target=torch.tensor([r['mean_label']+r['signed_label'] for r in chunk],device=device,dtype=z.dtype)
            if not torch.equal(z,target):raise RuntimeError('split supplied action differs from independent task')
            for j,out in enumerate(trace):
                prior=ref.initial(s).sparse if j==0 else trace[j-1].state.sparse
                active=torch.zeros(prior.shape[:2],device=device,dtype=torch.bool).scatter_(1,out.raw.ids,True)
                if not torch.equal(out.state.sparse[~active],prior[~active]):raise RuntimeError('inactive expert changed')
                zz,e,r=ref.chart(out.state)
                recovered=ref.reconstruct(zz,e,r)
                worst=max(worst,float((recovered-out.state.sparse).abs().max()))
            for i,v in enumerate(logits):
                mask=v==v.amax(-1,keepdim=True);unique=mask.sum(-1)==1
                correct[i]+=int((unique&mask.gather(1,y[:,i,None]).squeeze(1)).sum());ties[i]+=int((~unique).sum())
            total+=len(chunk)
    return dict(rows=total,unique_correct=correct,ties=ties,supplied_action=True,reconstruction_max=worst)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','published-summary','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--summary-sha256',required=True);p.add_argument('--source-head',required=True)
    a=p.parse_args()
    if not torch.cuda.is_available() or not torch.version.hip or not torch.cuda.get_device_properties(0).gcnArchName.startswith('gfx803'):raise RuntimeError('native gfx803 required')
    dev=torch.device('cuda',0);root=Path(__file__).resolve().parents[1];source=source_identity(root,a.source_head)
    if sha(a.published_summary)!=a.summary_sha256:raise ValueError('published summary digest')
    published=json.loads(a.published_summary.read_text())
    mp=a.run/'manifest.json';manifest=json.loads(mp.read_text())
    if sha(mp)!=published['manifest_sha256']:raise ValueError('manifest digest')
    rows=list(joint_records())
    if task_digest(rows)!=manifest['task_sha256']:raise ValueError('task identity')
    train=[r for r in rows if r['split']=='train'];sample=select_contexts(rows,manifest['settings']['contexts'])
    splits={k:[r for r in sample if r['split']==k] for k in ('train','composition')}
    result=[]
    for report in published['arms']:
        ident=report['identity'];arm=ident['arm'];seed=ident['seed']
        if arm not in ('fresh','warm'):continue
        step=ident['settings']['schedule'][-1]
        d=a.run/f'{arm}_{seed}';cp=d/f'step_{step:08d}.pt';rp=cp.with_suffix('.json')
        digest=sha(cp);receipt=json.loads(rp.read_text())
        if digest!=report['checkpoint_sha256'] or digest!=receipt['sha256'] or ident!=receipt['identity']:raise ValueError('checkpoint identity')
        q=torch.load(cp,map_location='cpu',weights_only=True)
        if q['identity']!=ident or q['step']!=step or q['probe']!=receipt['probe'] or q['kind']!='x369_geometry_training_snapshot':raise ValueError('snapshot identity')
        ex=q['execution'];model=SourceGeometryOrderedLearner(ex['width'],ex['experts'],expert_width=ex['expert_width']).to(dev,torch.float32).eval()
        model.history.source.history_cap=ex['source']['history_cap'];model.load_state_dict(q['model'],strict=True)
        if execution_contract(model)!=ex:raise ValueError('execution contract')
        before=model_digest(model);original_probe=probe(model,train,dev)
        affine={k:metrics(model,v,dev) for k,v in splits.items()}
        # Older unsupervised checkpoints precede projection repair. Never claim replay if changed.
        probe_equal=original_probe==q['probe']
        configure_consumer(model,'typed');model.eval()
        typed={k:metrics(model,v,dev) for k,v in splits.items()}
        relations=relation_audit(model,dev)
        reference={k:split_reference(model,v,dev) for k,v in splits.items()}
        if before!=model_digest(model):raise RuntimeError('weights mutated')
        result.append(dict(arm=arm,seed=seed,checkpoint_sha256=digest,original_probe_equal=probe_equal,affine=affine,typed=typed,typed_relations=relations,split_reference=reference,training_steps=0))
        print(json.dumps(dict(arm=arm,seed=seed,original_probe_equal=probe_equal,affine=affine,typed=typed)),flush=True)
    if source_identity(root,a.source_head)!=source:raise RuntimeError('source changed')
    exclusive_json(a.output,dict(kind='frozen_typed_consumer',source=source,summary_sha256=a.summary_sha256,arms=result,training_steps=0,release_admitted=False))
if __name__=='__main__':main()
