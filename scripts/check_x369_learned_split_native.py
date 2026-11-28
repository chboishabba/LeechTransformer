"""Native signed-generator execution and independently observed residual task.
Original Leech-LILA: Anatolii Kornienko; DASHI extension.
No optimization, no F5, no unrestricted learned-source equivalence.
"""
import argparse,json
from collections import defaultdict
from itertools import product
from pathlib import Path
import torch
from scripts.x369_learned_generator_weld import fit_generators,certify
from scripts.x369_split_carrier import LearnedGeneratorSplitCarrier,SplitState
from scripts.audit_x369_joint_action import joint_records,task_digest,MEANS
from scripts.audit_x369_ordered_signed_task import TRIADS,endpoint
from scripts.train_x369_ordered_action import batch
from scripts.x369_source_geometry import literal_controls
from scripts.x369_source_acceptance import source_identity,model_digest
from scripts.run_x369_ordered_campaign import exclusive_json
from models.geometric_stabilizer_codec import same_bits

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-head',required=True);p.add_argument('--receipt',type=Path,required=True)
    a=p.parse_args()
    if not torch.cuda.is_available() or not torch.version.hip or not torch.cuda.get_device_properties(0).gcnArchName.startswith('gfx803'):
        raise RuntimeError('native gfx803 required')
    dev=torch.device('cuda',0);root=Path(__file__).resolve().parents[1];source=source_identity(root,a.source_head)
    rows=list(joint_records());m=fit_generators(rows);cert=certify(m)
    torch.manual_seed(41);model=LearnedGeneratorSplitCarrier(m).to(dev).eval();before=model_digest(model)
    groups=defaultdict(list)
    for r in rows:groups[r['split'],len(r['word'])].append(r)
    counts=defaultdict(int);sparse_rows=0
    with torch.no_grad():
        for (split,length),rs in groups.items():
            for start in range(0,len(rs),256):
                chunk=rs[start:start+256];s,o,y=batch(chunk,'signed',dev,torch.float32)
                c,t=literal_controls(o);z=s[:,:6]
                for j in range(length):z=model.advance_semantic(z,c[:,j],t[:,j])
                expected=torch.tensor([r['mean_label']+r['signed_label'] for r in chunk],device=dev,dtype=z.dtype)
                assert torch.equal(z,expected)
                for i,v in enumerate(model.prototype(z[:,:3],z[:,3:])):
                    top=v==v.amax(-1,keepdim=True)
                    assert (top.sum(-1)==1).all() and (v.argmax(-1)==y[:,i]).all()
                counts[split]+=len(chunk)
                if start==0:
                    logits,trace=model.rollout(s,o);prior=model.initial(s)
                    for j,out in enumerate(trace):
                        active=torch.zeros(prior.sparse.shape[:2],device=dev,dtype=torch.bool).scatter_(1,out.raw.ids,True)
                        assert same_bits(out.state.sparse[~active],prior.sparse[~active])
                        assert torch.equal(out.state.semantic,model.advance_semantic(prior.semantic,c[:,j],t[:,j]))
                        zz,e,r=model.chart(out.state)
                        assert torch.allclose(model.reconstruct(zz,e,r),out.state.sparse,atol=2e-6,rtol=2e-5)
                        prior=out.state
                    assert torch.equal(prior.semantic,z);sparse_rows+=len(chunk)
            print(json.dumps(dict(split=split,length=length,rows=counts[split])),flush=True)
        # Two histories with identical semantic endpoint and distinct literal prefix.
        # A query for the first literal distinguishes them, even after common suffix.
        g=(1,1,1,1);seed=(1,0,0);d=(1,-1,0)
        assert endpoint('ab',g,seed,d)==endpoint('ba',g,seed,d)
        assert endpoint('abj',g,seed,d)==endpoint('baj',g,seed,d)
    assert before==model_digest(model) and source==source_identity(root,a.source_head)
    exclusive_json(a.receipt,dict(kind='learned_split_native',source=source,task_sha256=task_digest(rows),certificate=cert,rows=dict(counts),sparse_integration_rows=sparse_rows,residual_separator=dict(words=['abj','baj'],query='first_literal',answers=['a','b'],semantic_equal=True),training_steps=0,release_admitted=False))
if __name__=='__main__':main()
