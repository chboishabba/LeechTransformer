"""Deterministic translation contracts, not fitted-source acceptance."""
import tempfile,json
from pathlib import Path
from itertools import product
from types import SimpleNamespace
import torch
from torch import nn
from models.geometric_fair_residual import FairResidualLinear
from models.geometric_stabilizer_codec import same_bits
from scripts.run_x369_fair_ordered_gpu import FairLearner
from scripts.x369_source_geometry import signed_seed
from scripts.x369_source_acceptance import translation_provenance,digest
from scripts.x369_translate_paired_checkpoint import (effective_dense,project_controls,Metrics,
    canonical_index,seed_weld,reopen_signed_seed,translate_controls)
from scripts.audit_x369_joint_action import MEANS


def certificate():
    for dt in (torch.float32,torch.float64):
        for ternary in (False,True):
            base=nn.Linear(6,3,dtype=dt)
            with torch.no_grad():
                base.weight.copy_(torch.arange(18,dtype=dt).reshape(3,6)/16-.5)
                base.bias.copy_(torch.tensor([-.25,0.,.25],dtype=dt))
            layer=FairResidualLinear(base,torch.eye(9,dtype=dt).reshape(9,3,3),ternary)
            with torch.no_grad():layer.coefficients.copy_(torch.arange(18,dtype=dt).reshape_as(layer.coefficients)/4-2)
            model=nn.Sequential(nn.Sequential(layer)).eval();dense,receipt=effective_dense(model)
            x=torch.arange(36,dtype=dt).reshape(6,6)/8-2
            assert same_bits(model(x),dense(x)) and same_bits(dense[0][0].weight,layer.base+layer.delta())
            assert type(model[0][0]) is FairResidualLinear and type(dense[0][0]) is nn.Linear and len(receipt)==1
    obs=[(0,8,(1,2),'a'),(0,8,(-1,4),'b'),(1,8,(3,4),'c'),(1,8,(3,4),'d')]
    m,r=project_controls(obs);assert m[0,8]==(0,3) and r[0]['squared_error']=='4' and r[0]['collision']['right']=='b' and r[1]['exact']
    x=torch.tensor([[0.,1.],[2.,3.]]);y=torch.tensor([[-0.,1.],[2.,3.]])
    m=Metrics();m.add('zero',x,y,[0,1]);r=m.fields['zero']
    assert r['bitwise_rows']==1 and r['max_abs']==0 and r['first_difference']['left']['hex']!=r['first_difference']['right']['hex']
    m.add('signed',x,x+1,[0,1]);assert m.fields['signed']['first_difference']['signed_delta']==['1','1']
    assert canonical_index(4)==canonical_index(8)==8
    seed=torch.tensor([v+(0,0,0) for v in product((-1,0,1),repeat=6)],dtype=torch.float32)
    assert same_bits(reopen_signed_seed(signed_seed(seed,2,3)),seed)
    old=nn.Module();old.width=3;old.experts=2;old.paired=True
    old.mean_seed=nn.Linear(3,6,bias=False);old.residual_seed=nn.Linear(6,6,bias=False);old.embed=nn.Linear(7,3)
    old.initial=lambda seed:FairLearner.initial(old,seed)
    with torch.no_grad():
        old.mean_seed.weight.zero_();old.mean_seed.weight[:3].copy_(torch.eye(3))
        old.residual_seed.weight.zero_();old.residual_seed.weight[:3,:3].copy_(torch.eye(3))
        old.embed.weight.zero_();old.embed.bias.copy_(torch.tensor([1.,2.,3.]))
    old.eval();new=SimpleNamespace(width=3,experts=2,control_embed=nn.Embedding(48,3))
    r=seed_weld(old,new);assert r['rank']==6 and r['bidirectional_on_seed_image'] and all(v['rank']==0 for v in r['standard_action_defects'].values())
    r=translate_controls(old,new,MEANS,'representative')
    assert len(r['raw_literals'])==10 and len(r['projected_keys'])==8 and any(not v['exact'] for v in r['projected_keys'])
    assert same_bits(new.control_embed.weight[8],old.embed.bias) and same_bits(new.control_embed.weight[4],old.embed.bias)
    rr=translate_controls(old,new,MEANS,'polar-ls')
    from fractions import Fraction
    assert all(Fraction(b['selected_squared_error'])<=Fraction(a['selected_squared_error']) for a,b in zip(r['projected_keys'],rr['projected_keys']))
    with torch.no_grad():old.residual_seed.weight.zero_()
    r=seed_weld(old,new);assert r['rank']==3 and not r['bidirectional_on_seed_image'] and len(r['lost_seed_directions'])==3
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)/'receipt.json';src={'head':'abc','files':{'a':'b'}};cfg={'width':12};execution={'shape':(2,3)};rt={'arch':'gfx803'}
        obj=dict(kind='x369_translated_source_candidate',status='translated_candidate',candidate_sha256='c',candidate_model_sha256='f',source=src,execution_contract=execution,runtime=rt,config=cfg,training_steps=0,original_new_source_training=False,optimizer_migrated=False,release_admitted=False,historical_sha256='h')
        def save(v):p.write_text(json.dumps(v));return digest(p.read_bytes())
        assert translation_provenance(p,save(obj),'c',src,cfg,'f',execution,rt)['origin']=='translated_historical_paired'
        for k,v in [('candidate_sha256','bad'),('candidate_model_sha256','bad'),('source',{}),('execution_contract',{}),('runtime',{}),('config',{}),('training_steps',1),('original_new_source_training',True),('optimizer_migrated',True),('release_admitted',True),('status','materialized_trace_mismatch')]:
            try:translation_provenance(p,save(dict(obj,**{k:v})),'c',src,cfg,'f',execution,rt)
            except ValueError:pass
            else:raise AssertionError(k)
        assert translation_provenance(None,None,'c',src,cfg,'f',execution,rt)['origin']=='unspecified_raw_state'
    print('PASS 4 effective-weight cases, exact LS and signed collision, signed-zero bytes, 729 seeds, full/rank-deficient seed welds, 10 literals/8 keys, 11 provenance tamper controls; zero training')


if __name__=='__main__':certificate()
