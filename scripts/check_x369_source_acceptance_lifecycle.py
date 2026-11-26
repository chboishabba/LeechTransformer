"""Non-inference source acceptance lifecycle regression. No checkpoint or GPU."""
import tempfile,subprocess,copy
from pathlib import Path
from types import SimpleNamespace as S
import torch
from torch import nn
from scripts.x369_source_acceptance import source_identity,stage_plan,execution_contract,run_stage


def rejects(f):
    try:f()
    except ValueError:return
    raise AssertionError('invalid accepted')


def certificate():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);(root/'models').mkdir();(root/'scripts').mkdir()
        p=root/'models'/'a.py';p.write_text('x=1\n');(root/'scripts'/'b.py').write_text('y=2\n')
        def git(*args):return subprocess.check_output(['git','-C',tmp,*args],text=True).strip()
        git('init','-q');git('add','.');git('-c','user.name=Audit','-c','user.email=audit@example.com','commit','-qm','fixture');head=git('rev-parse','HEAD')
        assert len(source_identity(root,head)['files'])==2
        q=root/'scripts'/'foreign.py';q.write_text('z=3\n');rejects(lambda:source_identity(root,head));q.unlink()
        p.write_text('x=3\n');rejects(lambda:source_identity(root,head));p.write_text('x=1\n')
        rejects(lambda:source_identity(root,'wrong'))
        assert len(source_identity(root,head)['files'])==2
    a=S(hom_contexts=0,graph_budget=0,future_budget=0,interval_bounds=None,factorial_contexts=0,previous_source_receipt=None,resource_rows=0,only_stages=None)
    assert stage_plan(a)==('packed_trace','linear_decoder','kernel')
    a.resource_rows=4;a.future_budget=200;a.only_stages=['resources','future_consumer'];assert stage_plan(a)==('future_consumer','resources')
    a.only_stages=['resources','resources'];rejects(lambda:stage_plan(a))
    a.only_stages=['factorial'];rejects(lambda:stage_plan(a))
    a.factorial_contexts=1;a.previous_source_receipt='prior.json';a.only_stages=['cross_checkpoint'];rejects(lambda:stage_plan(a))
    a.only_stages=['cross_checkpoint','factorial'];assert stage_plan(a)==('factorial','cross_checkpoint')
    m=nn.Module();m.width=12;m.experts=4;m.expert_width=12;m.typed_consumer=False;m.history=nn.Module();m.history.paired=True;m.history.source=nn.Module();m.history.source.history_cap=1.;m.register_buffer('label_perm_0',torch.arange(3),persistent=False);m.eval()
    base=execution_contract(m);assert not m.state_dict()
    m.history.source.history_cap=2.;assert execution_contract(m)!=base;m.history.source.history_cap=1.
    m.label_perm_0[0]=2;assert execution_contract(m)!=base;m.label_perm_0[0]=0
    m.history.train();assert execution_contract(m)!=base;m.eval();assert execution_contract(m)==base
    r=dict(requested_stages=('graph','kernel'),stage_lifecycle={'graph':'not_started','kernel':'not_started'},stages={},complete=False,execution_complete=False,release_admitted=False)
    snapshots=[];save=lambda:snapshots.append(copy.deepcopy(r))
    run_stage(r,'resources',lambda:(_ for _ in ()).throw(AssertionError('skipped stage called')),save)
    assert not snapshots
    run_stage(r,'graph',lambda:dict(status='time_budget',universal_weld=False),save)
    assert r['stage_lifecycle']['graph']=='returned' and not r['complete'] and not r['release_admitted']
    rejects(lambda:run_stage(r,'graph',lambda:None,save))
    def fail():
        r['stages']['kernel']={'blocks':{'mean_dc':{'status':'unsat'}}};save();raise RuntimeError('later block failed')
    try:run_stage(r,'kernel',fail,save)
    except RuntimeError:pass
    else:raise AssertionError('error swallowed')
    assert r['stage_lifecycle']['kernel']=='error' and r['stages']['kernel']['partial']['blocks']['mean_dc']['status']=='unsat'
    assert r['stages']['graph']['status']=='time_budget' and not r['execution_complete']
    assert snapshots[-1]==r and any(s['stage_lifecycle']['kernel']=='running' for s in snapshots)
    print('PASS source clean/dirty/untracked/wrong-head, 6 stage-plan cases, 3 execution-contract mutations, skipped/bounded/duplicate/failure/partial/prior receipt controls')




def admission_certificate():
    from scripts.x369_source_acceptance import read_checkpoint,digest,parse_args
    from unittest.mock import patch
    a=parse_args(['--checkpoint','not-read.pt','--sha256','0'*64,'--source-head','test','--output','not-created.json','--only-stages','admission','--expert-width','17','--plan'])
    assert stage_plan(a)==('admission',) and a.expert_width==17 and a.plan
    a.resource_rows=4;a.only_stages=['resources','admission'];assert stage_plan(a)==('admission','resources')
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)/'fixture.pt'
        def write(obj):
            torch.save(obj,p);return digest(p.read_bytes())
        d=write({'weight':torch.tensor([0.,-0.,2.],dtype=torch.float32)})
        assert torch.equal(read_checkpoint(p,d,torch.float32)['weight'].view(torch.int32),torch.tensor([0.,-0.,2.]).view(torch.int32))
        with patch('torch.load',side_effect=AssertionError('deserialization before hash')):
            rejects(lambda:read_checkpoint(p,'0'*64,torch.float32))
            rejects(lambda:read_checkpoint(p,'bad',torch.float32))
        for obj in ({}, {'weight':torch.tensor([1.],dtype=torch.float64)}, {'weight':torch.tensor([float('nan')])}, {'model':{'weight':torch.ones(1)}}, {1:torch.ones(1)}, {'weight':torch.sparse_coo_tensor([[0]],[1.],(2,))}):
            d=write(obj);rejects(lambda:read_checkpoint(p,d,torch.float32))
        d=write({'weight':torch.ones(1)});rejects(lambda:read_checkpoint(p,d,torch.float64))
    print('PASS admission-only/order, exact signed zero, wrong hash/malformed hash before deserialization, 7 invalid payload controls')


if __name__=='__main__':
    certificate()
    admission_certificate()
