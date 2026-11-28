"""Executed CPU helper checks, not a fitted model or full runner replay."""
import tempfile
import torch
from pathlib import Path
from scripts.audit_x369_reachable_bins import code_bins,bounded_witnesses
from scripts.x369_ordered_checkpoint import save_checkpoint,load_checkpoint,fingerprint

q=torch.tensor([[0.],[0.],[1.],[1.]],dtype=torch.float64)
codes=code_bins(q,.125)
h=torch.tensor([[0.,1.],[0.,-1.],[1.,0.],[1.,0.]])
def observe(word):
    x=h.clone()
    for u in word:
        y=x[:,0].clone()
        if u=='swap':x=x.flip(-1)
    return {'output':y}
r=bounded_witnesses(codes,('hold','swap'),observe,2)
assert r['collision_pairs']==2 and r['separated_pairs']==1 and r['unseparated_pairs']==1
assert r['first_separation_depth']=={'1':0,'2':1} and r['witnesses'][0]['word']==['swap','hold']
assert bounded_witnesses([0,1,2,3],('hold',),observe,1)['collision_pairs']==0
try:code_bins(torch.tensor([[float('nan')]]),.1)
except ValueError:pass
else:raise AssertionError
print('PASS',r)
with tempfile.TemporaryDirectory() as d:
    p=Path(d);(p/'code').write_text('v1');s=fingerprint(p,['code']);m=torch.nn.Linear(3,2);x=torch.ones(1,3);y=m(x).detach();sha=save_checkpoint(p/'state.pt',m,{'step':1},s)
    n=torch.nn.Linear(3,2);n.load_state_dict(load_checkpoint(p/'state.pt',sha,s)['state'],strict=True);assert torch.equal(y,n(x))
    for fn in (lambda:save_checkpoint(p/'state.pt',m,{},s),lambda:load_checkpoint(p/'state.pt','0'*64,s),lambda:load_checkpoint(p/'state.pt',sha,{'code':'other'})):
        try:fn()
        except (ValueError,FileExistsError):pass
        else:raise AssertionError('fail closed')
print('PASS checkpoint roundtrip; overwrite, changed SHA and source mismatch rejected')
